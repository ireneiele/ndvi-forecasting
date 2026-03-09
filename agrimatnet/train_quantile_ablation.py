"""
Training script probabilistico per AgriMatNet basato su quantili.
"""
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm.auto import tqdm

from dataset_builder.torch_dataset import CacheTimeSeriesDataset
from agrimatnet.train_utils import str2bool, set_seed, move_to_device, masked_mse
from agrimatnet.model_quantile import AgriMatNetQuantile, quantile_loss

# torch.cuda.set_per_process_memory_fraction(0.25, 0) 


def parse_quantiles(value):
    parts = value.split(",")
    quantiles = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            tau = float(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Quantile non numerico: {part}") from exc
        quantiles.append(tau)
    if not quantiles:
        raise argparse.ArgumentTypeError("Specificare almeno un quantile.")
    quantiles_sorted = sorted(quantiles)
    if quantiles_sorted != quantiles:
        raise argparse.ArgumentTypeError("I quantili devono essere forniti in ordine crescente.")
    if len(set(quantiles)) != len(quantiles):
        raise argparse.ArgumentTypeError("I quantili devono essere tutti distinti.")
    for tau in quantiles:
        if not (0.0 < tau < 1.0):
            raise argparse.ArgumentTypeError(f"Quantile {tau} non valido. Deve essere compreso tra 0 e 1.")
    return quantiles


def collate_variable(batch):
    """
    Padding delle sequenze e costruzione maschere per Transformer, includendo i delta_days per pesare le loss.
    """
    list_keys = {
        "history_timestamps",
        "future_timestamps",
        "target_timestamps",
        "climate",
        "area",
        "source",
        "history_start",
        "history_end",
        "future_start",
        "future_end",
    }

    collated = {}
    for key in list_keys:
        collated[key] = [item[key] for item in batch]

    history_lengths = [item["history"].shape[0] for item in batch]
    future_lengths = [item["future"].shape[0] for item in batch]
    max_history = max(history_lengths)
    max_future = max(future_lengths)

    padded_history = []
    padded_history_mask = []
    padded_future = []
    padded_future_mask = []
    padded_future_noise = []
    target_delta_days = []

    for item in batch:
        history = item["history"]
        history_mask = item["history_mask"]
        future = item["future"]
        future_mask = item["future_mask"]
        future_noise = item["future_noise"]

        # calcoliamo la differenza tra l'ultimo history_timestamps e i 5 target_timestamps da predirre, questo servirà per pesare l'mse
        history_end = np.datetime64(item["history_timestamps"][-1])
        target_ts = np.array(item["target_timestamps"], dtype="datetime64[ns]")
        delta_days = (target_ts - history_end).astype("timedelta64[D]").astype(np.float32)
        target_delta_days.append(torch.tensor(delta_days, dtype=torch.float32))

        pad_rows_history = max_history - history.shape[0]
        if pad_rows_history > 0:
            history = torch.nn.functional.pad(history, (0, 0, 0, pad_rows_history))
            history_mask = torch.nn.functional.pad(history_mask, (0, 0, 0, pad_rows_history), value=True)
        padded_history.append(history)
        padded_history_mask.append(history_mask)

        pad_rows_future = max_future - future.shape[0]
        if pad_rows_future > 0:
            future = torch.nn.functional.pad(future, (0, 0, 0, pad_rows_future))
            future_mask = torch.nn.functional.pad(future_mask, (0, 0, 0, pad_rows_future), value=True)
            future_noise = torch.nn.functional.pad(future_noise, (0, 0, 0, pad_rows_future))
        padded_future.append(future)
        padded_future_mask.append(future_mask)
        padded_future_noise.append(future_noise)

    collated["history"] = torch.stack(padded_history)
    collated["history_mask"] = torch.stack(padded_history_mask)
    collated["history_pad_mask"] = torch.tensor(
        [[i >= length for i in range(max_history)] for length in history_lengths],
        dtype=torch.bool,
    )

    collated["future"] = torch.stack(padded_future)
    collated["future_mask"] = torch.stack(padded_future_mask)
    collated["future_noise"] = torch.stack(padded_future_noise)
    collated["future_pad_mask"] = torch.tensor(
        [[i >= length for i in range(max_future)] for length in future_lengths],
        dtype=torch.bool,
    )

    collated["future_target_positions"] = torch.stack([item["future_target_positions"] for item in batch])
    collated["target"] = torch.stack([item["target"] for item in batch])
    collated["target_mask"] = torch.stack([item["target_mask"] for item in batch])
    collated["target_delta_days"] = torch.stack(target_delta_days)

    return collated

# funzioni aggiunte per l'ablation study:

def mask_columns(tensor, mask, indices):
    # indices è la lista di colonne da spegnere (covariate o target storico) su cui mask_columns imposta i valori a 0 e la maschera a True.
    if not indices:
        return
    tensor[..., indices] = 0.0
    # mask[..., indices] = True


def apply_ablation(batch, ablation_cfg, feature_idx):
    """
    Spegne selettivamente covariate future, covariate storiche e target storico.
    """
    cov_idxs = feature_idx["covariates"] # Indice che fa riferimento a tutte le covariate da spegnere
    def safe_mask(tensor, mask_tensor):
        feat_dim = tensor.shape[-1]
        bad = [i for i in cov_idxs if i >= feat_dim or i < -feat_dim]
        if bad:
            raise ValueError(f"Indice covariata fuori range: {bad} con feat_dim={feat_dim}")
        mask_columns(tensor, mask_tensor, cov_idxs)

    if ablation_cfg["future_covariates_off"]:
        safe_mask(batch["future"], batch["future_mask"])
        if "future_noise" in batch:
            noise_dim = batch["future_noise"].shape[-1]
            noise_cov_idxs = [i for i in cov_idxs if -noise_dim <= i < noise_dim] # Solo al rumore
            batch["future_noise"][..., noise_cov_idxs] = 0.0

    if ablation_cfg["history_covariates_off"]:
        safe_mask(batch["history"], batch["history_mask"])

    if ablation_cfg["target_history_off"]:
        target_idx = -1
        mask_columns(batch["history"], batch["history_mask"], [target_idx])
        # mask_columns(batch["future"], batch["future_mask"], [target_idx])
        # if "future_noise" in batch:
        #    batch["future_noise"][..., target_idx] = 0.0

    return batch

def build_feature_index(feature_names):
    """
    Ricava gli indici di colonna per covariate e target storico a partire dal metadata cache.
    """
    if not feature_names:
        raise ValueError("feature_names mancante dal dataset.")
    target_history_idx = len(feature_names) - 1
    covariate_indices = list(range(target_history_idx))
    return {"target_history": target_history_idx, "covariates": covariate_indices}

# fine aggiunta funzioni l'ablation study


def masked_time_weighted_mse(preds, targets, mask, delta_days, alpha):
    weights = 1.0 / (1.0 + alpha * delta_days)
    weights = weights * (~mask).float()
    denom = weights.sum().clamp(min=1e-8)
    return ((preds - targets) ** 2 * weights).sum() / denom


def train_epoch(model, loader, optimizer, device, quantiles, ablation_cfg, feature_idx, time_weight_alpha=None):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_items = 0
    progress = tqdm(loader, desc="Train", leave=False)
    for batch in progress:
        batch = move_to_device(batch, device)
        batch = apply_ablation(batch, ablation_cfg, feature_idx)
        preds = model(batch)
        targets = batch["target"]
        mask = batch["target_mask"]
        delta_days = None
        weights = None
        if time_weight_alpha is not None:
            delta_days = batch["target_delta_days"]
            weights = 1.0 / (1.0 + time_weight_alpha * delta_days)
        loss = quantile_loss(preds, targets, mask, quantiles, weights=weights)
        median_idx = len(quantiles) // 2
        if time_weight_alpha is not None:
            mse = masked_time_weighted_mse(
                preds[..., median_idx],
                targets,
                mask,
                delta_days,
                alpha=time_weight_alpha,
            )
        else:
            mse = masked_mse(preds[..., median_idx], targets, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if weights is not None:
            batch_items = weights.sum().item()
        else:
            batch_items = (~mask).sum().item()
        total_loss += loss.item() * batch_items
        total_mse += mse.item() * batch_items
        total_items += batch_items
        progress.set_postfix({"loss": loss.item(), "mse": mse.item()})
    denom = max(total_items, 1)
    return (total_loss / denom, total_mse / denom)


@torch.no_grad()
def evaluate(model, loader, device, quantiles, ablation_cfg, feature_idx, time_weight_alpha=None):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_items = 0
    progress = tqdm(loader, desc="Validate", leave=False)
    for batch in progress:
        batch = move_to_device(batch, device)
        batch = apply_ablation(batch, ablation_cfg, feature_idx)
        preds = model(batch)
        targets = batch["target"]
        mask = batch["target_mask"]
        delta_days = None
        weights = None
        if time_weight_alpha is not None:
            delta_days = batch["target_delta_days"]
            weights = 1.0 / (1.0 + time_weight_alpha * delta_days)
        loss = quantile_loss(preds, targets, mask, quantiles, weights=weights)
        median_idx = len(quantiles) // 2
        if time_weight_alpha is not None:
            mse = masked_time_weighted_mse(
                preds[..., median_idx],
                targets,
                mask,
                delta_days,
                alpha=time_weight_alpha,
            )
        else:
            mse = masked_mse(preds[..., median_idx], targets, mask)

        if weights is not None:
            batch_items = weights.sum().item()
        else:
            batch_items = (~mask).sum().item()
        total_loss += loss.item() * batch_items
        total_mse += mse.item() * batch_items
        total_items += batch_items
        progress.set_postfix({"loss": loss.item(), "mse": mse.item()})
    denom = max(total_items, 1)
    return (total_loss / denom, total_mse / denom)


def parse_args():
    parser = argparse.ArgumentParser(description="Training probabilistico di AgriMatNet (quantile).")
    parser.add_argument("--cache-root", default="timeSeries/cache/train", help="Directory con le cache npz/json.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Percorso checkpoint da cui ripartire (default: checkpoint_last_quantile.pth).")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--saving-frequency",
        type=int,
        default=1,
        help="Salva i pesi model_epoch ogni N epoche (<=0 per disabilitare).",
    )
    parser.add_argument(
        "--min-crop-pixels",
        type=float,
        default=0.0,
        help="Percentuale minima di crop_pixels richiesta per includere un minicubo (0 disabilita il filtro).",
    )
    parser.add_argument(
        "--data-summary-path",
        type=str,
        default="timeSeries/dataSummary.csv",
        help="Percorso del file dataSummary.csv da cui leggere crop_pixels.",
    )
    parser.add_argument(
        "--quantiles",
        type=parse_quantiles,
        default=[0.1, 0.5, 0.9],
        help="Quantili separati da virgola (es. 0.1,0.5,0.9).",
    )
    parser.add_argument(
        "--apply-scaling",
        type=str2bool,
        default=True,
        help="Abilita l'uso dello scaler salvato nella cache (true/false).",
    )
    parser.add_argument(
        "--feature-engineering",
        type=str2bool,
        default=False,
        help="Abilita le feature ingegnerizzate (cumuli/contatori) calcolate prima dello scaling.",
    )
    parser.add_argument(
        "--discretize-target",
        type=str2bool,
        default=False,
        help="Discretizza il target NDVI in classi (usato per esperimenti di classificazione).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="quantile_experiment",
        help="Nome esperimento usato per salvare checkpoint e log.",
    )
    parser.add_argument(
        "--lr-reduce-on-plateau",
        type=str2bool,
        default=False,
        help="Abilita il lr scheduler ReduceLROnPlateau (true/false).",
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.2,
        help="Fattore di riduzione del LR quando scatta ReduceLROnPlateau.",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="Numero di epoche senza miglioramento prima di ridurre il LR.",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=5e-5,
        help="Valore minimo del learning rate consentito dal scheduler.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=128,
        help="Dimensione delle embedding interne (d_model) dei Transformer encoder.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Numero di layer in ciascun Transformer encoder.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Numero di teste di attenzione multi-head.",
    )
    parser.add_argument(
        "--dim-feedforward",
        type=int,
        default=256,
        help="Dimensione del feed-forward interno agli encoder.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout applicato in AgriMatNet (positional encoding e testa).",
    )
    parser.add_argument(
        "--time-weight-alpha",
        type=float,
        default=None,
        help="Peso temporale per il MSE: alpha in 1/(1+alpha*delta_days). None per disattivare.",
    )
    parser.add_argument(
        "--ablate-future-covariates",
        type=str2bool,
        default=False,
        help="Spegne le covariate future (meteo/temporali) prima del forward.",
    )
    parser.add_argument(
        "--ablate-history-covariates",
        type=str2bool,
        default=False,
        help="Spegne le covariate storiche (meteo/temporali) prima del forward.",
    )
    parser.add_argument(
        "--ablate-target-history",
        type=str2bool,
        default=False,
        help="Spegne la colonna di target storico in history e future.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("===== AgriMatNet Quantile Training Configuration =====")
    for key, value in sorted(vars(args).items()):
        print(f"{key}: {value}")
    print("======================================================")
    set_seed(args.seed)

    dataset = CacheTimeSeriesDataset(
        cache_dir=args.cache_root,
        apply_scaling=args.apply_scaling,
        data_summary_path=args.data_summary_path,
        min_crop_pixels=args.min_crop_pixels,
        feature_engineering=args.feature_engineering,
        discretize_target=args.discretize_target,
    )

    feature_idx = build_feature_index(dataset.feature_names)
    ablation_cfg = {
        "future_covariates_off": args.ablate_future_covariates,
        "history_covariates_off": args.ablate_history_covariates,
        "target_history_off": args.ablate_target_history,
    }
    time_weight_alpha = args.time_weight_alpha

    train_size = int(len(dataset) * args.train_split)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable,
        num_workers=0,
    )

    input_dim = len(dataset.feature_names)
    model = AgriMatNetQuantile(
        input_dim=input_dim,
        quantiles=args.quantiles,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_reduce_on_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.lr_min,
        )

    output_dir = Path("checkpoints") / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = output_dir / "model_best.pth"
    resume_path = Path(args.resume_from) if args.resume_from else output_dir / "checkpoint_last.pth"

    best_val_loss = float("inf")
    val_history = []
    start_epoch = 1

    if resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        val_history = list(checkpoint.get("val_history", val_history))
        start_epoch = checkpoint["epoch"] + 1
        print(f"Ripartenza da {resume_path} (epoch {checkpoint['epoch']})")
    elif args.resume_from:
        print(f"Checkpoint {resume_path} non trovato, training da zero.")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_mse = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.quantiles,
            ablation_cfg,
            feature_idx,
            time_weight_alpha=time_weight_alpha,
        )
        val_loss, val_mse = evaluate(
            model,
            val_loader,
            device,
            args.quantiles,
            ablation_cfg,
            feature_idx,
            time_weight_alpha=time_weight_alpha,
        )
        print(
            f"Epoch {epoch:02d} | train_pinball={train_loss:.6f} | val_pinball={val_loss:.6f} "
            f"| train_mse={train_mse:.6f} | val_mse={val_mse:.6f}"
        )
        val_history.append(
            {
                "epoch": epoch,
                "train_pinball": float(train_loss),
                "val_pinball": float(val_loss),
                "train_mse": float(train_mse),
                "val_mse": float(val_mse),
            }
        )

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "val_history": list(val_history),
            "args": {
                **vars(args),
                "quantiles": list(args.quantiles),
            },
        }
        if scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = scheduler.state_dict()

        torch.save(checkpoint_data, output_dir / "checkpoint_last.pth")
        if args.saving_frequency > 0 and epoch % args.saving_frequency == 0:
            torch.save(model.state_dict(), output_dir / f"model_epoch{epoch:02d}.pth")
        if is_best:
            torch.save(checkpoint_data, output_dir / "checkpoint_best.pth")
            torch.save(model.state_dict(), best_model_path)

        if scheduler is not None:
            scheduler.step(val_loss)

    final_weights = output_dir / "model_final.pth"
    torch.save(model.state_dict(), final_weights)
    history_path = output_dir / "val_losses.csv"
    with history_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["epoch", "train_pinball", "val_pinball", "train_mse", "val_mse"])
        for entry in val_history:
            writer.writerow(
                [
                    entry["epoch"],
                    entry["train_pinball"],
                    entry["val_pinball"],
                    entry["train_mse"],
                    entry["val_mse"],
                ]
            )


if __name__ == "__main__":
    main()

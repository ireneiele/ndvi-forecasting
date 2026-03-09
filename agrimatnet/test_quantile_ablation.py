



"""
Script di test per il modello quantile di AgriMatNet.
Calcola la pinball loss media e altri indicatori, salvando anche un grafico delle bande di quantile.
"""
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_builder.torch_dataset import CacheTimeSeriesDataset
from agrimatnet.train_utils import move_to_device, str2bool
from agrimatnet.model_quantile import AgriMatNetQuantile
from agrimatnet.train_quantile_ablation import collate_variable, parse_quantiles

# funzioni di ablation
def mask_columns(tensor, mask, indices):
    if not indices:
        return
    tensor[..., indices] = 0.0


def apply_ablation(batch, ablation_cfg, feature_idx):
    if ablation_cfg["future_covariates_off"]:
        mask_columns(batch["future"], batch["future_mask"], feature_idx["covariates"])
    if ablation_cfg["history_covariates_off"]:
        mask_columns(batch["history"], batch["history_mask"], feature_idx["covariates"])
    if ablation_cfg["target_history_off"]:
        target_idx = feature_idx["target_history"]
        mask_columns(batch["history"], batch["history_mask"], [target_idx])
        mask_columns(batch["future"], batch["future_mask"], [target_idx])
    return batch


def build_feature_index(feature_names):
    if not feature_names:
        raise ValueError("feature_names mancante dal dataset.")
    target_history_idx = len(feature_names) - 1
    covariate_indices = list(range(target_history_idx))
    return {"target_history": target_history_idx, "covariates": covariate_indices}


@torch.no_grad()
def evaluate(model, loader, device, quantiles, ablation_cfg, feature_idx, scaler=None):
    model.eval()
    total_pinball = 0.0
    pinball_sum_sq = 0.0
    total_crps = 0.0
    total_crps_sq = 0.0
    total_items = 0
    per_quantile_loss = np.zeros(len(quantiles), dtype=np.float64)
    per_quantile_loss_sq = np.zeros(len(quantiles), dtype=np.float64)
    coverage_hits = 0
    coverage_total = 0
    total_sq_error = 0.0
    total_sq_error_sq = 0.0
    total_abs_error = 0.0
    total_abs_error_sq = 0.0
    total_abs_target = 0.0
    total_naive_abs_error = 0.0
    total_naive_count = 0
    rows = []
    sample_plot_data = None
    median_idx = int(np.argmin([abs(q - 0.5) for q in quantiles]))

    for batch in loader:
        batch = move_to_device(batch, device)
        # Save original history target for naive baseline before any ablation.
        history_target_true = batch["history"][:, :, -1].clone()
        history_mask_true = batch["history_mask"][:, :, -1].clone()
        batch = apply_ablation(batch, ablation_cfg, feature_idx)
        preds = model(batch)  # (B, T, Q)
        targets = batch["target"]  # (B, T)
        mask = batch["target_mask"]  # (B, T)

        preds_np = preds.detach().cpu().numpy()
        targets_np = targets.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        history_target_np = history_target_true.detach().cpu().numpy()
        history_mask_np = history_mask_true.detach().cpu().numpy().astype(bool)

        if scaler is not None and scaler.has_target_stats():
            preds_np = scaler.inverse_transform_target(preds_np)
            targets_np = scaler.inverse_transform_target(targets_np, mask_np)
            history_target_np = scaler.inverse_transform_target(history_target_np, history_mask_np)

        preds_median = preds_np[..., median_idx]

        batch_items = (~mask).sum().item()
        #total_pinball += pinball.item() * batch_items
        total_items += batch_items

        diff = targets_np[:, :, None] - preds_np
        keep = (~mask_np)[:, :, None]

        losses = []
        for q_idx, tau in enumerate(quantiles):
            diff_tau = diff[:, :, q_idx]
            loss_tau = np.abs(diff_tau) * (tau * (diff_tau >= 0) + (1 - tau) * (diff_tau < 0))
            loss_tau = loss_tau * keep[..., 0]
            losses.append(loss_tau)

        loss_stack = np.stack(losses, axis=-1)
        per_item_pinball = loss_stack.mean(axis=-1)
        total_pinball += float(per_item_pinball.sum())
        pinball_sum_sq += float((per_item_pinball ** 2).sum())

        crps_per_item = 2 * np.trapz(loss_stack, quantiles, axis=-1)
        total_crps += float(crps_per_item.sum())
        total_crps_sq += float((crps_per_item ** 2).sum())

        for q_idx in range(len(quantiles)):
            per_quantile_loss[q_idx] += float(losses[q_idx].sum())
            per_quantile_loss_sq[q_idx] += float((losses[q_idx] ** 2).sum())

        lower = preds_np[..., 0]
        upper = preds_np[..., -1]
        actual = targets_np
        valid = ~mask_np
        within_band = (actual >= lower) & (actual <= upper) & valid
        coverage_hits += within_band.sum()
        coverage_total += valid.sum()

        # metriche puntuali sul q=0.5
        diff_med = preds_median - targets_np
        diff_med_masked = diff_med[valid]
        tgt_masked = targets_np[valid]
        if diff_med_masked.size > 0:
            sq_errors = diff_med_masked ** 2
            abs_errors = np.abs(diff_med_masked)
            total_sq_error += float(sq_errors.sum())
            total_sq_error_sq += float((sq_errors ** 2).sum())
            total_abs_error += float(abs_errors.sum())
            total_abs_error_sq += float((abs_errors ** 2).sum())
            total_abs_target += float(np.abs(tgt_masked).sum())

        # naive: ultimo valore valido della history
        for idx in range(batch_size := preds_np.shape[0]):
            keep = valid[idx]
            target_vals = targets_np[idx][keep]
            if target_vals.size == 0:
                continue
            hist_valid = history_target_np[idx][~history_mask_np[idx]]
            if hist_valid.size > 0:
                naive_val = hist_valid[-1]
                total_naive_abs_error += float(np.abs(target_vals - naive_val).sum())
                total_naive_count += target_vals.size

        batch_size = preds_np.shape[0]
        for idx in range(batch_size):
            keep_idx = ~mask_np[idx]
            valid_count = int(keep_idx.sum())

            if sample_plot_data is None and valid_count > 0:
                sample_plot_data = {
                    "timestamps": [str(ts) for ts in batch["target_timestamps"][idx]],
                    "targets": targets_np[idx][keep_idx],
                    "preds": preds_np[idx][keep_idx],
                }

            row = {
                "area": batch["area"][idx],
                "source": batch["source"][idx],
                "history_start": batch["history_start"][idx],
                "history_end": batch["history_end"][idx],
                "future_start": batch["future_start"][idx],
                "future_end": batch["future_end"][idx],
                "target_timestamps": "|".join(str(ts) for ts in batch["target_timestamps"][idx]),
                "targets": "|".join(
                    "" if mask_np[idx][j] else f"{targets_np[idx][j]:.6f}"
                    for j in range(len(targets_np[idx]))
                ),
            }
            for q_idx, tau in enumerate(quantiles):
                key = f"pred_q{tau:.2f}".replace(".", "p")
                row[key] = "|".join(
                    "" if mask_np[idx][j] else f"{preds_np[idx][j, q_idx]:.6f}"
                    for j in range(len(preds_np[idx]))
                )
            rows.append(row)

    def compute_std(sum_val, sum_sq, count):
        if count > 1:
            mean_val = sum_val / count
            var = (sum_sq / count) - (mean_val ** 2)
            return math.sqrt(max(var, 0.0))
        return float("nan")

    metrics = {}
    if total_items > 0:
        metrics["pinball_loss"] = total_pinball / total_items
        metrics["pinball_loss_std"] = compute_std(total_pinball, pinball_sum_sq, total_items)
        metrics["crps"] = total_crps / total_items
        metrics["crps_std"] = compute_std(total_crps, total_crps_sq, total_items)
    else:
        metrics["pinball_loss"] = float("nan")
        metrics["pinball_loss_std"] = float("nan")
        metrics["crps"] = float("nan")
        metrics["crps_std"] = float("nan")

    if coverage_total > 0:
        metrics["coverage"] = coverage_hits / coverage_total
        metrics["coverage_std"] = math.sqrt(metrics["coverage"] * (1 - metrics["coverage"])) if coverage_total > 1 else float(
            "nan"
        )
    else:
        metrics["coverage"] = float("nan")
        metrics["coverage_std"] = float("nan")

    if total_items > 0:
        mse = total_sq_error / total_items
        metrics["mse"] = mse
        metrics["rmse"] = math.sqrt(mse)
        metrics["mae"] = total_abs_error / total_items
        metrics["wmape"] = (total_abs_error / total_abs_target) if total_abs_target > 0 else float("nan")
        metrics["mse_std"] = compute_std(total_sq_error, total_sq_error_sq, total_items)
        metrics["mae_std"] = compute_std(total_abs_error, total_abs_error_sq, total_items)
        rmse = metrics["rmse"]
        metrics["rmse_std"] = 0.5 * metrics["mse_std"] / rmse if rmse > 0 and not math.isnan(metrics["mse_std"]) else float("nan")
        metrics["wmape_std"] = (
            metrics["mae_std"] * total_items / total_abs_target
            if total_abs_target > 0 and not math.isnan(metrics["mae_std"])
            else float("nan")
        )
    else:
        metrics.update(
            {
                "mse": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "wmape": float("nan"),
                "mse_std": float("nan"),
                "mae_std": float("nan"),
                "rmse_std": float("nan"),
                "wmape_std": float("nan"),
            }
        )

    if total_naive_count > 0 and total_naive_abs_error > 0 and total_items > 0:
        naive_den = total_naive_abs_error / total_naive_count
        metrics["mase"] = (total_abs_error / total_items) / naive_den
        metrics["mase_std"] = metrics["mae_std"] / naive_den if naive_den > 0 and not math.isnan(metrics["mae_std"]) else float("nan")
    else:
        metrics["mase"] = float("nan")
        metrics["mase_std"] = float("nan")

    per_q_metrics = []
    for q_idx, tau in enumerate(quantiles):
        per_loss = per_quantile_loss[q_idx] / total_items if total_items > 0 else float("nan")
        per_std = (
            compute_std(per_quantile_loss[q_idx], per_quantile_loss_sq[q_idx], total_items) if total_items > 0 else float("nan")
        )
        per_q_metrics.append((tau, per_loss, per_std))

    return metrics, per_q_metrics, rows, sample_plot_data


def plot_quantiles(sample_data, quantiles, output_path):
    if sample_data is None:
        return

    x_idx = np.arange(len(sample_data["targets"]))
    targets = sample_data["targets"]
    preds = sample_data["preds"]
    timestamps = sample_data["timestamps"]

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.scatter(x_idx, targets, color="black", label="Target", zorder=3)

    lower = preds[:, 0]
    upper = preds[:, -1]
    ax.fill_between(x_idx, lower, upper, color="tab:blue", alpha=0.2, label=f"Banda [{quantiles[0]}, {quantiles[-1]}]")

    for q_idx, tau in enumerate(quantiles):
        ax.plot(x_idx, preds[:, q_idx], label=f"q={tau:.2f}")

    ax.set_xticks(x_idx)
    ax.set_xticklabels(timestamps, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Valore previsto")
    ax.set_title("Previsioni quantili AgriMatNet")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Test probabilistico di AgriMatNet (quantile).")
    parser.add_argument("--cache-root", required=True, help="Cartella cache (train/val/test).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weights", required=True, help="File .pth con i pesi del modello.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Cartella output risultati. Se omesso usa la cartella del checkpoint --weights.",
    )
    parser.add_argument("--no-scaling", action="store_true", help="Disabilita lo scaler salvato nella cache.")
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=None,
        help="Percorso esplicito per lo scaler da usare (es. quello del training).",
    )
    parser.add_argument(
        "--quantiles",
        type=parse_quantiles,
        default=None,
        help="Quantili separati da virgola (es. 0.1,0.5,0.9). Se omesso vengono letti dal checkpoint.",
    )
    parser.add_argument("--d-model", type=int, default=128, help="Dimensione d_model degli encoder.")
    parser.add_argument("--num-layers", type=int, default=2, help="Numero di layer del Transformer encoder.")
    parser.add_argument("--num-heads", type=int, default=4, help="Numero di teste di attenzione.")
    parser.add_argument("--dim-feedforward", type=int, default=256, help="Dimensione FFN interna.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout usato in AgriMatNet.")
    parser.add_argument(
        "--feature-engineering",
        type=str2bool,
        default=False,
        help="Abilita le feature ingegnerizzate calcolate nel dataset.",
    )
    parser.add_argument(
        "--discretize-target",
        type=str2bool,
        default=False,
        help="Discretizza il target NDVI in classi (per esperimenti di classificazione).",
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

    checkpoint = torch.load(args.weights, map_location="cpu")
    quantiles = args.quantiles
    if quantiles is None:
        stored_args = checkpoint.get("args", {})
        quantiles = stored_args.get("quantiles")
        if quantiles is None:
            raise ValueError("Quantili non specificati né nel checkpoint né via CLI.")

    dataset = CacheTimeSeriesDataset(
        cache_dir=args.cache_root,
        apply_scaling=not args.no_scaling,
        scaler_path=args.scaler_path,
        feature_engineering=args.feature_engineering,
        discretize_target=args.discretize_target,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable,
        num_workers=0,
    )

    input_dim = len(dataset.feature_names)
    model = AgriMatNetQuantile(
        input_dim=input_dim,
        quantiles=quantiles,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    feature_idx = build_feature_index(dataset.feature_names)
    ablation_cfg = {
        "future_covariates_off": args.ablate_future_covariates,
        "history_covariates_off": args.ablate_history_covariates,
        "target_history_off": args.ablate_target_history,
    }

    metrics, per_q_metrics, rows, sample_plot_data = evaluate(
        model,
        loader,
        device,
        quantiles,
        ablation_cfg,
        feature_idx,
        scaler=dataset.scaler if not args.no_scaling else None,
    )

    experiment_dir = Path(args.output_dir).resolve() if args.output_dir else Path(args.weights).resolve().parent
    experiment_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = experiment_dir / "test_predictions_quantile.csv"
    fieldnames = [
        "area",
        "source",
        "history_start",
        "history_end",
        "future_start",
        "future_end",
        "target_timestamps",
        "targets",
    ]
    for tau in quantiles:
        key = f"pred_q{tau:.2f}".replace(".", "p")
        fieldnames.append(key)

    with predictions_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics_path = experiment_dir / "test_metrics_quantile.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["metric", "value"])
        writer.writerow(["pinball_loss", metrics["pinball_loss"]])
        writer.writerow(["pinball_loss_std", metrics["pinball_loss_std"]])
        writer.writerow(["crps", metrics["crps"]])
        writer.writerow(["crps_std", metrics["crps_std"]])
        writer.writerow(["coverage", metrics["coverage"]])
        writer.writerow(["coverage_std", metrics["coverage_std"]])
        writer.writerow(["mse", metrics["mse"]])
        writer.writerow(["mse_std", metrics["mse_std"]])
        writer.writerow(["rmse", metrics["rmse"]])
        writer.writerow(["rmse_std", metrics["rmse_std"]])
        writer.writerow(["mae", metrics["mae"]])
        writer.writerow(["mae_std", metrics["mae_std"]])
        writer.writerow(["wmape", metrics["wmape"]])
        writer.writerow(["wmape_std", metrics["wmape_std"]])
        writer.writerow(["mase", metrics["mase"]])
        writer.writerow(["mase_std", metrics["mase_std"]])
        for tau, value, std in per_q_metrics:
            writer.writerow([f"pinball_q{tau:.2f}", value])
            writer.writerow([f"pinball_q{tau:.2f}_std", std])

    plot_path = experiment_dir / "quantile_forecast.png"
    plot_quantiles(sample_plot_data, quantiles, plot_path)

    print(
        f"Pinball loss media su {args.cache_root}: {metrics['pinball_loss']:.6f} (std {metrics['pinball_loss_std']:.6f})"
    )
    print(f"CRPS medio: {metrics['crps']:.6f} (std {metrics['crps_std']:.6f})")
    print(
        f"Coverage osservato su banda [{quantiles[0]}, {quantiles[-1]}]: {metrics['coverage']:.6f} (std {metrics['coverage_std']:.6f})"
    )
    print(
        f"MSE (q=0.5): {metrics['mse']:.6f} (std {metrics['mse_std']:.6f}) | RMSE: {metrics['rmse']:.6f} (std {metrics['rmse_std']:.6f}) | MAE: {metrics['mae']:.6f} (std {metrics['mae_std']:.6f})"
    )
    print(f"WMAPE: {metrics['wmape']:.6f} (std {metrics['wmape_std']:.6f}) | MASE: {metrics['mase']:.6f} (std {metrics['mase_std']:.6f})")
    for tau, value, std in per_q_metrics:
        print(f"Pinball loss quantile {tau:.2f}: {value:.6f} (std {std:.6f})")
    print(f"Salvati i risultati in: {predictions_path}, {metrics_path} e {plot_path}")


if __name__ == "__main__":
    main()

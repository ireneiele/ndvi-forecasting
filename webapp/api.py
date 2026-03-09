from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agrimatnet.model_quantile import AgriMatNetQuantile
from dataset_builder.scaler import Scaler

# =========================
# DEFAULT CONFIGURAZIONE API
# =========================
# Modifica questi valori se vuoi cambiare i default senza usare env var.
DEFAULT_MODEL_PATH = "weights/model_best.pth"
DEFAULT_SCALER_PATH = "dataset_builder/scaler.json"
DEFAULT_QUANTILES = [0.1, 0.5, 0.9]
DEFAULT_INPUT_DIM = None  # Se None viene letto dal checkpoint
DEFAULT_D_MODEL = 128
DEFAULT_NUM_LAYERS = 8  # Usato solo se non inferibile dal checkpoint
DEFAULT_NUM_HEADS = 8
DEFAULT_DIM_FEEDFORWARD = 512
DEFAULT_DROPOUT = 0.1

# Default payload /predict
DEFAULT_APPLY_INPUT_SCALING = True
DEFAULT_INVERSE_TARGET_SCALING = True

# Feature engineering online: da base (es. 19) a full (es. 28).
DEFAULT_AUTO_FEATURE_ENGINEERING = True
DEFAULT_BASE_INPUT_DIM = 19
DEFAULT_ENGINEERED_FEATURES = 9
# Indici sulle feature base: aggiorna questi se il tuo ordine colonne e diverso.
DEFAULT_RAIN_FEATURE_INDEX = 1
DEFAULT_TEMP_FEATURE_INDEX = 2


def _parse_quantiles(raw: str) -> List[float]:
    # Converte una stringa CSV (es. "0.1,0.5,0.9") in lista di float.
    quantiles = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        quantiles.append(float(part))
    # Validazioni minime per evitare configurazioni runtime incoerenti.
    if not quantiles:
        raise ValueError("QUANTILES non puo essere vuoto.")
    if quantiles != sorted(quantiles):
        raise ValueError("QUANTILES deve essere ordinato in modo crescente.")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("QUANTILES contiene duplicati.")
    for q in quantiles:
        if not (0.0 < q < 1.0):
            raise ValueError(f"Quantile non valido: {q}")
    return quantiles


def _infer_num_layers(state_dict: dict) -> int:
    # Ricava il numero di layer dai nomi dei parametri salvati nel checkpoint.
    prefix = "history_encoder.layers."
    indices = set()
    for key in state_dict:
        if key.startswith(prefix):
            tail = key[len(prefix) :]
            layer_idx = tail.split(".", maxsplit=1)[0]
            if layer_idx.isdigit():
                indices.add(int(layer_idx))
    return (max(indices) + 1) if indices else 2


def _to_2d_array(name: str, values: List[List[Optional[float]]]) -> np.ndarray:
    # Garantisce che l'input JSON sia una matrice 2D non vuota e rettangolare.
    if not values:
        raise ValueError(f"{name} e vuoto.")
    width = len(values[0])
    if width == 0:
        raise ValueError(f"{name} contiene righe vuote.")
    for idx, row in enumerate(values):
        if len(row) != width:
            raise ValueError(f"{name} non rettangolare: riga {idx} ha {len(row)} colonne, attese {width}.")
    arr = np.array(values, dtype=np.float32)
    return arr


def _to_mask(values: np.ndarray, given_mask: Optional[List[List[bool]]], name: str) -> np.ndarray:
    # I NaN vengono sempre trattati come valori invalidi da mascherare.
    nan_mask = np.isnan(values)
    if given_mask is None:
        return nan_mask
    mask = np.array(given_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError(f"{name} shape {mask.shape} non compatibile con i dati {values.shape}.")
    # Se l'utente passa una mask esplicita, la uniamo ai NaN rilevati.
    return np.logical_or(mask, nan_mask)


def _env_int(name: str, default: int) -> int:
    # Helper per leggere env var intere con fallback.
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    # Helper per leggere env var float con fallback.
    return float(os.getenv(name, str(default)))


def _stats_between_targets(sequence: np.ndarray, rain_idx: int, temp_idx: int, boundaries: List[int]):
    # Replica la logica _between_targets_stats usata nel dataset builder.
    t_len = sequence.shape[0]
    rain_feat = np.full((t_len,), np.nan, dtype=np.float32)
    cold_feat = np.full((t_len,), np.nan, dtype=np.float32)
    hot_feat = np.full((t_len,), np.nan, dtype=np.float32)
    for prev, bound in zip([-1] + boundaries[:-1], boundaries):
        if bound < 0 or bound >= t_len:
            continue
        window = sequence[prev + 1 : bound + 1]
        rain_feat[bound] = float(window[:, rain_idx].sum())
        temp_col = window[:, temp_idx]
        cold_feat[bound] = float((temp_col < 10.0).sum())
        hot_feat[bound] = float((temp_col > 30.0).sum())
    return rain_feat, cold_feat, hot_feat


def _rolling_stats_by_time(
    sequence: np.ndarray, timestamps: np.ndarray, rain_idx: int, temp_idx: int, window_days: int
):
    # Replica la logica rolling per finestre temporali (7d/14d) del dataset builder.
    rain_feat = np.zeros((len(timestamps),), dtype=np.float32)
    cold_feat = np.zeros((len(timestamps),), dtype=np.float32)
    hot_feat = np.zeros((len(timestamps),), dtype=np.float32)
    for idx, current_ts in enumerate(timestamps):
        start = current_ts - np.timedelta64(window_days, "D")
        mask = (timestamps >= start) & (timestamps <= current_ts)
        window = sequence[mask]
        if window.size == 0:
            continue
        rain_feat[idx] = float(window[:, rain_idx].sum())
        temp_col = window[:, temp_idx]
        cold_feat[idx] = float((temp_col < 10.0).sum())
        hot_feat[idx] = float((temp_col > 30.0).sum())
    return rain_feat, cold_feat, hot_feat


def _rolling_stats_by_steps(sequence: np.ndarray, rain_idx: int, temp_idx: int, window_size: int):
    # Fallback se non sono forniti timestamp: usa una finestra sugli ultimi N step.
    rain_feat = np.zeros((len(sequence),), dtype=np.float32)
    cold_feat = np.zeros((len(sequence),), dtype=np.float32)
    hot_feat = np.zeros((len(sequence),), dtype=np.float32)
    for idx in range(len(sequence)):
        start = max(0, idx - window_size + 1)
        window = sequence[start : idx + 1]
        rain_feat[idx] = float(window[:, rain_idx].sum())
        temp_col = window[:, temp_idx]
        cold_feat[idx] = float((temp_col < 10.0).sum())
        hot_feat[idx] = float((temp_col > 30.0).sum())
    return rain_feat, cold_feat, hot_feat


def _build_engineered_features(
    sequence: np.ndarray,
    boundaries: List[int],
    rain_idx: int,
    temp_idx: int,
    timestamps: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Costruisce le 9 colonne engineered nello stesso ordine del training.
    rain_bt, cold_bt, hot_bt = _stats_between_targets(sequence, rain_idx, temp_idx, boundaries)
    if timestamps is not None:
        rain7, cold7, hot7 = _rolling_stats_by_time(sequence, timestamps, rain_idx, temp_idx, 7)
        rain14, cold14, hot14 = _rolling_stats_by_time(sequence, timestamps, rain_idx, temp_idx, 14)
    else:
        rain7, cold7, hot7 = _rolling_stats_by_steps(sequence, rain_idx, temp_idx, 7)
        rain14, cold14, hot14 = _rolling_stats_by_steps(sequence, rain_idx, temp_idx, 14)
    return np.stack([rain_bt, cold_bt, hot_bt, rain7, rain14, cold7, hot7, cold14, hot14], axis=1).astype(np.float32)


def _expand_with_feature_engineering(
    history: np.ndarray,
    future: np.ndarray,
    history_mask: np.ndarray,
    future_mask: np.ndarray,
    target_positions: List[int],
    rain_idx: int,
    temp_idx: int,
    target_idx: int,
    history_timestamps: Optional[List[str]] = None,
    future_timestamps: Optional[List[str]] = None,
):
    # Costruisce feature engineered per history e future, inserendole prima del target.
    history_bounds = [idx for idx in range(history.shape[0]) if not history_mask[idx, target_idx]]
    eng_history = _build_engineered_features(
        history,
        history_bounds,
        rain_idx,
        temp_idx,
        np.array(history_timestamps, dtype="datetime64[ns]") if history_timestamps is not None else None,
    )

    combined_seq = np.concatenate([history, future], axis=0)
    combined_bounds = history_bounds + [history.shape[0] + idx for idx in target_positions]
    combined_timestamps = None
    if history_timestamps is not None and future_timestamps is not None:
        combined_timestamps = np.concatenate(
            [
                np.array(history_timestamps, dtype="datetime64[ns]"),
                np.array(future_timestamps, dtype="datetime64[ns]"),
            ]
        )
    eng_combined = _build_engineered_features(combined_seq, combined_bounds, rain_idx, temp_idx, combined_timestamps)
    eng_future = eng_combined[history.shape[0] :]

    expanded_history = np.concatenate(
        [history[:, :target_idx], eng_history, history[:, target_idx : target_idx + 1]], axis=1
    )
    expanded_future = np.concatenate(
        [future[:, :target_idx], eng_future, future[:, target_idx : target_idx + 1]], axis=1
    )
    expanded_history_mask = np.concatenate(
        [history_mask[:, :target_idx], np.isnan(eng_history), history_mask[:, target_idx : target_idx + 1]], axis=1
    )
    expanded_future_mask = np.concatenate(
        [future_mask[:, :target_idx], np.isnan(eng_future), future_mask[:, target_idx : target_idx + 1]], axis=1
    )
    return expanded_history, expanded_future, expanded_history_mask, expanded_future_mask


def _infer_scaler_feature_dim(scaler: Scaler) -> Optional[int]:
    # Restituisce il numero di feature previste dallo scaler.
    if scaler.mode in {"standardization", "standardization_arcsinh"} and scaler.mean is not None:
        return int(len(scaler.mean))
    if scaler.mode == "min-max" and scaler.min_value is not None:
        return int(len(scaler.min_value))
    return None


def _scale_base_columns_after_expand(
    history: np.ndarray, future: np.ndarray, scaler: Scaler, base_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    # Dopo espansione (28), scala solo le 19 feature originali:
    # [0..base_dim-2] + [ultima colonna target].
    base_non_target = base_dim - 1
    base_indices = list(range(base_non_target)) + [history.shape[1] - 1]
    hist_base = history[:, base_indices]
    fut_base = future[:, base_indices]
    history[:, base_indices] = scaler.transform(hist_base)
    future[:, base_indices] = scaler.transform(fut_base)
    return history, future


class PredictionRequest(BaseModel):
    history: List[List[Optional[float]]] = Field(..., description="Sequenza storica [T_history, F].")
    future: List[List[Optional[float]]] = Field(..., description="Sequenza futura [T_future, F].")
    history_mask: Optional[List[List[bool]]] = Field(
        default=None, description="Maschera opzionale history, True sui valori da ignorare."
    )
    future_mask: Optional[List[List[bool]]] = Field(
        default=None, description="Maschera opzionale future, True sui valori da ignorare."
    )
    future_target_positions: Optional[List[int]] = Field(
        default=None,
        description="Indici dei timestep nel future da predire. Default: tutti i timestep.",
    )
    history_timestamps: Optional[List[str]] = Field(
        default=None, description="Timestamp history ISO (opzionale, usato per rolling 7d/14d reale)."
    )
    future_timestamps: Optional[List[str]] = Field(
        default=None, description="Timestamp future ISO (opzionale, usato per rolling 7d/14d reale)."
    )
    apply_input_scaling: bool = Field(
        default=DEFAULT_APPLY_INPUT_SCALING,
        description="Se true e presente lo scaler, scala history/future prima del forward.",
    )
    inverse_target_scaling: bool = Field(
        default=DEFAULT_INVERSE_TARGET_SCALING,
        description="Se true e disponibile target stats nello scaler, riporta output in scala originale.",
    )


class PredictionResponse(BaseModel):
    quantiles: List[float]
    predictions: List[List[float]]


def _load_runtime():
    # Risolve il percorso pesi del modello dal runtime environment.
    model_path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH non trovato: {model_path}")

    # Supporta sia checkpoint completi (con model_state_dict) sia state_dict puro.
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        stored_args = checkpoint.get("args", {})
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        stored_args = {}
    else:
        raise ValueError("Checkpoint non valido: atteso dict o checkpoint con model_state_dict.")

    if "history_proj.weight" not in state_dict:
        raise ValueError("state_dict non compatibile con AgriMatNetQuantile: manca history_proj.weight.")

    # Usa i quantili del checkpoint se presenti, altrimenti default statico.
    default_quantiles = stored_args.get("quantiles", DEFAULT_QUANTILES)
    quantiles_raw = os.getenv("QUANTILES", ",".join(str(q) for q in default_quantiles))
    quantiles = _parse_quantiles(quantiles_raw)

    # Ricostruisce i parametri architetturali con priorita: env var -> args checkpoint -> fallback.
    checkpoint_input_dim = state_dict["history_proj.weight"].shape[1]
    input_dim_fallback = checkpoint_input_dim if DEFAULT_INPUT_DIM is None else DEFAULT_INPUT_DIM
    input_dim = _env_int("INPUT_DIM", stored_args.get("input_dim", input_dim_fallback))
    d_model = _env_int("D_MODEL", stored_args.get("d_model", DEFAULT_D_MODEL))
    num_layers_fallback = stored_args.get("num_layers", _infer_num_layers(state_dict))
    num_layers = _env_int("NUM_LAYERS", num_layers_fallback if num_layers_fallback is not None else DEFAULT_NUM_LAYERS)
    num_heads = _env_int("NUM_HEADS", stored_args.get("num_heads", DEFAULT_NUM_HEADS))
    dim_feedforward = _env_int("DIM_FEEDFORWARD", stored_args.get("dim_feedforward", DEFAULT_DIM_FEEDFORWARD))
    dropout = _env_float("DROPOUT", stored_args.get("dropout", DEFAULT_DROPOUT))

    # Inizializza e carica il modello in modalita eval (solo inferenza).
    model = AgriMatNetQuantile(
        input_dim=input_dim,
        quantiles=quantiles,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Carica facoltativamente lo scaler per avere pipeline coerente con training/test.
    scaler_path_raw = os.getenv("SCALER_PATH", DEFAULT_SCALER_PATH)
    scaler = None
    if scaler_path_raw:
        scaler_path = Path(scaler_path_raw).resolve()
        if not scaler_path.exists():
            raise FileNotFoundError(f"SCALER_PATH non trovato: {scaler_path}")
        scaler = Scaler.load(scaler_path)

    return {
        "model": model,
        "quantiles": quantiles,
        "input_dim": input_dim,
        "model_path": str(model_path),
        "scaler": scaler,
    }


# Runtime caricato una volta sola all'avvio del processo API.
runtime = _load_runtime()
app = FastAPI(title="NDVI Forecast API", version="0.1.0")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": runtime["model_path"],
        "input_dim": runtime["input_dim"],
        "quantiles": runtime["quantiles"],
        "scaler_loaded": runtime["scaler"] is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        # Converte history/future da JSON a matrici numpy validate.
        history = _to_2d_array("history", payload.history)
        future = _to_2d_array("future", payload.future)

        # Verifica consistenza feature tra input e architettura del modello.
        if history.shape[1] != future.shape[1]:
            raise ValueError(
                f"Numero feature incoerente: history={history.shape[1]} future={future.shape[1]}."
            )

        # Costruisce le maschere finali includendo sia mask utente sia NaN.
        history_mask = _to_mask(history, payload.history_mask, "history_mask")
        future_mask = _to_mask(future, payload.future_mask, "future_mask")

        # Se non specificato, predice su tutti i timestep del future.
        if payload.future_target_positions is not None:
            if not payload.future_target_positions:
                raise ValueError("future_target_positions non puo essere vuoto.")
            target_positions = payload.future_target_positions
        else:
            target_positions = list(range(future.shape[0]))

        # Controlla che tutti gli indici target siano nel range della sequenza futura.
        max_idx = future.shape[0] - 1
        for idx in target_positions:
            if idx < 0 or idx > max_idx:
                raise ValueError(
                    f"Indice in future_target_positions fuori range: {idx}, valido [0, {max_idx}]."
                )

        # Se il modello e stato addestrato con ft_eng, espande da base_dim a input_dim.
        input_dim_received = history.shape[1]
        model_input_dim = runtime["input_dim"]
        can_expand = (
            DEFAULT_AUTO_FEATURE_ENGINEERING
            and input_dim_received == DEFAULT_BASE_INPUT_DIM
            and model_input_dim == DEFAULT_BASE_INPUT_DIM + DEFAULT_ENGINEERED_FEATURES
        )

        # Applica scaling al momento giusto.
        scaler = runtime["scaler"]
        scale_after_expand = False
        scale_base_after_expand = False
        if payload.apply_input_scaling and scaler is not None:
            scaler_dim = _infer_scaler_feature_dim(scaler)
            if scaler_dim is not None and scaler_dim == input_dim_received:
                # Se poi espandiamo 19->28, lo scaling viene applicato dopo alle sole 19 originali.
                if can_expand:
                    scale_base_after_expand = True
                else:
                    history = scaler.transform(history)
                    future = scaler.transform(future)
            elif scaler_dim is not None and scaler_dim == model_input_dim:
                scale_after_expand = True
            elif scaler_dim is not None:
                raise ValueError(
                    f"Scaler non compatibile: attese {scaler_dim} feature ma input/modello sono {input_dim_received}/{model_input_dim}."
                )
            else:
                # Fallback: se non riesco a inferire la dimensione, provo sulle feature ricevute.
                history = scaler.transform(history)
                future = scaler.transform(future)

        if input_dim_received != model_input_dim:
            if not can_expand:
                raise ValueError(
                    f"Numero feature non compatibile con il modello: ricevute {input_dim_received}, attese {model_input_dim}."
                )
            target_idx = input_dim_received - 1
            rain_idx = int(os.getenv("RAIN_FEATURE_INDEX", str(DEFAULT_RAIN_FEATURE_INDEX)))
            temp_idx = int(os.getenv("TEMP_FEATURE_INDEX", str(DEFAULT_TEMP_FEATURE_INDEX)))
            if rain_idx < 0 or rain_idx >= target_idx:
                raise ValueError(f"RAIN_FEATURE_INDEX fuori range: {rain_idx}.")
            if temp_idx < 0 or temp_idx >= target_idx:
                raise ValueError(f"TEMP_FEATURE_INDEX fuori range: {temp_idx}.")
            if payload.history_timestamps is not None and len(payload.history_timestamps) != history.shape[0]:
                raise ValueError(
                    f"history_timestamps lunghezza {len(payload.history_timestamps)} non coerente con history {history.shape[0]}."
                )
            if payload.future_timestamps is not None and len(payload.future_timestamps) != future.shape[0]:
                raise ValueError(
                    f"future_timestamps lunghezza {len(payload.future_timestamps)} non coerente con future {future.shape[0]}."
                )
            history, future, history_mask, future_mask = _expand_with_feature_engineering(
                history,
                future,
                history_mask,
                future_mask,
                target_positions,
                rain_idx,
                temp_idx,
                target_idx,
                payload.history_timestamps,
                payload.future_timestamps,
            )

        if scale_after_expand and scaler is not None:
            history = scaler.transform(history)
            future = scaler.transform(future)
        elif scale_base_after_expand and scaler is not None:
            history, future = _scale_base_columns_after_expand(
                history, future, scaler, DEFAULT_BASE_INPUT_DIM
            )

        # Sostituisce i NaN con 0: il modello ignora quei punti tramite mask.
        history = np.nan_to_num(history, nan=0.0).astype(np.float32)
        future = np.nan_to_num(future, nan=0.0).astype(np.float32)

        # Prepara il mini-batch in formato atteso dal forward di AgriMatNetQuantile.
        batch = {
            "history": torch.from_numpy(history).unsqueeze(0),
            "future": torch.from_numpy(future).unsqueeze(0),
            "history_mask": torch.from_numpy(history_mask).unsqueeze(0),
            "future_mask": torch.from_numpy(future_mask).unsqueeze(0),
            "history_pad_mask": torch.zeros((1, history.shape[0]), dtype=torch.bool),
            "future_pad_mask": torch.zeros((1, future.shape[0]), dtype=torch.bool),
            "future_target_positions": torch.tensor([target_positions], dtype=torch.long),
        }

        # Inferenza senza gradiente per ridurre overhead e memoria.
        with torch.no_grad():
            preds = runtime["model"](batch).cpu().numpy()[0]

        # Riporta output in scala originale del target, se disponibile e richiesto.
        if payload.inverse_target_scaling and scaler is not None and scaler.has_target_stats():
            preds = scaler.inverse_transform_target(preds)

        return PredictionResponse(
            quantiles=runtime["quantiles"],
            predictions=preds.astype(float).tolist(),
        )
    # Errori di validazione input/config vengono restituiti come 400.
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Qualsiasi altro errore viene trattato come errore interno 500.
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore interno: {exc}") from exc

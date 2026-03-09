from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agrimatnet.model_quantile import AgriMatNetQuantile
from dataset_builder.scaler import Scaler


def _parse_quantiles(raw: str) -> List[float]:
    quantiles = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        quantiles.append(float(part))
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
    nan_mask = np.isnan(values)
    if given_mask is None:
        return nan_mask
    mask = np.array(given_mask, dtype=bool)
    if mask.shape != values.shape:
        raise ValueError(f"{name} shape {mask.shape} non compatibile con i dati {values.shape}.")
    return np.logical_or(mask, nan_mask)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


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
    apply_input_scaling: bool = Field(
        default=True, description="Se true e presente lo scaler, scala history/future prima del forward."
    )
    inverse_target_scaling: bool = Field(
        default=True, description="Se true e disponibile target stats nello scaler, riporta output in scala originale."
    )


class PredictionResponse(BaseModel):
    quantiles: List[float]
    predictions: List[List[float]]


def _load_runtime():
    model_path = Path(os.getenv("MODEL_PATH", "weights/model_best.pth")).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MODEL_PATH non trovato: {model_path}")

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

    default_quantiles = stored_args.get("quantiles", [0.1, 0.5, 0.9])
    quantiles_raw = os.getenv("QUANTILES", ",".join(str(q) for q in default_quantiles))
    quantiles = _parse_quantiles(quantiles_raw)

    input_dim = _env_int("INPUT_DIM", state_dict["history_proj.weight"].shape[1])
    d_model = _env_int("D_MODEL", stored_args.get("d_model", 128))
    num_layers = _env_int("NUM_LAYERS", stored_args.get("num_layers", _infer_num_layers(state_dict)))
    num_heads = _env_int("NUM_HEADS", stored_args.get("num_heads", 4))
    dim_feedforward = _env_int("DIM_FEEDFORWARD", stored_args.get("dim_feedforward", 256))
    dropout = _env_float("DROPOUT", stored_args.get("dropout", 0.1))

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

    scaler_path_raw = os.getenv("SCALER_PATH")
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
        history = _to_2d_array("history", payload.history)
        future = _to_2d_array("future", payload.future)

        if history.shape[1] != future.shape[1]:
            raise ValueError(
                f"Numero feature incoerente: history={history.shape[1]} future={future.shape[1]}."
            )
        if history.shape[1] != runtime["input_dim"]:
            raise ValueError(
                f"Numero feature non compatibile con il modello: ricevute {history.shape[1]}, attese {runtime['input_dim']}."
            )

        scaler = runtime["scaler"]
        if payload.apply_input_scaling and scaler is not None:
            history = scaler.transform(history)
            future = scaler.transform(future)

        history_mask = _to_mask(history, payload.history_mask, "history_mask")
        future_mask = _to_mask(future, payload.future_mask, "future_mask")

        history = np.nan_to_num(history, nan=0.0).astype(np.float32)
        future = np.nan_to_num(future, nan=0.0).astype(np.float32)

        if payload.future_target_positions is None:
            target_positions = list(range(future.shape[0]))
        else:
            if not payload.future_target_positions:
                raise ValueError("future_target_positions non puo essere vuoto.")
            target_positions = payload.future_target_positions

        max_idx = future.shape[0] - 1
        for idx in target_positions:
            if idx < 0 or idx > max_idx:
                raise ValueError(
                    f"Indice in future_target_positions fuori range: {idx}, valido [0, {max_idx}]."
                )

        batch = {
            "history": torch.from_numpy(history).unsqueeze(0),
            "future": torch.from_numpy(future).unsqueeze(0),
            "history_mask": torch.from_numpy(history_mask).unsqueeze(0),
            "future_mask": torch.from_numpy(future_mask).unsqueeze(0),
            "history_pad_mask": torch.zeros((1, history.shape[0]), dtype=torch.bool),
            "future_pad_mask": torch.zeros((1, future.shape[0]), dtype=torch.bool),
            "future_target_positions": torch.tensor([target_positions], dtype=torch.long),
        }

        with torch.no_grad():
            preds = runtime["model"](batch).cpu().numpy()[0]

        if payload.inverse_target_scaling and scaler is not None and scaler.has_target_stats():
            preds = scaler.inverse_transform_target(preds)

        return PredictionResponse(
            quantiles=runtime["quantiles"],
            predictions=preds.astype(float).tolist(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Errore interno: {exc}") from exc

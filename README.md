# Release (quantile ablation)

Questa cartella contiene solo il codice necessario per training e test del modello **AgriMatNet quantile** con script di **ablation**.

Incluso:
- `agrimatnet/train_quantile_ablation.py`
- `agrimatnet/test_quantile_ablation.py`
- `agrimatnet/model_quantile.py`
- `agrimatnet/layers.py`
- `agrimatnet/train_utils.py`
- `dataset_builder/torch_dataset.py`
- `dataset_builder/scaler.py`
- `requirements.txt`

Escluso volutamente:
- dataset/cache (`timeSeries/...`)
- checkpoint/output/log
- script e codice baseline/competitor

## Esecuzione
Dalla cartella `release/`:

```bash
python agrimatnet/train_quantile_ablation.py --cache-root /percorso/alla/cache/train
python agrimatnet/test_quantile_ablation.py --cache-root /percorso/alla/cache/test --weights /percorso/model_best.pth
```

## API FastAPI per GUI

Questa release include un servizio FastAPI per esporre il modello via HTTP.

### 1) Avvio `uvicorn`

```bash
source .venv/bin/activate
uvicorn webapp.api:app --host 0.0.0.0 --port 8000 --reload
```

Variabili ambiente principali:
- `MODEL_PATH` percorso pesi modello (default: `weights/model_best.pth`)
- `SCALER_PATH` percorso scaler (default: `dataset_builder/scaler.json`)
- `QUANTILES` quantili separati da virgola (es. `0.1,0.5,0.9`)
- `INPUT_DIM`, `D_MODEL`, `NUM_LAYERS`, `NUM_HEADS`, `DIM_FEEDFORWARD`, `DROPOUT` override architettura

Esempio con env var:

```bash
MODEL_PATH=weights/model_best.pth \
SCALER_PATH=dataset_builder/scaler.json \
QUANTILES=0.1,0.5,0.9 \
uvicorn webapp.api:app --host 0.0.0.0 --port 8000 --reload
```

### 2) Health check

```bash
curl http://127.0.0.1:8000/health
```

### 3) Predict (esempio 5 timestep futuri)

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      [0.10,0.00,12.3,3.2,0.40,0.15,0.80,0.00,0.22,0.11,0.05,0.00,0.00,0.10,0.00,0.00,0.00,0.00,0.35],
      [0.12,0.00,12.7,3.1,0.41,0.16,0.81,0.00,0.23,0.10,0.05,0.00,0.00,0.11,0.00,0.00,0.00,0.00,0.36]
    ],
    "future": [
      [0.13,0.00,13.0,3.0,0.42,0.17,0.82,0.00,0.24,0.10,0.05,0.00,0.00,0.12,0.00,0.00,0.00,0.00,0.00],
      [0.14,0.00,13.2,3.0,0.43,0.18,0.82,0.00,0.25,0.10,0.05,0.00,0.00,0.13,0.00,0.00,0.00,0.00,0.00],
      [0.15,0.00,13.4,2.9,0.44,0.18,0.83,0.00,0.26,0.09,0.05,0.00,0.00,0.14,0.00,0.00,0.00,0.00,0.00],
      [0.16,0.00,13.6,2.9,0.45,0.19,0.83,0.00,0.27,0.09,0.05,0.00,0.00,0.15,0.00,0.00,0.00,0.00,0.00],
      [0.17,0.00,13.8,2.8,0.46,0.19,0.84,0.00,0.28,0.09,0.05,0.00,0.00,0.16,0.00,0.00,0.00,0.00,0.00]
    ],
    "forecast_horizon": 5,
    "apply_input_scaling": true,
    "inverse_target_scaling": true
  }'
```

### 4) Significato parametri `POST /predict`

- `history`: sequenza storica `[T_history, F]` (F=19 base, oppure F=28 se già con feature engineering).
- `future`: sequenza futura `[T_future, F]` con stesso numero di feature di `history`.
- `history_mask` (opzionale): matrice booleana come `history`; `true` indica valore da ignorare.
- `future_mask` (opzionale): matrice booleana come `future`; `true` indica valore da ignorare.
- `future_target_positions` (opzionale): indici esatti dei timestep futuri da predire (es. `[0,2,4]`).
- `forecast_horizon` (opzionale): alternativa a `future_target_positions`; predice i primi `N` step (`[0..N-1]`).
- `history_timestamps` (opzionale): timestamp ISO della history, usati per calcolo rolling 7d/14d reale.
- `future_timestamps` (opzionale): timestamp ISO del future, usati per calcolo rolling 7d/14d reale.
- `apply_input_scaling`: applica scaling input se lo scaler è configurato.
- `inverse_target_scaling`: riporta l'output in scala target originale se lo scaler ha statistiche target.

Note:
- usare solo uno tra `future_target_positions` e `forecast_horizon`;
- se non passi nessuno dei due, il modello predice tutti i timestep di `future`;
- output `predictions` ha shape `[num_timestep_predetti, num_quantili]`.

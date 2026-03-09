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

### 1) Avvio servizio

```bash
uvicorn webapp.api:app --host 0.0.0.0 --port 8000
```

Variabili ambiente supportate:

- `MODEL_PATH` (default: `weights/model_best.pth`)
- `SCALER_PATH` (opzionale, se vuoi scaling input + inverse scaling output)
- `QUANTILES` (es. `0.1,0.5,0.9`; default: da checkpoint se presente, altrimenti `0.1,0.5,0.9`)
- `INPUT_DIM`, `D_MODEL`, `NUM_LAYERS`, `NUM_HEADS`, `DIM_FEEDFORWARD`, `DROPOUT` (override opzionali)

Esempio:

```bash
MODEL_PATH=weights/model_best.pth \
SCALER_PATH=/percorso/scaler.json \
QUANTILES=0.1,0.5,0.9 \
uvicorn webapp.api:app --host 0.0.0.0 --port 8000
```

### 2) Endpoint disponibili

- `GET /health` stato servizio e configurazione runtime
- `POST /predict` inferenza NDVI quantile

Payload `POST /predict` (singolo campione):

```json
{
  "history": [[0.1, 0.2, 0.3], [0.2, 0.2, 0.35]],
  "future": [[0.25, 0.1, 0.4], [0.22, 0.12, 0.42], [0.2, 0.11, 0.41]],
  "history_mask": [[false, false, false], [false, false, false]],
  "future_mask": [[false, false, false], [false, false, false], [false, false, false]],
  "future_target_positions": [0, 1, 2],
  "apply_input_scaling": true,
  "inverse_target_scaling": true
}
```

Risposta:

```json
{
  "quantiles": [0.1, 0.5, 0.9],
  "predictions": [
    [0.41, 0.45, 0.5],
    [0.39, 0.44, 0.49],
    [0.38, 0.43, 0.48]
  ]
}
```

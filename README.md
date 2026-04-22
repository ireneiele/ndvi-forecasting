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

### 3) Predict (esempio 3 passaggi futuri)

Schema colonne dell'input base:

| Posizione | Nome colonna |
|---|---|
| 1 | `rainfall` |
| 2 | `avg_temperature` |
| 3-18 | altre covariate meteo/temporali presenti nella cache di training |
| 19 | `NDVI` storico, cioe il target passato |

Se `feature_engineering=true` e il modello e stato addestrato con le feature ingegnerizzate, la API espande automaticamente l'input da 19 a 28 colonne aggiungendo le 9 feature derivate prima dell'ultima colonna.

Quindi una finestra storica di 3 timestep si legge cosi:

```text
t-2: rainfall=0.09, avg_temperature=12.1, ..., NDVI=0.34
t-1: rainfall=0.10, avg_temperature=12.3, ..., NDVI=0.35
t0:  rainfall=0.12, avg_temperature=12.7, ..., NDVI=0.36
```

Nello stesso modo, `future` contiene i 3 timestep futuri da cui il modello deve produrre la previsione.
Anche `future` usa lo stesso schema a 19 colonne, ma la colonna `NDVI` dei timestep futuri non e nota: puoi lasciarla vuota o `NaN` e la API la esclude automaticamente.

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "history": [
      [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19],
      [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19],
      [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]
    ],
    "future": [
      [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19],
      [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19],
      [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19]
    ],
    "future_target_positions": [0,1,2],
    "apply_input_scaling": true,
    "feature_engineering": true,
    "inverse_target_scaling": true
  }'
```

Qui `f1` ... `f18` sono le feature di input e `f19` è il target storico (`NDVI`).

Lettura dell'esempio:
- ogni riga di `history` e `future` rappresenta un timestep
- `history` contiene il passato osservato, `future` i timestep da prevedere
- `future_target_positions: [0,1,2]` chiede la previsione sui primi 3 passaggi futuri

### 4) Significato parametri `POST /predict`

- `history`: sequenza storica `[T_history, F]` (F=19 base, oppure F=28 se già con feature engineering).
- `future`: sequenza futura `[T_future, F]` con stesso numero di feature di `history`.
- `history_mask` (opzionale): matrice booleana come `history`; `true` indica valore da ignorare.
- `future_mask` (opzionale): matrice booleana come `future`; `true` indica valore da ignorare.
- `future_target_positions` (opzionale): indici esatti dei timestep futuri da predire (es. `[0,2,4]`).
- `history_timestamps` (opzionale): timestamp ISO della history, usati per calcolo rolling 7d/14d reale.
- `future_timestamps` (opzionale): timestamp ISO del future, usati per calcolo rolling 7d/14d reale.
- `feature_engineering`: se `true`, la API aggiunge online le 9 feature ingegnerizzate quando riceve l'input base a 19 colonne.
- `apply_input_scaling`: applica scaling input se lo scaler è configurato.
- `inverse_target_scaling`: riporta l'output in scala target originale se lo scaler ha statistiche target.

Note:
- se non passi `future_target_positions`, il modello predice tutti i timestep di `future`;
- se mandi `history` e `future` con 19 feature base, la API può espanderle automaticamente a 28 feature se il modello e lo scaler sono stati addestrati con feature engineering;
- le feature ingegnerizzate usano `rainfall` e `avg_temperature` come colonne base, quindi l'ordine colonne deve essere coerente con quello usato in training;
- i `NaN` nel payload vengono mascherati automaticamente dal codice, quindi non serve azzerare manualmente `NDVI` futuro;
- output `predictions` ha shape `[num_timestep_predetti, num_quantili]`.

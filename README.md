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

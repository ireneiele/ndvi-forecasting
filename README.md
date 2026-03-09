<div align="center">
<h1>Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates
</h1>

[Irene Iele](https://scholar.google.com/citations?user=srLH7lkAAAAJ&hl=it&oi=ao)<sup>1</sup>, 
[Giulia Romoli](https://scholar.google.com/citations?user=mSFVXpIAAAAJ&hl=it&oi=ao)<sup>2</sup>, 
[Daniele Molino](https://scholar.google.com/citations?user=MxxVQxoAAAAJ&hl=it&oi=ao)<sup>1</sup>, 
[Elena Mulero Ayllón](https://scholar.google.com/citations?user=-BOMvaUAAAAJ&hl=it&oi=ao)<sup>1</sup>, 
[Filippo Ruffini](https://scholar.google.com/citations?user=eW7C8YMAAAAJ&hl=it&oi=ao)<sup>1,2</sup>, 
[Paolo Soda](https://scholar.google.com/citations?user=E7rcYCQAAAAJ&hl=it&oi=ao)<sup>1,2</sup>, 
[Matteo Tortora](https://matteotortora.github.io)<sup>3</sup>

<sup>1</sup>  University Campus Bio-Medico of Rome,
<sup>2</sup>  Umeå University,
<sup>3</sup>  University of Genoa
</div>

<div align="center">
<a href="https://arxiv.org/abs/2602.17683">
  <img src="https://img.shields.io/badge/arXiv-2602.17683-B31B1B?style=flat&logo=arxiv&logoColor=white"/>
</a>
</div>

<br/>

# Release (quantile ablation)

Questa cartella contiene solo il codice necessario per il training del modello **AgriMatNet quantile** con script di **ablation**.

Incluso:
- `agrimatnet/train_quantile_ablation.py`
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
```bash
python agrimatnet/train_quantile_ablation.py --cache-root /percorso/alla/cache/train
```

## Contact
For questions and comments, feel free to contact me: irene.iele@unicampus.it

## Citation
If you use this [work](https://arxiv.org/abs/2602.17683), please cite:

```bibtex
@article{iele2026probabilistic,
  title={Probabilistic NDVI Forecasting from Sparse Satellite Time Series and Weather Covariates},
  author={Iele, Irene and Romoli, Giulia and Molino, Daniele and Ayll{\'o}n, Elena Mulero and Ruffini, Filippo and Soda, Paolo and Tortora, Matteo},
  journal={arXiv preprint arXiv:2602.17683},
  year={2026}
}
}

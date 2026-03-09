"""
Strumenti per preparare e ispezionare i dataset delle serie temporali vegetative.
Le funzioni seguono lo stile di MATNet ma sono adattate al contesto agricolo.
"""

from .cache_builder_unified import DatasetCacheBuilder, build_default_cache
from .scaler import Scaler
from .torch_dataset import CacheTimeSeriesDataset

__all__ = [
    "DatasetCacheBuilder",
    "CacheTimeSeriesDataset",
    "build_default_cache",
    "Scaler",
]

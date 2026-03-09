"""
Strumenti per preparare e ispezionare i dataset delle serie temporali vegetative.
Le funzioni seguono lo stile di MATNet ma sono adattate al contesto agricolo.
"""

from .scaler import Scaler
from .torch_dataset import CacheTimeSeriesDataset

# Nella release ridotta il modulo cache_builder_unified puo non essere incluso.
try:
    from .cache_builder_unified import DatasetCacheBuilder, build_default_cache
except ModuleNotFoundError:
    DatasetCacheBuilder = None
    build_default_cache = None

__all__ = ["CacheTimeSeriesDataset", "Scaler"]
if DatasetCacheBuilder is not None and build_default_cache is not None:
    __all__.extend(["DatasetCacheBuilder", "build_default_cache"])

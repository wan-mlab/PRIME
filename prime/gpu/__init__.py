"""GPU-accelerated, VRAM-bounded backends for PRIME.

Importing this subpackage does NOT import cupy. It only registers the
public API symbols; cupy and ANN backends are loaded lazily on first use
so that PRIME still imports cleanly on machines without a GPU.

Public API:
  - :func:`prime.gpu.ensemble_mnn_correct` — GPU version of
    :func:`prime.ensemble_mnn_correct`.
  - :func:`prime.gpu.detect` — report which GPU/ANN backends are available.
"""

from prime.gpu._backend import detect
from prime.gpu.ensemble import ensemble_mnn_correct

__all__ = ["ensemble_mnn_correct", "detect"]

"""GPU backend detection with lazy imports.

The GPU path of PRIME requires cupy + at least one ANN backend (cuvs or
faiss-gpu). All imports are lazy so that the rest of PRIME still works
on a machine without a GPU. Helper functions here raise informative
errors when something is missing.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GPUEnv:
    cupy: bool
    cupy_sparse: bool
    cuvs: bool
    faiss_gpu: bool

    @property
    def has_gpu(self) -> bool:
        return self.cupy

    @property
    def has_ann(self) -> bool:
        return self.cuvs or self.faiss_gpu


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def _has_faiss_gpu() -> bool:
    try:
        import faiss  # type: ignore

        return faiss.get_num_gpus() > 0  # type: ignore[attr-defined]
    except Exception:
        return False


def detect() -> GPUEnv:
    """Detect available GPU dependencies without importing them globally."""
    return GPUEnv(
        cupy=_try_import("cupy"),
        cupy_sparse=_try_import("cupyx.scipy.sparse"),
        cuvs=_try_import("cuvs"),
        faiss_gpu=_has_faiss_gpu(),
    )


def require_gpu(env: Optional[GPUEnv] = None) -> GPUEnv:
    """Raise an informative ImportError if cupy is unavailable."""
    env = env or detect()
    if not env.has_gpu:
        raise ImportError(
            "prime.gpu requires cupy. Install a build matching your CUDA "
            "version, e.g. `pip install cupy-cuda12x`."
        )
    return env


def require_ann(env: Optional[GPUEnv] = None) -> GPUEnv:
    """Raise an informative ImportError if no GPU ANN backend is available."""
    env = env or detect()
    require_gpu(env)
    if not env.has_ann:
        raise ImportError(
            "prime.gpu needs a GPU ANN backend. Install one of:\n"
            "  - cuvs:       pip install cuvs-cu12 (or matching CUDA version)\n"
            "  - faiss-gpu:  pip install faiss-gpu  (or via conda-forge)"
        )
    return env


def pick_knn_backend(prefer: str = "auto", env: Optional[GPUEnv] = None) -> str:
    """Pick a GPU kNN backend by preference.

    Parameters
    ----------
    prefer
        ``"auto"`` (cuvs > faiss), ``"cuvs"``, or ``"faiss"``.

    Returns
    -------
    str
        ``"cuvs"`` or ``"faiss"``.
    """
    env = env or detect()
    require_ann(env)
    if prefer == "cuvs":
        if not env.cuvs:
            raise ImportError("cuvs backend requested but cuvs not installed.")
        return "cuvs"
    if prefer == "faiss":
        if not env.faiss_gpu:
            raise ImportError("faiss backend requested but faiss-gpu not available.")
        return "faiss"
    if prefer != "auto":
        raise ValueError(f"Unknown backend preference: {prefer!r}")
    return "cuvs" if env.cuvs else "faiss"


def free_pool() -> None:
    """Release all blocks held by cupy's default memory pool (best effort)."""
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass

"""PRIME: Projection-based Robust Integration via Manifold Embedding.

Batch effect correction for single-cell RNA-seq and spatial transcriptomics
data via ensemble random projections, consensus MNN graphs, and (for spatial
data) Laplacian-regularized embedding.

Subpackages
-----------
- ``prime.pp`` — streaming, memory-bounded preprocessing (e.g. HVG selection
  on 1M+ cells without loading the full matrix).
- ``prime.gpu`` — GPU-accelerated, VRAM-bounded backends (lazy: cupy is only
  imported when ``prime.gpu.*`` is actually called).
- ``prime.metrics`` — XLC, isolated-label, and related evaluation utilities.
- ``prime.plotting`` — scIB-style benchmark result tables.
"""

from prime import pp
from prime.core import (
    build_consensus_graph,
    ensemble_mnn_correct,
    find_multibatch_mnn_graph,
)
from prime.spatial import prime_st

__version__ = "0.2.0"

__all__ = [
    "ensemble_mnn_correct",
    "prime_st",
    "build_consensus_graph",
    "find_multibatch_mnn_graph",
    "pp",
    "__version__",
]

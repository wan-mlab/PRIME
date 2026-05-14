"""PRIME: Projection-based Robust Integration via Manifold Embedding.

Batch effect correction for single-cell RNA-seq and spatial transcriptomics
data via ensemble random projections, consensus MNN graphs, and (for spatial
data) Laplacian-regularized embedding.
"""

from prime.core import (
    build_consensus_graph,
    ensemble_mnn_correct,
    find_multibatch_mnn_graph,
)
from prime.spatial import prime_st

__version__ = "0.1.0"

__all__ = [
    "ensemble_mnn_correct",
    "prime_st",
    "build_consensus_graph",
    "find_multibatch_mnn_graph",
    "__version__",
]

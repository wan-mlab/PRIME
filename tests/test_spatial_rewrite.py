"""Regression tests for the spatial.py ERP-consensus-MNN Step-5 rewrite.

These lock in the invariants of the refactor that extracted the inline anchor
graph construction into the reusable helpers ``_random_projection_embedding``
and ``_build_rp_consensus_mnn_graph``:

* the anchor graph is still produced by ensemble random projection + consensus
  MNN (the prime.core principle);
* ``prime_st``'s anchor graph equals the standalone helper called with the same
  arguments when spatial reweighting is off (proves the extraction is faithful);
* output shape, graph symmetry, and absence of self-loops;
* the new ``context_method`` / ``base_embedding_method`` switches all run and do
  not alter the anchor *topology* (only weights / the solver RHS);
* the pipeline is deterministic for a fixed seed.

Run with:  pytest tests/test_spatial_rewrite.py
"""
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from prime.spatial import (
    prime_st,
    _build_rp_consensus_mnn_graph,
    _row_norm_log1p,
    _pick_hvgs,
)

anndata = pytest.importorskip("anndata")


# --------------------------------------------------------------------------- #
# Synthetic multi-batch spatial data
# --------------------------------------------------------------------------- #
def _make_synthetic(n_per_batch=120, n_genes=300, n_types=4, n_batches=3, seed=0):
    """Shared cell-type structure across batches (-> cross-batch MNNs exist)
    with a per-gene multiplicative batch effect; cell type follows the spatial
    quadrant so the spatial graph is informative."""
    rng = np.random.default_rng(seed)
    programs = rng.gamma(1.0, 1.0, size=(n_types, n_genes)) + 0.05

    Xs, batches, coords = [], [], []
    for b in range(n_batches):
        xy = rng.uniform(0.0, 1.0, size=(n_per_batch, 2))
        types = ((xy[:, 0] > 0.5).astype(int) + 2 * (xy[:, 1] > 0.5).astype(int)) % n_types
        batch_factor = np.exp(rng.normal(0.0, 0.3, size=n_genes))
        counts = rng.poisson(programs[types] * batch_factor[None, :] * 3.0).astype(np.float32)
        Xs.append(counts)
        batches += [f"batch{b}"] * n_per_batch
        coords.append(xy)

    import pandas as pd
    adata = anndata.AnnData(
        X=csr_matrix(np.vstack(Xs)),
        obs=pd.DataFrame({"batch": pd.Categorical(batches)}),
    )
    adata.obsm["spatial"] = np.vstack(coords).astype(np.float32)
    return adata


PARAMS = dict(
    batch_key="batch", spatial_key="spatial",
    n_hvg=150, hvg_flavor="seurat",  # seurat_v3 needs scikit-misc
    n_projections=8, rp_dim=30, k_mnn=15, consensus_threshold=0.3,
    k_spatial=6, n_comps=20, svd_dim_for_ctx=30,
    n_jobs=1, random_state=42, store_graphs=True, verbose=False, copy=True,
)


def _is_symmetric(W, tol=1e-6):
    D = (W.tocsr() - W.tocsr().T)
    return (abs(D).max() if D.nnz else 0.0) <= tol


def _no_self_loops(W):
    return float(np.abs(W.diagonal()).max()) == 0.0


@pytest.fixture(scope="module")
def adata():
    return _make_synthetic()


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_output_shape(adata):
    out = prime_st(adata.copy(), **PARAMS)
    assert out.obsm["X_prime"].shape == (adata.n_obs, PARAMS["n_comps"])


def test_graphs_symmetric_and_no_self_loops(adata):
    g = prime_st(adata.copy(), **PARAMS).uns["prime_graphs"]
    Wa, Ws = g["W_anchor"], g["W_spatial"]
    assert _is_symmetric(Wa) and _is_symmetric(Ws)
    assert _no_self_loops(Wa) and _no_self_loops(Ws)


def test_anchor_graph_equals_standalone_helper(adata):
    """prime_st's anchor graph must equal the standalone helper (reweighting off):
    this proves the Step-5 extraction did not change behavior."""
    p = dict(PARAMS, reweight_anchors_by_spatial_context=False)
    Wa_pipeline = prime_st(adata.copy(), **p).uns["prime_graphs"]["W_anchor"]

    # Reconstruct the helper's exact input: log1p-normalized, HVG-subset matrix.
    a = adata.copy()
    hvg = _pick_hvgs(a, batch_key="batch", n_hvg=PARAMS["n_hvg"], flavor="seurat")
    X_log_hvg = _row_norm_log1p(a.X)[:, hvg]
    Wa_helper = _build_rp_consensus_mnn_graph(
        X_log_hvg, a.obs["batch"].values,
        n_projections=PARAMS["n_projections"], rp_dim=PARAMS["rp_dim"],
        k_mnn=PARAMS["k_mnn"], consensus_threshold=PARAMS["consensus_threshold"],
        mnn_strategy="star", n_jobs=1, random_state=PARAMS["random_state"],
    )
    assert Wa_pipeline.nnz == Wa_helper.nnz
    D = (Wa_pipeline - Wa_helper)
    assert (abs(D).max() if D.nnz else 0.0) <= 1e-6


@pytest.mark.parametrize("strategy", ["star", "pairwise"])
def test_mnn_strategies(adata, strategy):
    g = prime_st(adata.copy(), **dict(PARAMS, mnn_strategy=strategy)).uns["prime_graphs"]
    assert g["W_anchor"].nnz > 0 and _is_symmetric(g["W_anchor"])
    assert g["mnn_strategy"] == strategy


@pytest.mark.parametrize("context_method", ["svd", "random_projection"])
def test_context_methods_run(adata, context_method):
    g = prime_st(adata.copy(), **dict(PARAMS, context_method=context_method)).uns["prime_graphs"]
    assert g["context_method"] == context_method
    assert g["anchor_projection_method"] == "random_projection"


def test_context_method_does_not_change_anchor_topology(adata):
    """Z_gate only reweights anchors; the anchor edge *set* must be identical
    regardless of context_method."""
    a = prime_st(adata.copy(), **dict(PARAMS, context_method="svd"))
    b = prime_st(adata.copy(), **dict(PARAMS, context_method="random_projection"))
    Wa_a = a.uns["prime_graphs"]["W_anchor"]
    Wa_b = b.uns["prime_graphs"]["W_anchor"]
    assert set(zip(*Wa_a.nonzero())) == set(zip(*Wa_b.nonzero()))


def test_experimental_base_embedding(adata):
    g = prime_st(adata.copy(), **dict(PARAMS, base_embedding_method="random_projection"))
    assert g.obsm["X_prime"].shape == (adata.n_obs, PARAMS["n_comps"])
    assert g.uns["prime_graphs"]["base_embedding_method"] == "random_projection"


def test_metadata_records_methods_and_seed(adata):
    g = prime_st(adata.copy(), **PARAMS).uns["prime_graphs"]
    for key in ("anchor_projection_method", "context_method", "base_embedding_method",
                "rp_dim", "n_projections", "random_state", "mnn_strategy"):
        assert key in g


def test_deterministic_for_fixed_seed(adata):
    a = prime_st(adata.copy(), **PARAMS).uns["prime_graphs"]["W_anchor"]
    b = prime_st(adata.copy(), **PARAMS).uns["prime_graphs"]["W_anchor"]
    assert a.nnz == b.nnz
    D = (a - b)
    assert (abs(D).max() if D.nnz else 0.0) == 0.0

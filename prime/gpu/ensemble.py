"""GPU-accelerated ensemble MNN batch correction.

This is the GPU counterpart of ``prime.core.ensemble_mnn_correct``. It is
designed to keep VRAM bounded so 1M+ cells fit on a 16 GB GPU:

  - Expression matrix stays sparse on GPU (cupy.scipy.sparse CSR).
  - Each random projection runs as a single cuSPARSE sparse @ dense matmul.
  - The projected embedding is downcast to fp16 for the kNN/MNN step.
  - kNN is run in chunks via cuvs (or faiss-gpu) so the query-side VRAM
    footprint is O(chunk_size * d), not O(n_cells * d).
  - The memory pool is freed between projections to avoid fragmentation
    OOM after long-running ensembles.
  - The actual correction (gene-wise additive update + Gaussian smoothing)
    is still performed on CPU in gene-chunks: it is already memory-bounded
    by ``chunk_size`` and densifies the output anyway.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from anndata import AnnData
from scipy import sparse

from prime.core import _compute_smoothing_matrix
from prime.gpu._backend import free_pool, is_oom_error, require_ann
from prime.gpu._knn import chunked_knn, mutual_neighbors


__all__ = ["ensemble_mnn_correct"]


def _to_gpu_csr(X):
    """Move X to GPU as a cupyx.scipy.sparse CSR (float32)."""
    import cupy as cp
    import cupyx.scipy.sparse as cpsp

    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    if not sparse.isspmatrix_csr(X):
        X = X.tocsr()
    X = X.astype(np.float32, copy=False)
    return cpsp.csr_matrix(
        (cp.asarray(X.data), cp.asarray(X.indices), cp.asarray(X.indptr)),
        shape=X.shape,
    )


def _l2_normalize_rows_gpu(X_gpu):
    """L2-normalize each row of a sparse CSR GPU matrix in place."""
    import cupy as cp

    sq = X_gpu.multiply(X_gpu).sum(axis=1)  # (n, 1)
    norms = cp.sqrt(cp.asarray(sq).ravel())
    norms[norms == 0] = 1.0
    inv = (1.0 / norms).astype(cp.float32)
    # Scale rows via the CSR data buffer.
    indptr = X_gpu.indptr
    data = X_gpu.data
    # Build a per-nnz scaling vector.
    row_lens = cp.diff(indptr)
    scale = cp.repeat(inv, row_lens.tolist() if row_lens.size < 1024 else row_lens)
    # Fallback for large row_lens repeats (cp.repeat supports array repeats since 11.x).
    data *= scale
    return X_gpu


def _consensus_edges_to_csr(
    rows: np.ndarray,
    cols: np.ndarray,
    n_cells: int,
    n_projections: int,
    consensus_threshold: float,
) -> sparse.csr_matrix:
    """Aggregate edges from all projections, threshold, and symmetrize.

    Each (row, col) appearance contributes 1/n_projections of weight. Edges
    surviving ``consensus_threshold`` are kept; their final weight is the
    fraction of projections that voted for them.
    """
    if rows.size == 0:
        return sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)

    keys = rows.astype(np.int64) * np.int64(n_cells) + cols.astype(np.int64)
    uniq, counts = np.unique(keys, return_counts=True)
    freq = counts.astype(np.float32) / float(n_projections)
    keep = freq >= float(consensus_threshold)
    if not np.any(keep):
        return sparse.csr_matrix((n_cells, n_cells), dtype=np.float32)
    uniq_k = uniq[keep]
    weights = freq[keep]
    r = (uniq_k // n_cells).astype(np.int32)
    c = (uniq_k % n_cells).astype(np.int32)
    G = sparse.csr_matrix((weights, (r, c)), shape=(n_cells, n_cells))
    return G.maximum(G.T).tocsr()


def _gpu_consensus_graph(
    X,
    batch_labels: np.ndarray,
    n_projections: int,
    target_dim: int,
    k_neighbors: int,
    consensus_threshold: float,
    knn_chunk_size: int,
    knn_backend: str,
    projection_dtype: str,
    random_state: int,
    verbose: bool,
) -> sparse.csr_matrix:
    """Build the consensus MNN graph on GPU and return it on CPU as sparse."""
    import cupy as cp

    require_ann()

    X_gpu = _to_gpu_csr(X)
    X_gpu = _l2_normalize_rows_gpu(X_gpu)

    n_cells, n_genes = X_gpu.shape
    rng = np.random.default_rng(random_state)
    unique_batches = np.unique(batch_labels)
    batch_idx = {b: np.where(batch_labels == b)[0] for b in unique_batches}

    pdtype = cp.float16 if projection_dtype == "float16" else cp.float32

    all_rows: List[np.ndarray] = []
    all_cols: List[np.ndarray] = []

    for t in range(int(n_projections)):
        if verbose:
            print(f"[prime.gpu] projection {t + 1}/{n_projections}")

        seed_t = int(rng.integers(0, 2**31 - 1))
        gpu_rng = cp.random.default_rng(seed_t)
        # Sparse-friendly Gaussian random projection.
        R = gpu_rng.standard_normal(
            (n_genes, int(target_dim)), dtype=cp.float32
        ) / cp.sqrt(cp.float32(target_dim))

        Z = (X_gpu @ R).astype(pdtype)
        del R
        free_pool()

        # Need Z accessible as a contiguous fp32 cupy array for ANN backends.
        Z_fp32 = Z.astype(cp.float32) if Z.dtype != cp.float32 else Z

        # All pairs of batches.
        for i, b1 in enumerate(unique_batches):
            for b2 in unique_batches[i + 1 :]:
                idx_a = batch_idx[b1]
                idx_b = batch_idx[b2]
                Z_a = Z_fp32[cp.asarray(idx_a)]
                Z_b = Z_fp32[cp.asarray(idx_b)]

                nbrs_ab = chunked_knn(
                    Z_b, Z_a, k=int(k_neighbors),
                    chunk_size=int(knn_chunk_size),
                    backend=knn_backend, free_between_chunks=True,
                )
                nbrs_ba = chunked_knn(
                    Z_a, Z_b, k=int(k_neighbors),
                    chunk_size=int(knn_chunk_size),
                    backend=knn_backend, free_between_chunks=True,
                )
                del Z_a, Z_b
                free_pool()

                rr, cc = mutual_neighbors(nbrs_ab, nbrs_ba, idx_a, idx_b)
                if rr.size:
                    all_rows.append(rr)
                    all_cols.append(cc)
                    # Symmetric: also append the reverse edges.
                    all_rows.append(cc)
                    all_cols.append(rr)

        del Z, Z_fp32
        free_pool()

    del X_gpu
    free_pool()

    rows = np.concatenate(all_rows) if all_rows else np.empty(0, dtype=np.int64)
    cols = np.concatenate(all_cols) if all_cols else np.empty(0, dtype=np.int64)
    return _consensus_edges_to_csr(
        rows, cols, n_cells, n_projections, consensus_threshold
    )


def _apply_correction_cpu(
    X,
    consensus_graph: sparse.csr_matrix,
    batch_labels: np.ndarray,
    sigma: float,
    chunk_size: int,
    verbose: bool,
) -> np.ndarray:
    """CPU correction: gene-chunked additive update + Gaussian smoothing.

    Mirrors the second half of ``prime.core.ensemble_mnn_correct`` but
    extracted so the GPU path can reuse it.
    """
    n_cells, n_genes = X.shape
    rows, cols = consensus_graph.nonzero()
    weights = consensus_graph.data

    # Cross-batch only.
    mask = batch_labels[rows] != batch_labels[cols]
    rows = rows[mask]
    cols = cols[mask]
    weights = weights[mask]

    if rows.size == 0:
        if verbose:
            print("[prime.gpu] No cross-batch consensus edges. Returning a copy of X.")
        return X.toarray() if sparse.issparse(X) else X.copy()

    weight_sums = np.zeros(n_cells, dtype=np.float64)
    np.add.at(weight_sums, rows, weights)

    smoothing_mat = _compute_smoothing_matrix(X, sigma=sigma)

    X_corrected = np.zeros((n_cells, n_genes), dtype=np.float32)
    cs = min(int(chunk_size), n_genes)
    for i in range(0, n_genes, cs):
        end = min(i + cs, n_genes)
        if sparse.issparse(X):
            X_chunk = X[:, i:end].toarray()
        else:
            X_chunk = np.asarray(X[:, i:end])

        raw_corr = np.zeros_like(X_chunk, dtype=np.float64)
        diffs = X_chunk[cols] - X_chunk[rows]
        weighted = diffs * weights[:, np.newaxis]
        np.add.at(raw_corr, rows, weighted)

        good = weight_sums > 0
        raw_corr[good] /= weight_sums[good][:, np.newaxis]

        final = smoothing_mat.dot(raw_corr)
        X_corrected[:, i:end] = (X_chunk + final).astype(np.float32)

    return X_corrected


def ensemble_mnn_correct(
    adata: AnnData,
    batch_key: str,
    *,
    n_projections: int = 10,
    target_dim: int = 50,
    k_neighbors: int = 20,
    consensus_threshold: float = 0.4,
    sigma: float = 0.1,
    random_state: int = 42,
    chunk_size: int = 2000,
    knn_chunk_size: int = 50_000,
    knn_backend: str = "auto",
    projection_dtype: str = "float16",
    verbose: bool = True,
) -> np.ndarray:
    """GPU-accelerated ensemble MNN batch correction (1M+ cells).

    Same algorithm as :func:`prime.ensemble_mnn_correct`, but:

      - Random projections run on GPU via cuSPARSE.
      - kNN runs on GPU via cuvs (or faiss-gpu), chunked to bound VRAM.
      - fp16 is used for the projection / kNN intermediate (controlled by
        ``projection_dtype``).
      - cupy's memory pool is freed between projections to prevent
        fragmentation-driven OOM.
      - The correction step itself runs on CPU in gene-chunks
        (``chunk_size``), since it already has bounded memory and outputs
        a dense matrix that's usually too large to keep on GPU anyway.

    Parameters
    ----------
    adata
        AnnData with expression in ``.X`` and batch labels in
        ``.obs[batch_key]``. ``.X`` may be sparse or dense; sparse is
        strongly recommended for >100K cells.
    batch_key
        Column in ``adata.obs`` holding batch IDs.
    n_projections, target_dim, k_neighbors, consensus_threshold, sigma,
    random_state, chunk_size
        Same meaning as :func:`prime.ensemble_mnn_correct`.
    knn_chunk_size
        Maximum number of query rows per ANN call. Lower this if VRAM is
        tight; raise for throughput. Default 50,000 fits comfortably in
        16 GB even at k=50.
    knn_backend
        ``"auto"``, ``"cuvs"``, or ``"faiss"``. Default tries cuvs first.
    projection_dtype
        ``"float16"`` (default, half VRAM) or ``"float32"``.
    verbose
        Print progress per projection.

    Returns
    -------
    np.ndarray (n_cells, n_genes), float32
        Batch-corrected expression matrix. Assign back to ``adata.X`` or
        ``adata.layers[...]`` as you would with the CPU version.

    Notes
    -----
    Requires cupy and a GPU ANN backend. Install with::

        pip install "prime-sc[gpu-cuvs]"     # recommended
        # or
        pip install "prime-sc[gpu-faiss]"

    plus a cupy build matching your CUDA version (e.g. cupy-cuda12x).
    """
    if batch_key not in adata.obs:
        raise ValueError(f"Batch key {batch_key!r} not found in adata.obs")
    if projection_dtype not in {"float16", "float32"}:
        raise ValueError(
            f"projection_dtype must be 'float16' or 'float32', got {projection_dtype!r}"
        )

    batch_labels = adata.obs[batch_key].values

    try:
        consensus = _gpu_consensus_graph(
            adata.X,
            batch_labels,
            n_projections=n_projections,
            target_dim=target_dim,
            k_neighbors=k_neighbors,
            consensus_threshold=consensus_threshold,
            knn_chunk_size=knn_chunk_size,
            knn_backend=knn_backend,
            projection_dtype=projection_dtype,
            random_state=random_state,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001 — re-raised below unless OOM
        # Reclaim whatever VRAM we can so the caller can retry with smaller
        # settings (or fall back to the CPU path) in the same process.
        free_pool()
        if is_oom_error(exc):
            raise MemoryError(
                "prime.gpu.ensemble_mnn_correct ran out of GPU memory while "
                "building the consensus graph. The kNN step already backs off "
                "its chunk size automatically, so the bottleneck is most likely "
                "the sparse expression matrix or a single random projection "
                "living on the GPU. Try: lower target_dim "
                f"(currently {target_dim}) or k_neighbors (currently "
                f"{k_neighbors}); keep projection_dtype='float16'; subset to "
                "highly-variable genes before calling; or use a larger-VRAM "
                "GPU. The CPU equivalent prime.ensemble_mnn_correct has no VRAM "
                "limit and produces an equivalent result."
            ) from exc
        raise
    if verbose:
        print(f"[prime.gpu] consensus graph: {consensus.nnz} edges")

    return _apply_correction_cpu(
        adata.X,
        consensus,
        batch_labels,
        sigma=sigma,
        chunk_size=chunk_size,
        verbose=verbose,
    )

"""Streaming preprocessing for very large AnnData (1M+ cells).

All routines here keep memory at O(n_genes * n_batches) rather than
O(n_cells * n_genes), so they can run on commodity machines even when the
full expression matrix would exceed RAM.

Backed-mode AnnData is supported (and recommended) for inputs larger than
RAM. Pass either an in-memory AnnData or a path to an .h5ad file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import anndata as ad
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "prime.pp.streaming requires anndata. "
        "Install with `pip install anndata`."
    ) from e


__all__ = [
    "streaming_hvg",
    "streaming_gene_filter",
    "normalize_log1p_sparse",
]


def normalize_log1p_sparse(
    X: Union[sparse.spmatrix, np.ndarray],
    target_sum: float = 1e4,
) -> Union[sparse.spmatrix, np.ndarray]:
    """Row-normalize to ``target_sum`` then log1p, preserving sparsity.

    Works on both sparse and dense inputs; the sparse path never densifies.
    """
    if sparse.issparse(X):
        X = X.astype(np.float32, copy=True)
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()
        row_sums = np.asarray(X.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        scale = (target_sum / row_sums).astype(np.float32)
        # Scale rows in place via the CSR data buffer (avoids a diags @ X copy).
        row_starts = X.indptr[:-1]
        row_ends = X.indptr[1:]
        for i in range(X.shape[0]):
            X.data[row_starts[i] : row_ends[i]] *= scale[i]
        X.data = np.log1p(X.data)
        return X

    X = np.asarray(X, dtype=np.float32)
    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    X = X * (target_sum / row_sums)
    return np.log1p(X)


class _BackedReader:
    """Wrap an .h5ad file path and read row chunks of X directly via h5py.

    This bypasses anndata's backed-mode sparse-dataset class, which is
    incompatible with scipy >= 1.15 (the ``_validate_indices`` API was
    removed). For dense X we read straight from the HDF5 dataset; for
    sparse X we slice the (data, indices, indptr) buffers ourselves.
    """

    def __init__(self, path: Union[str, Path]):
        import h5py  # local import: only needed for backed path

        self._h5 = h5py.File(str(path), "r")
        # Read obs / var via anndata for convenience, but keep X access via h5py.
        meta = ad.read_h5ad(str(path), backed="r")
        self.obs = meta.obs.copy()
        self.var = meta.var.copy()
        self.shape = (meta.n_obs, meta.n_vars)
        meta.file.close()

        x = self._h5["X"]
        if hasattr(x, "keys"):
            self._sparse = True
            # h5py loads these lazily; keep handles around.
            self._indptr = x["indptr"]
            self._indices = x["indices"]
            self._data = x["data"]
        else:
            self._sparse = False
            self._x = x

    @property
    def n_obs(self) -> int:
        return self.shape[0]

    @property
    def n_vars(self) -> int:
        return self.shape[1]

    @property
    def isbacked(self) -> bool:
        return True

    def chunk_X(self, start: int, end: int):
        if self._sparse:
            ip = self._indptr[start : end + 1]
            off = int(ip[0])
            nnz = int(ip[-1] - off)
            data = self._data[off : off + nnz]
            indices = self._indices[off : off + nnz]
            indptr = (ip - off).astype(np.int64)
            return sparse.csr_matrix(
                (data, indices, indptr),
                shape=(int(end - start), self.shape[1]),
            )
        return np.asarray(self._x[start:end])

    def close(self) -> None:
        try:
            self._h5.close()
        except Exception:
            pass


def _open_input(adata_or_path):
    """Return (reader, opened_flag).

    The reader is either:
      - a ``_BackedReader`` (when input is a path), or
      - the AnnData object itself (in-memory),
    and exposes ``.shape``, ``.obs``, ``.isbacked``, and a ``chunk_X``-like
    way to get row slices via :func:`_chunk_X`.
    """
    if isinstance(adata_or_path, (str, Path)):
        return _BackedReader(adata_or_path), True
    return adata_or_path, False


def _chunk_X(reader, start: int, end: int) -> Union[sparse.spmatrix, np.ndarray]:
    """Read a row slice of X, handling backed and in-memory cases uniformly."""
    if isinstance(reader, _BackedReader):
        return reader.chunk_X(start, end)
    X = reader.X[start:end]
    if hasattr(X, "to_memory"):
        X = X.to_memory()
    return X


def streaming_gene_filter(
    adata_or_path,
    min_cells: int = 10,
    chunk_size: int = 20_000,
) -> np.ndarray:
    """Return a boolean mask of genes expressed in >= ``min_cells`` cells.

    One streaming pass; peak memory ~ chunk_size rows of X.

    Parameters
    ----------
    adata_or_path
        AnnData object (in-memory or backed) or path to an .h5ad file.
    min_cells
        Drop genes detected in fewer than this many cells.
    chunk_size
        Cells per streamed chunk.

    Returns
    -------
    np.ndarray (bool, shape=(n_genes,))
        Mask of genes to keep.
    """
    reader, opened = _open_input(adata_or_path)
    try:
        n_cells, n_genes = reader.shape
        nnz_per_gene = np.zeros(n_genes, dtype=np.int64)
        for s in range(0, n_cells, chunk_size):
            e = min(s + chunk_size, n_cells)
            X = _chunk_X(reader, s, e)
            if sparse.issparse(X):
                nnz_per_gene += np.asarray((X != 0).sum(axis=0)).ravel()
            else:
                nnz_per_gene += (X != 0).sum(axis=0).astype(np.int64)
        return nnz_per_gene >= int(min_cells)
    finally:
        if opened:
            reader.close()


def streaming_hvg(
    adata_or_path,
    batch_key: Optional[str] = None,
    n_top_genes: int = 3000,
    chunk_size: int = 20_000,
    min_cells: int = 10,
    target_sum: float = 1e4,
    aggregation: str = "median_rank",
    flavor: str = "seurat",
) -> np.ndarray:
    """Compute highly-variable genes in O(n_genes * n_batches) memory.

    Two streaming passes over the data:
      1. Gene non-zero counts (to filter sparsely expressed genes).
      2. Per-batch streaming mean & second moment on log1p-normalized
         expression; produces a per-batch normalized-variance score that
         is aggregated across batches.

    Peak memory is roughly ``chunk_size * n_genes * 12 bytes`` (one CSR
    chunk) plus ``n_batches * n_genes * 16 bytes`` (sufficient statistics).
    Works equally well for 100K or 10M cells.

    Parameters
    ----------
    adata_or_path
        AnnData (in-memory or backed) or path to an .h5ad file. Backed mode
        is recommended for >1M cells.
    batch_key
        Optional column in ``adata.obs`` for batch-aware HVG selection.
        If given, HVGs are scored per batch and aggregated by
        ``aggregation``. If None, all cells are treated as one batch.
    n_top_genes
        Number of top genes to mark as highly variable.
    chunk_size
        Cells per streamed chunk. Higher = faster but more RAM.
    min_cells
        Pre-filter: drop genes detected in fewer than this many cells.
    target_sum
        Per-cell normalization target (passed to ``log1p`` normalization).
    aggregation
        How to combine per-batch scores when ``batch_key`` is given:
          - ``"median_rank"``: rank genes within each batch by normalized
            variance, take median rank across batches, pick top
            ``n_top_genes`` by median rank (close to Seurat-v3 behavior).
          - ``"mean_score"``: average normalized variance across batches,
            then pick top ``n_top_genes``.
          - ``"union"``: top ``n_top_genes`` per batch, take the union.
    flavor
        Currently only ``"seurat"`` (log1p-normalized variance/mean) is
        supported in the streaming path. Seurat-v3-style raw-count Loess
        fitting requires a different algorithm and is not implemented here.

    Returns
    -------
    np.ndarray (bool, shape=(n_genes,))
        Mask of selected HVGs.
    """
    if flavor != "seurat":
        raise NotImplementedError(
            "streaming_hvg currently supports flavor='seurat' only. "
            "For seurat_v3-style HVG on large data, subsample to ~100k cells "
            "and run scanpy.pp.highly_variable_genes(flavor='seurat_v3')."
        )
    if aggregation not in {"median_rank", "mean_score", "union"}:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")

    reader, opened = _open_input(adata_or_path)
    try:
        n_cells, n_genes = reader.shape

        # ---- Pass 1: gene non-zero counts ----
        nnz_per_gene = np.zeros(n_genes, dtype=np.int64)
        for s in range(0, n_cells, chunk_size):
            e = min(s + chunk_size, n_cells)
            X = _chunk_X(reader, s, e)
            if sparse.issparse(X):
                nnz_per_gene += np.asarray((X != 0).sum(axis=0)).ravel()
            else:
                nnz_per_gene += (X != 0).sum(axis=0).astype(np.int64)
        gene_keep = nnz_per_gene >= int(min_cells)

        # ---- Batch codes ----
        if batch_key is not None:
            if batch_key not in reader.obs:
                raise ValueError(f"batch_key {batch_key!r} not in adata.obs")
            batch_codes = reader.obs[batch_key].astype("category").cat.codes.values
            n_batches = int(batch_codes.max()) + 1
        else:
            batch_codes = np.zeros(n_cells, dtype=np.int32)
            n_batches = 1

        # ---- Pass 2: per-batch streaming sufficient statistics ----
        sum_x = np.zeros((n_batches, n_genes), dtype=np.float64)
        sum_x2 = np.zeros((n_batches, n_genes), dtype=np.float64)
        counts = np.zeros(n_batches, dtype=np.int64)

        for s in range(0, n_cells, chunk_size):
            e = min(s + chunk_size, n_cells)
            X = _chunk_X(reader, s, e)
            b = batch_codes[s:e]

            X_norm = normalize_log1p_sparse(X, target_sum=target_sum)

            for bid in np.unique(b):
                mask = b == bid
                Xb = X_norm[mask]
                counts[bid] += Xb.shape[0]
                if sparse.issparse(Xb):
                    sum_x[bid] += np.asarray(Xb.sum(axis=0)).ravel()
                    sum_x2[bid] += np.asarray(Xb.multiply(Xb).sum(axis=0)).ravel()
                else:
                    sum_x[bid] += Xb.sum(axis=0)
                    sum_x2[bid] += (Xb * Xb).sum(axis=0)

        # ---- Per-batch normalized variance ----
        safe_counts = np.maximum(counts, 1).astype(np.float64)
        means = sum_x / safe_counts[:, None]
        var = sum_x2 / safe_counts[:, None] - means**2
        var = np.maximum(var, 0.0)
        nvar = var / np.maximum(means, 1e-12)

        # Mask out filtered genes (set score to -inf so they sort last)
        nvar[:, ~gene_keep] = -np.inf

        # Batches that ended up with 0 cells should not contribute
        active = counts > 0
        if not np.any(active):
            raise RuntimeError("No cells in any batch — cannot compute HVG.")
        nvar_active = nvar[active]

        # ---- Aggregate across batches ----
        if aggregation == "median_rank":
            ranks = (-nvar_active).argsort(axis=1).argsort(axis=1)
            score = np.median(ranks, axis=0)
            top_idx = np.argsort(score)[: int(n_top_genes)]
        elif aggregation == "mean_score":
            score = nvar_active.mean(axis=0)
            top_idx = np.argsort(-score)[: int(n_top_genes)]
        elif aggregation == "union":
            top_idx = set()
            for row in nvar_active:
                top_idx |= set(np.argsort(-row)[: int(n_top_genes)].tolist())
            top_idx = np.fromiter(top_idx, dtype=np.int64)

        hvg_mask = np.zeros(n_genes, dtype=bool)
        hvg_mask[top_idx] = True
        # Honor the min_cells pre-filter even if a filtered gene snuck in
        hvg_mask &= gene_keep
        return hvg_mask

    finally:
        if opened:
            reader.close()

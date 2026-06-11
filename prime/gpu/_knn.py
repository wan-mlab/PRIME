"""Chunked GPU approximate-nearest-neighbor search.

Wraps cuvs / faiss-gpu behind a single function ``chunked_knn`` that:
  - Builds an ANN index on ``data`` (on GPU)
  - Queries ``queries`` in chunks of at most ``chunk_size`` rows
  - Returns neighbor indices on CPU (as numpy int64)

Chunking caps the query-side VRAM footprint so this works on 16 GB GPUs
for 1M+ points at k <= 50.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from prime.gpu._backend import (
    detect,
    free_pool,
    is_oom_error,
    pick_knn_backend,
    require_ann,
)


def _to_gpu_fp32(arr):
    import cupy as cp

    if isinstance(arr, cp.ndarray):
        return arr.astype(cp.float32, copy=False)
    return cp.asarray(np.ascontiguousarray(arr, dtype=np.float32))


def _cuvs_search(data_gpu, queries_gpu, k: int):
    """Build a CAGRA index on data_gpu and search queries_gpu."""
    from cuvs.neighbors import cagra  # type: ignore

    build_params = cagra.IndexParams(metric="sqeuclidean")
    index = cagra.build(build_params, data_gpu)
    search_params = cagra.SearchParams()
    distances, indices = cagra.search(search_params, index, queries_gpu, k)
    return indices  # cupy array (n_queries, k)


def _faiss_search(data_gpu, queries_gpu, k: int):
    """Build a flat L2 index on GPU and search."""
    import cupy as cp
    import faiss  # type: ignore

    data_np = cp.asnumpy(data_gpu).astype(np.float32)
    queries_np = cp.asnumpy(queries_gpu).astype(np.float32)
    d = data_np.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    gpu_index.add(data_np)
    _, indices = gpu_index.search(queries_np, k)
    return cp.asarray(indices, dtype=cp.int64)


def chunked_knn(
    data,
    queries,
    k: int,
    chunk_size: int = 50_000,
    backend: str = "auto",
    free_between_chunks: bool = True,
    min_chunk_size: int = 1024,
) -> np.ndarray:
    """Find k nearest neighbors in ``data`` for each row of ``queries``.

    The query side is processed in chunks of at most ``chunk_size`` rows. If a
    chunk triggers a GPU out-of-memory error, the chunk size is halved and the
    *same* chunk is retried, down to ``min_chunk_size``; only if a chunk that
    small still cannot fit is the error re-raised. This makes the query-side
    VRAM footprint self-adapting rather than crashing on the first chunk that is
    too large for the currently free VRAM.

    Parameters
    ----------
    data
        (n_index, d) array (cupy or numpy). Will be moved to GPU.
    queries
        (n_query, d) array.
    k
        Number of neighbors to return.
    chunk_size
        Maximum number of query rows processed per ANN call, used as the
        *starting* size. Lower it if VRAM is tight; raise for throughput. It is
        reduced automatically on OOM (see ``min_chunk_size``).
    backend
        ``"auto"`` (cuvs > faiss), ``"cuvs"``, or ``"faiss"``.
    free_between_chunks
        Release cupy's memory pool between query chunks. Prevents
        fragmentation-driven OOM on long-running pipelines.
    min_chunk_size
        Smallest query chunk the adaptive backoff will fall to before giving up
        and re-raising the OOM. Defaults to 1024.

    Returns
    -------
    np.ndarray (int64, shape=(n_query, k))
        Neighbor indices in ``data``. Returned on CPU.

    Raises
    ------
    MemoryError
        If a chunk of ``min_chunk_size`` rows still does not fit in VRAM. The
        index side (``data``) and ``k`` dominate what is left; reduce
        ``target_dim``/``k`` or use a larger-VRAM GPU.
    """
    env = require_ann()
    backend = pick_knn_backend(backend, env)

    import cupy as cp

    data_gpu = _to_gpu_fp32(data)
    n_query = queries.shape[0]
    out = np.empty((n_query, int(k)), dtype=np.int64)

    cur_chunk = max(1, int(chunk_size))
    floor = max(1, min(int(min_chunk_size), cur_chunk))

    s = 0
    while s < n_query:
        e = min(s + cur_chunk, n_query)
        q_gpu = idx_gpu = None
        try:
            q_gpu = _to_gpu_fp32(queries[s:e])
            if backend == "cuvs":
                idx_gpu = _cuvs_search(data_gpu, q_gpu, k=int(k))
            else:
                idx_gpu = _faiss_search(data_gpu, q_gpu, k=int(k))
            out[s:e] = cp.asnumpy(idx_gpu).astype(np.int64, copy=False)
        except Exception as exc:  # noqa: BLE001 — re-raised below unless OOM
            if not is_oom_error(exc):
                raise
            # Drop partial allocations from the failed attempt, reclaim VRAM,
            # and retry the SAME chunk with a smaller size.
            del q_gpu, idx_gpu
            free_pool()
            if cur_chunk <= floor:
                raise MemoryError(
                    f"GPU out of memory in chunked_knn even at "
                    f"chunk_size={cur_chunk} (k={k}, "
                    f"n_index={int(data_gpu.shape[0])}, "
                    f"dim={int(data_gpu.shape[1])}). Reduce target_dim or k, "
                    f"set projection_dtype='float16', or use a larger-VRAM GPU."
                ) from exc
            cur_chunk = max(floor, cur_chunk // 2)
            continue  # retry [s, s + cur_chunk) without advancing s
        # Success: release this chunk's buffers and advance.
        del q_gpu, idx_gpu
        if free_between_chunks:
            free_pool()
        s = e

    del data_gpu
    if free_between_chunks:
        free_pool()
    return out


def mutual_neighbors(
    nbrs_ab: np.ndarray,
    nbrs_ba: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mutual nearest neighbor edges between two cell sets A and B.

    Parameters
    ----------
    nbrs_ab
        (|A|, k) array: ``nbrs_ab[i, :]`` are indices into B of the k nearest
        B-cells to A[i].
    nbrs_ba
        (|B|, k) array: nearest A-cells of each B-cell, indexed into A.
    idx_a, idx_b
        Global cell indices for sets A and B.

    Returns
    -------
    rows, cols : np.ndarray, np.ndarray
        Edge endpoints in the global index space.
    """
    n_a, k = nbrs_ab.shape
    rows_list, cols_list = [], []

    # Build a set of (a_local, b_local) candidate edges from A's perspective.
    a_rep = np.repeat(np.arange(n_a, dtype=np.int64), k)
    b_cand = nbrs_ab.ravel().astype(np.int64)
    cand_ab = a_rep * (np.int64(nbrs_ba.shape[0]) + 1) + b_cand

    # And the reciprocal set from B's perspective (note transposed key order).
    n_b = nbrs_ba.shape[0]
    b_rep = np.repeat(np.arange(n_b, dtype=np.int64), k)
    a_cand = nbrs_ba.ravel().astype(np.int64)
    cand_ba = a_cand * (np.int64(n_b) + 1) + b_rep

    mutual_keys = np.intersect1d(cand_ab, cand_ba, assume_unique=False)
    if mutual_keys.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    a_local = (mutual_keys // (n_b + 1)).astype(np.int64)
    b_local = (mutual_keys % (n_b + 1)).astype(np.int64)
    rows_list.append(idx_a[a_local])
    cols_list.append(idx_b[b_local])

    return np.concatenate(rows_list), np.concatenate(cols_list)

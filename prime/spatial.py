from __future__ import annotations

# ---- Standard / typing ----
from inspect import signature
from typing import Optional, Tuple, Dict, Any

# ---- Numeric / sparse ----
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, issparse, diags, identity
from scipy.sparse.linalg import cg, LinearOperator

# ---- ML utilities ----
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

# ---- Single-cell ----
import scanpy as sc
from anndata import AnnData


# ============================================================
# 1. Low-level utilities
# ============================================================

def _as_csr(X) -> csr_matrix:
    """Make sure X is a CSR sparse matrix."""
    return X.tocsr() if issparse(X) else csr_matrix(X)


def _row_norm_log1p(X, target_sum: float = 1e4) -> csr_matrix:
    """
    Library-size normalize each cell (row) to `target_sum`, then apply log1p.
    Equivalent to scanpy.pp.normalize_total + scanpy.pp.log1p, but kept sparse.
    """
    X = _as_csr(X).astype(np.float32)
    rs = np.asarray(X.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0  # avoid divide-by-zero for empty cells
    scale = (target_sum / rs).astype(np.float32)
    X = X.multiply(scale[:, None])
    X.data = np.log1p(X.data).astype(np.float32)
    return X.tocsr()


def _pick_hvgs(
    adata: AnnData,
    *,
    batch_key: str,
    n_hvg: int,
    flavor: str = "seurat_v3",
    layer: Optional[str] = None,
) -> np.ndarray:
    """Batch-aware HVG selection. Writes adata.var['highly_variable']."""
    X_for_hvg = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    tmp = AnnData(
        X=_as_csr(X_for_hvg),
        obs=adata.obs[[batch_key]].copy(),
        var=adata.var.copy(),
    )
    sc.pp.highly_variable_genes(
        tmp,
        batch_key=batch_key,
        n_top_genes=n_hvg,
        flavor=flavor,
        subset=False,
        inplace=True,
    )
    adata.var["highly_variable"] = tmp.var["highly_variable"].values
    return adata.var["highly_variable"].values.astype(bool)


def _svd_embedding(X: csr_matrix, n_comps: int, random_state: int = 0) -> np.ndarray:
    """Sparse-friendly TruncatedSVD embedding (essentially PCA without centering)."""
    svd = TruncatedSVD(n_components=n_comps, random_state=random_state)
    Z = svd.fit_transform(X)
    return Z.astype(np.float32)


# ============================================================
# 2. MNN (Mutual Nearest Neighbors) helpers
# ============================================================

def _find_mutual_nn(a_to_b: np.ndarray, b_to_a: np.ndarray) -> np.ndarray:
    """Given kNN index arrays, return mutual NN pairs as (i_in_a, j_in_b)."""
    forward = set()
    for i in range(a_to_b.shape[0]):
        for j in a_to_b[i]:
            forward.add((i, j))

    pairs = []
    for j in range(b_to_a.shape[0]):
        for i in b_to_a[j]:
            if (i, j) in forward:
                pairs.append((i, j))

    return np.asarray(pairs, dtype=np.int64)


def _mnn_between_groups(
    X: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    k: int,
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """MNN between two index sets in a shared feature space X (dense)."""
    if len(idx1) < 2 or len(idx2) < 2:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    k12 = min(k, len(idx2))
    k21 = min(k, len(idx1))

    nn2 = NearestNeighbors(n_neighbors=k12, metric="euclidean", n_jobs=n_jobs).fit(X[idx2])
    _, a_to_b = nn2.kneighbors(X[idx1])

    nn1 = NearestNeighbors(n_neighbors=k21, metric="euclidean", n_jobs=n_jobs).fit(X[idx1])
    _, b_to_a = nn1.kneighbors(X[idx2])

    mutual = _find_mutual_nn(a_to_b, b_to_a)
    if mutual.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    r = idx1[mutual[:, 0]]
    c = idx2[mutual[:, 1]]
    return r.astype(np.int64), c.astype(np.int64)


def _multibatch_mnn_edges(
    X: np.ndarray,
    batch_labels: np.ndarray,
    *,
    k: int,
    strategy: str = "star",   # "star" or "pairwise"
    n_jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return symmetric (rows, cols) MNN edges across batches."""
    batches = np.unique(batch_labels)

    if strategy not in ("star", "pairwise"):
        raise ValueError("strategy must be 'star' or 'pairwise'")

    if strategy == "star":
        sizes = {b: int(np.sum(batch_labels == b)) for b in batches}
        ref = max(sizes, key=sizes.get)
        pairs_to_do = [(ref, b) for b in batches if b != ref]
    else:
        pairs_to_do = []
        for i, b1 in enumerate(batches):
            for b2 in batches[i + 1:]:
                pairs_to_do.append((b1, b2))

    rows, cols = [], []
    for b1, b2 in pairs_to_do:
        idx1 = np.where(batch_labels == b1)[0]
        idx2 = np.where(batch_labels == b2)[0]
        r, c = _mnn_between_groups(X, idx1, idx2, k=k, n_jobs=n_jobs)
        if r.size == 0:
            continue
        # symmetric edges
        rows.append(r); cols.append(c)
        rows.append(c); cols.append(r)

    if not rows:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    return np.concatenate(rows), np.concatenate(cols)


# ============================================================
# 3. Spatial graph + spatial context
# ============================================================

def _build_spatial_graph(
    spatial_coords: np.ndarray,
    batch_labels: np.ndarray,
    Z_gate: np.ndarray,
    *,
    k_spatial: int = 6,
    n_jobs: int = 1,
    spatial_weight_mode: str = "rbf",   # "rbf" or "binary"
    gate_by_expr: bool = True,
    expr_gate_beta: float = 1.0,
    eps: float = 1e-8,
) -> csr_matrix:
    """
    Within-batch spatial kNN graph (symmetric).
    Edge weight =  RBF(spatial distance) * RBF(expression distance)^beta
    """
    n = spatial_coords.shape[0]
    rows, cols, w_list = [], [], []

    for b in np.unique(batch_labels):
        idx = np.where(batch_labels == b)[0]
        if len(idx) < 2:
            continue
        coords = spatial_coords[idx]
        k = min(k_spatial, len(idx) - 1)
        if k < 1:
            continue

        nn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=n_jobs).fit(coords)
        dists, nbrs = nn.kneighbors(coords)

        r = np.repeat(idx, k)
        c = idx[nbrs.reshape(-1)]

        if spatial_weight_mode == "binary":
            w_sp = np.ones_like(r, dtype=np.float32)
        else:
            d2 = (dists.reshape(-1).astype(np.float32)) ** 2
            sigma2 = np.median(d2) + eps
            w_sp = np.exp(-d2 / (2.0 * sigma2)).astype(np.float32)

        if gate_by_expr:
            diff = Z_gate[r] - Z_gate[c]
            d2e = np.sum(diff * diff, axis=1).astype(np.float32)
            tau2 = np.median(d2e) + eps
            w_expr = np.exp(-d2e / (2.0 * tau2)).astype(np.float32)
            w_sp = w_sp * (w_expr ** expr_gate_beta)

        rows.append(r); cols.append(c); w_list.append(w_sp)
        rows.append(c); cols.append(r); w_list.append(w_sp)

    if not rows:
        return csr_matrix((n, n), dtype=np.float32)

    rows = np.concatenate(rows).astype(np.int64)
    cols = np.concatenate(cols).astype(np.int64)
    w = np.concatenate(w_list).astype(np.float32)

    Ws = coo_matrix((w, (rows, cols)), shape=(n, n), dtype=np.float32).tocsr()
    Ws.sum_duplicates()
    Ws.setdiag(0.0)
    Ws.eliminate_zeros()
    Ws = (Ws + Ws.T) * 0.5
    Ws.eliminate_zeros()
    return Ws


def _spatial_context_from_graph(Ws: csr_matrix, Z_ctx: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Spatial-context vector for each spot:
        S = D^{-1} W_s Z_ctx     (row-normalized neighborhood mean)
    Then L2-normalize each row so it acts like a direction in feature space.
    """
    d = np.asarray(Ws.sum(axis=1)).ravel().astype(np.float32)
    d[d == 0] = 1.0
    S = (Ws @ Z_ctx) / d[:, None]
    S = normalize(S, axis=1)
    return S.astype(np.float32)


def _rbf_similarity(A: np.ndarray, rows: np.ndarray, cols: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """RBF similarity between rows A[rows] and A[cols] using median heuristic for sigma^2."""
    diff = A[rows] - A[cols]
    d2 = np.sum(diff * diff, axis=1).astype(np.float32)
    sigma2 = np.median(d2) + eps
    return np.exp(-d2 / (2.0 * sigma2)).astype(np.float32)


# ============================================================
# 4. Laplacian + CG solver
# ============================================================

def _laplacian(W: csr_matrix) -> csr_matrix:
    """Unnormalized graph Laplacian L = D - W."""
    d = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
    return diags(d, offsets=0, format="csr") - W


def _cg_solve_matrix_rhs(
    A: csr_matrix,
    B: np.ndarray,
    *,
    tol: float = 1e-5,
    maxiter: int = 200,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Solve A X = B for multiple RHS columns using CG per column.
    Cross-version compatible with SciPy (old `tol=` and new `rtol=` APIs).
    Uses Jacobi (diagonal) preconditioner.
    """
    n, k = B.shape

    diagA = A.diagonal().astype(np.float64)
    diagA[diagA == 0] = 1.0
    M = LinearOperator((n, n), matvec=lambda x: x / diagA, dtype=np.float64)

    cg_params = signature(cg).parameters
    use_new_api = ("rtol" in cg_params)

    X = np.zeros_like(B, dtype=np.float64)
    infos = []

    for j in range(k):
        b = B[:, j].astype(np.float64)
        x0 = b.copy()
        if use_new_api:
            x, info = cg(A, b, x0=x0, rtol=tol, atol=0.0, maxiter=maxiter, M=M)
        else:
            x, info = cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
        X[:, j] = x
        infos.append(int(info))

    stats = {
        "cg_info_per_dim": infos,
        "cg_converged_dims": int(np.sum(np.array(infos) == 0)),
        "cg_failed_dims": int(np.sum(np.array(infos) != 0)),
        "cg_api": "rtol/atol" if use_new_api else "tol",
    }
    return X.astype(np.float32), stats


# ============================================================
# 5. Main API: PRIME
# ============================================================

def prime_st(
    adata: AnnData,
    *,
    batch_key: str,
    spatial_key: str = "spatial",
    layer: Optional[str] = None,
    # HVG / normalization
    n_hvg: int = 3000,
    hvg_flavor: str = "seurat_v3",
    target_sum: float = 1e4,
    # ERP MNN anchors
    n_projections: int = 10,
    rp_dim: int = 50,
    k_mnn: int = 20,
    consensus_threshold: float = 0.4,
    mnn_strategy: str = "star",          # "star" or "pairwise"
    # Spatial graph
    k_spatial: int = 6,
    gate_spatial_by_expr: bool = True,
    expr_gate_beta: float = 1.0,
    # Anchor reweighting by spatial context
    reweight_anchors_by_spatial_context: bool = True,
    spatial_power: float = 1.0,
    # Embedding + solver
    n_comps: int = 30,
    svd_dim_for_ctx: int = 50,
    lambda_anchor: float = 5.0,
    lambda_spatial: float = 1.0,
    solver_tol: float = 1e-5,
    solver_maxiter: int = 200,
    # Misc
    n_jobs: int = 1,
    random_state: int = 0,
    key_added: str = "X_prime",
    store_graphs: bool = False,
    graph_key: str = "prime_graphs",
    copy: bool = False,
    verbose: bool = True,
) -> Optional[AnnData]:
    """
    PRIME: Projection-based Robust Integration with Mutual-NN and spatial Embedding.

    Integrate multiple Visium / spatial transcriptomics slices by solving:

        (I + lambda_anchor * L_a + lambda_spatial * L_s) Z = Z0

    where L_a is the graph Laplacian of an ERP-consensus MNN anchor graph
    (across batches) and L_s is the Laplacian of within-batch spatial kNN graphs.
    Z0 is the TruncatedSVD embedding of HVG-only log1p data.
    """

    if copy:
        adata = adata.copy()

    if batch_key not in adata.obs:
        raise ValueError(f"{batch_key} not in adata.obs")
    if spatial_key not in adata.obsm:
        raise ValueError(f"{spatial_key} not in adata.obsm")

    batch_labels = adata.obs[batch_key].values
    spatial_coords = np.asarray(adata.obsm[spatial_key])
    n = adata.n_obs
    nb = len(np.unique(batch_labels))

    if verbose:
        sc.logging.info(
            f"[PRIME] n={n:,}, batches={nb}, "
            f"n_hvg={n_hvg}, n_proj={n_projections}, rp_dim={rp_dim}, k_mnn={k_mnn}, "
            f"k_spatial={k_spatial}, n_comps={n_comps}"
        )

    # ---- 1) HVGs ----
    hvg_mask = _pick_hvgs(adata, batch_key=batch_key, n_hvg=n_hvg,
                          flavor=hvg_flavor, layer=layer)
    if hvg_mask.sum() < 50:
        raise RuntimeError(f"Too few HVGs selected: {int(hvg_mask.sum())}")

    # ---- 2) log1p-normalized HVG matrix (sparse) ----
    X_base = adata.layers[layer] if (layer is not None and layer in adata.layers) else adata.X
    X_log_hvg = _row_norm_log1p(X_base, target_sum=target_sum)[:, hvg_mask]

    # ---- 3) Base SVD embeddings ----
    Z_ctx = _svd_embedding(X_log_hvg,
                           n_comps=max(svd_dim_for_ctx, n_comps),
                           random_state=random_state)
    Z0 = Z_ctx[:, :n_comps].copy()
    Z_gate = Z_ctx[:, :svd_dim_for_ctx].copy()

    # ---- 4) Within-batch spatial graph ----
    Ws = _build_spatial_graph(
        spatial_coords, batch_labels, Z_gate,
        k_spatial=k_spatial,
        n_jobs=n_jobs,
        spatial_weight_mode="rbf",
        gate_by_expr=gate_spatial_by_expr,
        expr_gate_beta=expr_gate_beta,
    )

    # ---- 5) ERP consensus MNN anchor graph ----
    X_norm = normalize(X_log_hvg, axis=1)  # L2-normalize rows; keeps sparsity
    keys_all = []
    for t in range(n_projections):
        rp = GaussianRandomProjection(n_components=rp_dim,
                                      random_state=random_state + t)
        Xp = rp.fit_transform(X_norm)  # dense (n, rp_dim)

        r, c = _multibatch_mnn_edges(
            Xp, batch_labels,
            k=k_mnn, strategy=mnn_strategy, n_jobs=n_jobs,
        )
        if r.size > 0:
            keys_all.append(r.astype(np.int64) * n + c.astype(np.int64))

    if not keys_all:
        raise RuntimeError(
            "No MNN edges found across projections. "
            "Try increasing k_mnn, lowering consensus_threshold, "
            "or using mnn_strategy='pairwise'."
        )

    all_keys = np.concatenate(keys_all)
    uniq_keys, counts = np.unique(all_keys, return_counts=True)
    freq = counts.astype(np.float32) / float(n_projections)

    keep = freq >= float(consensus_threshold)
    if keep.sum() == 0:
        raise RuntimeError(
            "No consensus MNN edges survived consensus_threshold. "
            "Try lowering consensus_threshold or increasing n_projections."
        )

    rows = (uniq_keys[keep] // n).astype(np.int64)
    cols = (uniq_keys[keep] % n).astype(np.int64)
    w_expr = freq[keep].astype(np.float32)

    # Optional spatial-context reweighting of anchors
    if reweight_anchors_by_spatial_context and Ws.nnz > 0:
        S = _spatial_context_from_graph(Ws, Z_gate)
        w_ctx = _rbf_similarity(S, rows, cols)
        w_anchor = w_expr * (w_ctx ** float(spatial_power))
    else:
        w_anchor = w_expr

    Wa = coo_matrix((w_anchor, (rows, cols)),
                    shape=(n, n), dtype=np.float32).tocsr()
    Wa.sum_duplicates()
    Wa.setdiag(0.0)
    Wa.eliminate_zeros()
    Wa = (Wa + Wa.T) * 0.5
    Wa.eliminate_zeros()

    if verbose:
        sc.logging.info(
            f"[PRIME] anchor edges={Wa.nnz:,}, spatial edges={Ws.nnz:,}"
        )

    # ---- 6) Laplacian-regularized CG solve ----
    La = _laplacian(Wa)
    Ls = _laplacian(Ws)

    A = (identity(n, format="csr", dtype=np.float32)
         + (lambda_anchor * La).astype(np.float32)
         + (lambda_spatial * Ls).astype(np.float32))
    A.eliminate_zeros()

    Z, solver_stats = _cg_solve_matrix_rhs(
        A.astype(np.float64).tocsr(),
        Z0.astype(np.float32),
        tol=solver_tol,
        maxiter=solver_maxiter,
    )

    adata.obsm[key_added] = Z.astype(np.float32)

    if store_graphs:
        adata.uns[graph_key] = {
            "W_anchor": Wa,
            "W_spatial": Ws,
            "n_hvg": int(hvg_mask.sum()),
            "n_projections": int(n_projections),
            "rp_dim": int(rp_dim),
            "k_mnn": int(k_mnn),
            "consensus_threshold": float(consensus_threshold),
            "mnn_strategy": mnn_strategy,
            "lambda_anchor": float(lambda_anchor),
            "lambda_spatial": float(lambda_spatial),
            "reweight_anchors_by_spatial_context": bool(reweight_anchors_by_spatial_context),
            "spatial_power": float(spatial_power),
            "gate_spatial_by_expr": bool(gate_spatial_by_expr),
            "expr_gate_beta": float(expr_gate_beta),
            "solver": "cg",
            **solver_stats,
        }

    if verbose:
        sc.logging.info(
            f"[PRIME] stored adata.obsm['{key_added}'] "
            f"(cg converged dims: {solver_stats['cg_converged_dims']}/{n_comps})"
        )

    return adata if copy else None
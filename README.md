# PRIME

**P**rojection-based **R**obust **I**ntegration via **M**anifold **E**mbedding

[![Python](https://img.shields.io/badge/python-%E2%89%A53.9-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

PRIME is a Python package for **batch effect correction** in single-cell RNA-seq (scRNA-seq) and **multi-slice integration** in spatial transcriptomics. It combines:

- **Ensemble random projections** of the expression matrix
- **Consensus mutual nearest neighbor (MNN) graphs** built across many low-dimensional views
- **(Spatial only)** Laplacian-regularized embedding that jointly respects MNN anchors and within-slice spatial neighborhoods

Two main entry points cover both modalities:

| Function | Use case | Output |
|----------|----------|--------|
| `prime.ensemble_mnn_correct` | scRNA-seq batch correction | Corrected expression matrix (`n_cells × n_genes`) |
| `prime.prime_st` | Spatial transcriptomics integration | Integrated embedding in `adata.obsm["X_prime"]` |

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/XinchaoWu99/PRIME.git
cd PRIME
pip install -e .
```

To include optional plotting and evaluation extras:

```bash
pip install -e ".[all]"
```

**Requirements:** Python ≥ 3.9, NumPy, Pandas, SciPy, scikit-learn, AnnData, Scanpy.

---

## Quick start

### 1. scRNA-seq batch correction

```python
import scanpy as sc
import prime

# adata.X: cell × gene matrix (raw counts or normalized expression)
# adata.obs["batch"]: batch labels per cell
adata = sc.read_h5ad("your_data.h5ad")

# Run PRIME — returns the batch-corrected expression matrix
X_corrected = prime.ensemble_mnn_correct(
    adata,
    batch_key="batch",
    n_projections=10,        # number of random projections
    target_dim=50,           # dimension of each random projection
    k_neighbors=20,          # k for MNN search
    consensus_threshold=0.4, # fraction of projections an edge must appear in
    sigma=0.1,               # Gaussian smoothing bandwidth
    random_state=42,
)

# Store and use downstream
adata.layers["prime"] = X_corrected
adata.X = X_corrected

sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=["batch", "cell_type"])
```

### 2. Spatial transcriptomics integration

For Visium / Stereo-seq / MERFISH and similar datasets with multiple slices:

```python
import scanpy as sc
import prime

# adata.obs["slice"]: slice / batch ID
# adata.obsm["spatial"]: (n_cells, 2) array of spatial coordinates
adata = sc.read_h5ad("multi_slice_spatial.h5ad")

prime.prime_st(
    adata,
    batch_key="slice",
    spatial_key="spatial",
    n_hvg=3000,
    n_projections=10,
    k_mnn=20,
    k_spatial=6,
    lambda_anchor=5.0,     # weight of cross-slice MNN constraint
    lambda_spatial=1.0,    # weight of within-slice spatial smoothness
    n_comps=30,
    key_added="X_prime",
)

# Integrated embedding is now in adata.obsm["X_prime"]
sc.pp.neighbors(adata, use_rep="X_prime")
sc.tl.umap(adata)
sc.pl.umap(adata, color=["slice", "region"])
```

---

## API reference

### `prime.ensemble_mnn_correct`

```python
prime.ensemble_mnn_correct(
    adata,
    batch_key,
    projection_keys=None,
    n_projections=10,
    target_dim=50,
    k_neighbors=20,
    consensus_threshold=0.4,
    sigma=0.1,
    random_state=42,
    key_added=None,
    inplace=True,
    chunk_size=2000,
)
```

**Key parameters**

| Parameter | Description |
|-----------|-------------|
| `adata` | `AnnData` with expression matrix in `.X` and batch labels in `.obs[batch_key]`. |
| `batch_key` | Column name in `adata.obs` holding batch IDs. |
| `n_projections` | Number of random projections in the ensemble. More projections → more robust consensus, higher cost. |
| `target_dim` | Output dimension of each random projection. |
| `k_neighbors` | Number of nearest neighbors per cell when building MNN graphs. |
| `consensus_threshold` | An MNN edge is kept only if it appears in ≥ this fraction of projections (0 – 1). |
| `sigma` | Bandwidth of the Gaussian smoothing kernel applied after MNN correction. |
| `chunk_size` | Number of genes processed per chunk; lower → less memory. |

**Returns:** `np.ndarray` of shape `(n_cells, n_genes)` containing the batch-corrected expression matrix. Assign it back to `adata.X` or `adata.layers[...]`.

### `prime.prime_st`

```python
prime.prime_st(
    adata,
    *,
    batch_key,
    spatial_key="spatial",
    layer=None,
    n_hvg=3000,
    hvg_flavor="seurat_v3",
    n_projections=10,
    rp_dim=50,
    k_mnn=20,
    consensus_threshold=0.4,
    mnn_strategy="star",          # "star" or "pairwise"
    k_spatial=6,
    gate_spatial_by_expr=True,
    reweight_anchors_by_spatial_context=True,
    n_comps=30,
    lambda_anchor=5.0,
    lambda_spatial=1.0,
    solver_tol=1e-5,
    solver_maxiter=200,
    random_state=0,
    key_added="X_prime",
    store_graphs=False,
    copy=False,
    verbose=True,
)
```

**Key parameters**

| Parameter | Description |
|-----------|-------------|
| `adata` | `AnnData` with counts in `.X`, batch labels in `.obs[batch_key]`, spatial coordinates in `.obsm[spatial_key]`. |
| `n_hvg` | Number of highly variable genes used as the SVD input. |
| `k_mnn` | k for cross-slice MNN search. |
| `k_spatial` | k for within-slice spatial kNN graph. |
| `mnn_strategy` | `"star"` anchors all slices to a hub, `"pairwise"` builds all pairwise MNNs (slower, denser). |
| `lambda_anchor` | Weight on the cross-slice MNN Laplacian regularizer. |
| `lambda_spatial` | Weight on the within-slice spatial Laplacian regularizer. |
| `n_comps` | Output embedding dimension. |
| `key_added` | Slot in `adata.obsm` where the integrated embedding is stored. |
| `copy` | If `True`, return a corrected copy of `adata` instead of modifying in place. |

**Returns:** `None` (writes `adata.obsm[key_added]`) or a new `AnnData` if `copy=True`.

The integration solves

```
(I + λ_anchor · L_anchor + λ_spatial · L_spatial) · Z = Z₀
```

where `L_anchor` is the Laplacian of the consensus MNN anchor graph, `L_spatial` is the within-slice spatial Laplacian, and `Z₀` is the TruncatedSVD embedding of HVG log1p-normalized expression. The linear system is solved column-wise by conjugate gradient.

---

## Evaluation metrics

PRIME ships with two evaluation utilities useful for benchmarking integration quality.

### Cross-slice Layer Continuity (XLC)

For data with ordinal labels (e.g., cortical layers L1–L6/WM), measures whether neighborhood structure preserves the layer ordering:

```python
from prime.metrics import xlc_score

scores = xlc_score(
    adata,
    label_key="layer",
    embedding_keys=["X_prime", "X_pca", "X_harmony"],
    batch_key="slice",
    k_values=(15, 30, 50),
    cross_slice=True,
    n_perm=100,
)
print(scores)   # DataFrame with one row per embedding
```

### Isolated label preservation

Silhouette-based score for cell types that appear in only a few batches — a stress test for over-correction:

```python
from prime.metrics import compute_isolated_label_scores

iso_scores = compute_isolated_label_scores(
    adata,
    label_key="cell_type",
    embedding_keys=["X_prime", "X_pca", "X_harmony"],
    batch_key="batch",
)
print(iso_scores)
```

---

## Visualization

Render scIB-style benchmark result tables with colored cells and per-metric bar plots:

```python
from prime.plotting import plot_scib_results_table, save_scib_results_publication_pdf

fig, ax, tab = plot_scib_results_table(results_df)
save_scib_results_publication_pdf(fig, ax, "benchmark_table.pdf")
```

Requires the optional `plotting` extras: `pip install -e ".[plotting]"`.

---

## Module layout

```
prime/
├── core.py         # ensemble_mnn_correct (scRNA-seq)
├── spatial.py      # prime_st (spatial transcriptomics)
├── metrics/
│   ├── xlc.py            # ordinal_layer_continuity, xlc_score
│   └── isolated_label.py # compute_isolated_label_scores, ...
└── plotting/
    └── benchmark.py      # plot_scib_results_table, ...
```

---

## Citation

If you use PRIME in your research, please cite (manuscript in preparation):

> Wu, X. *et al.* **PRIME: Projection-based Robust Integration via Manifold Embedding for single-cell and spatial transcriptomics.** *In preparation* (2026).

A BibTeX entry will be added here once the manuscript is published.

---

## License

PRIME is released under the [MIT License](LICENSE).

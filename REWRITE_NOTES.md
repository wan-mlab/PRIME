# Step-5 rewrite: ERP-consensus-MNN helper + alignment with `core.py`

This documents the `prime/spatial.py` rewrite (the ad-hoc anchor-graph loop →
reusable helpers) and evaluates whether it faithfully embodies the **ensemble
random projection + MNN principle** implemented in `prime/core.py` and
`prime/gpu/ensemble.py`.

**Verdict: aligned, on both counts that matter.**

1. **Faithful** — the extracted helper reproduces the core algorithm
   *bit-for-bit*. Old vs new give identical anchor graphs (7 238 edges,
   `max|ΔW_anchor| = 0`) and an identical final embedding (`max|ΔZ| = 0`).
2. **Appropriately scoped** — random projection is applied exactly where its
   mathematical guarantee (distance preservation) is the property the operation
   needs (MNN search, optional expression-distance gating), and deliberately
   *not* where a different property is needed (variance-ranked denoising for
   `Z0`; exact physical neighborhoods for the spatial graph).

---

## 1. What "the `core.py` principle" actually is

`core.build_consensus_graph` and `gpu.ensemble._gpu_consensus_graph` are the
canonical statement of the algorithm. End to end it is four ideas:

### (a) L2 row-normalization — compare *profiles*, not library size

Each cell's expression vector `xᵢ ∈ ℝ^G` is rescaled to unit length:

```
x̂ᵢ = xᵢ / ‖xᵢ‖₂
```

`G` = number of genes, `‖·‖₂` = Euclidean (L2) norm. Why this matters: for two
unit vectors the squared Euclidean distance and the cosine of their angle `θ`
are tied by the identity

```
‖x̂ᵢ − x̂ⱼ‖² = 2(1 − cos θ).
```

So once rows are normalized, Euclidean nearest-neighbor search is equivalent to
*cosine* nearest-neighbor search — it ranks cells by the **direction** of their
expression profile and ignores total counts (sequencing depth / library size).
That is exactly what you want before matching cells across batches.

### (b) Gaussian random projection — the Johnson–Lindenstrauss lemma

Draw a random matrix `R ∈ ℝ^{G×d}` with i.i.d. entries `R_{gk} ~ N(0, 1/d)`
(scikit-learn's `GaussianRandomProjection` convention; the GPU path writes the
same thing as `standard_normal / √d`). Embed:

```
Y = X̂ · R      (Y ∈ ℝ^{n×d},  d = target_dim ≪ G)
```

For any difference vector `u = x̂ᵢ − x̂ⱼ`, this map is **distance-preserving in
expectation and with high probability**:

```
E[‖Ru‖²] = ‖u‖²,        (1−ε)‖u‖² ≤ ‖Ru‖² ≤ (1+ε)‖u‖²
```

The Johnson–Lindenstrauss (JL) lemma says the right-hand inequality holds
simultaneously for all `n` points once

```
d  ≳  ε⁻² · ln n
```

(`ε ∈ (0,1)` is the distortion you tolerate; the constant is small). The meaning
for us: **if distances are preserved up to (1±ε), then the order of nearest
neighbours is preserved too**, so MNNs found in the cheap `d`-dimensional `Y`
are (almost always) the MNNs of the full `G`-dimensional space — at cost
`O(n·d)` instead of `O(n·G)`. This is *the* reason random projection is the
right reduction for a distance-based method like MNN. (SVD/PCA would also work
but is more expensive and, more importantly, is solving a *different* problem —
see §4.)

### (c) Mutual nearest neighbours (MNN) — cross-batch anchors

Within a projection, for a pair of batches `A, B`, cell `i ∈ A` and `j ∈ B` form
an MNN edge iff each is in the other's `k`-nearest-neighbour list:

```
j ∈ kNN_{A→B}(i)   AND   i ∈ kNN_{B→A}(j).
```

MNN pairs are cells in *matching biological states* across batches (Haghverdi
et al., 2018); they are the "anchors" that tell the integrator which cells
should be pulled together.

### (d) Ensemble consensus — variance reduction / bagging

A single random projection is a *noisy* view: with probability tied to `ε`, some
true neighbours are distorted out of the `k`-list and some spurious ones in.
So the algorithm repeats over `T = n_projections` independent projections (seeds
`random_state + t`) and scores each edge by how often it appears:

```
freq(i,j) = (# projections in which (i,j) is an MNN) / T  ∈ [0, 1]
keep edges with  freq(i,j) ≥ consensus_threshold.
```

This is bagging: averaging over `T` independent estimators cuts the variance of
the edge indicator roughly like `1/T`, so only **stable** anchors — those that
survive many random views of the geometry — are kept. This is the "Robust" in
PRIME.

---

## 2. The rewrite, line-by-line against the principle

The new `_build_rp_consensus_mnn_graph` is the spatial-module instantiation of
exactly (a)–(d):

| Principle step | `core.py` / `gpu.ensemble` | `_build_rp_consensus_mnn_graph` (new) |
|---|---|---|
| (a) L2 normalize rows | `normalize(X, axis=1)` | `normalize(X_log_hvg, axis=1)` |
| (b) ensemble RP, seed `rs+t` | `GaussianRandomProjection(target_dim, rs+i)` | `_random_projection_embedding(X_norm, rp_dim, rs+t)` |
| (c) cross-batch MNN | `find_multibatch_mnn_graph` (all pairs) | `_multibatch_mnn_edges(..., strategy=)` |
| (d) consensus vote + threshold | `keys=r*n+c; unique; freq=count/T; keep≥thr` | **identical code** |
| symmetrize | `G.maximum(G.T)` | `(Wa + Wa.T) * 0.5` |

The consensus block (step d) is *the same code* already present in
`gpu/ensemble._consensus_edges_to_csr` — flat `r*n+c` key encoding,
`np.unique(return_counts=True)`, `freq = counts/T`, `keep = freq ≥ threshold`.
The two symmetrizations (`max` vs averaging) coincide here because edges are
inserted in both directions with equal weight, so the two directions are equal
and `max = mean = the value`.

**Test evidence (`test_rewrite.py`, 16/16):** with identical inputs and seed,
the refactored `prime_st` reproduces the pre-refactor implementation exactly —
`W_anchor` edge count identical (7 238), `max|ΔW_anchor| = 0`, `W_spatial`
identical, and final `X_prime` identical (`max|ΔZ| = 0`). The extraction changed
no behavior on the default path.

---

## 3. The reusable pieces that were added

* `_random_projection_embedding(X, n_components, random_state)` — one Gaussian
  projection, sparse-friendly, returns dense `float32`. Now the single, reusable
  realization of step (b), used by both the anchor graph and (optionally) the
  context embedding.
* `_build_rp_consensus_mnn_graph(...)` — steps (a)–(d) as a self-contained
  function returning a symmetric, self-loop-free CSR anchor graph. `prime_st`
  Step 5 is now one call to it, followed by the (unchanged) spatial-context
  reweighting applied to the returned edges.
* New switches: `context_method ∈ {svd, random_projection}` for the context
  embedding `Z_gate`; `base_embedding_method ∈ {svd, random_projection}` for the
  solver RHS `Z0` (experimental — see §4).
* `store_graphs` metadata now records `anchor_projection_method`,
  `context_method`, `base_embedding_method`, `random_state`, `rp_dim`,
  `n_projections`, `mnn_strategy`.

---

## 4. Why the scoping is the *correct* reading of the principle

The principle is "use RP **for the distance-based MNN step**," not "replace every
matrix factorization with RP." JL guarantees *distance preservation*; it says
nothing about ordering directions by variance or denoising. The rewrite respects
that distinction:

* **Anchor graph (always RP).** MNN is pure nearest-neighbour search ⇒ JL applies
  ⇒ RP is the right tool. ✔ Always random projection.
* **`Z_gate` context (optionally RP).** The expression-distance gate and the
  spatial-context reweighting are also distance computations, so a
  distance-preserving RP is a legitimate drop-in. Offered via `context_method`,
  SVD by default. ✔ Optional, and — verified by test — it only changes anchor
  *weights*, never the anchor *edge set*.
* **`Z0` base embedding (SVD by default; RP experimental).** `Z0` is the solver's
  right-hand side — the biological signal the system is regularized toward. Here
  you want a **low-rank, denoised, variance-ordered** representation, which is
  what SVD/PCA gives and what RP explicitly does **not** (RP preserves distances
  but keeps every noisy direction with equal expected weight). Replacing `Z0`
  changes the objective `(I + λ_aL_a + λ_sL_s)Z = Z0`, so it is gated behind an
  experimental flag. ✔ Default unchanged.
* **Spatial graph (never projected).** Built by kNN on the raw 2-D physical
  coordinates. JL offers nothing at `d = 2`, and spatial neighbourhoods should be
  *exact*, not approximate. Keeping expression-derived anchors and physical
  spatial structure as separate terms is exactly how spatial-integration methods
  (GraphST, PRECAST, STAligner) are built. ✔ Untouched.

---

## 5. Honest caveats / where it is *not* identical to `core.py`

These are deliberate, pre-existing spatial-module choices — flagged for full
intellectual honesty, none of them break the principle:

1. **Input space.** `core.py` projects the full normalized gene matrix; the
   spatial path projects the **HVG-subset, log1p** matrix `X_log_hvg`. JL applies
   to whatever space you feed it, so the *mechanism* is identical, but the
   *space whose distances are preserved* differs (HVG-log vs all-gene). This is a
   reasonable, common choice and is unchanged by the rewrite.
2. **MNN topology default.** `core.py` matches **all** batch pairs;
   `prime_st` defaults to `mnn_strategy="star"` (match every batch to the largest
   reference batch) for scalability. `"pairwise"` reproduces `core.py`'s all-pairs
   behavior and is exposed as a parameter.
3. **Symmetrization.** averaging `(W+Wᵀ)/2` vs GPU's `max(W,Wᵀ)` — numerically
   identical here (symmetric inserts), so not a real divergence.

---

## 6. How to verify

* One-off equivalence — run during the rewrite against a saved copy of the
  pre-refactor `spatial.py`: 16/16 checks, including the bit-identical
  old-vs-new comparison (`max|ΔW_anchor| = 0`, `max|ΔZ| = 0`). This needs the
  pre-refactor file, so it is a development-time check rather than a repo test.
* Permanent regression suite: `pytest tests/test_spatial_rewrite.py` → 11/11.
  Locks in: shape `(n_obs, n_comps)`; `W_anchor`/`W_spatial` symmetric and
  self-loop-free; `prime_st`'s anchor graph **equals** the standalone helper;
  `context_method` doesn't change anchor topology; both MNN strategies and both
  context methods run; metadata recorded; determinism for a fixed seed.

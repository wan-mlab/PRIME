import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed


LAYER_TO_RANK = {
    "Layer1": 1,
    "Layer2": 2,
    "Layer3": 3,
    "Layer4": 4,
    "Layer5": 5,
    "Layer6": 6,
    "WM": 7,
}


def ordinal_layer_continuity(
    embedding,
    labels,
    slices=None,
    k_values=(15, 30, 50),
    cross_slice=True,
    n_perm=100,
    label_to_rank=None,
    obs_only=False,
    random_state=0,
):
    """
    Compute ordinal layer-neighborhood continuity for an integrated embedding.

    Parameters
    ----------
    embedding : array-like, shape (n_spots, n_dims)
        Integrated latent embedding.
    labels : array-like, shape (n_spots,)
        Layer labels, e.g. Layer1, ..., Layer6, WM.
    slices : array-like, shape (n_spots,), optional
        Slice/sample IDs. Required for cross-slice evaluation.
    k_values : tuple
        Neighborhood sizes.
    cross_slice : bool
        If True, only neighbors from other slices are used.
    n_perm : int
        Number of within-slice label permutations for null correction.
    label_to_rank : dict
        Mapping from layer label to ordered rank.
    random_state : int
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Observed, null, corrected continuity score, and permutation p-value.
    """

    rng = np.random.default_rng(random_state)

    X = np.asarray(embedding, dtype=float)
    raw_labels = np.asarray(labels)

    if label_to_rank is None:
        label_to_rank = LAYER_TO_RANK

    ranks = np.array(
        [label_to_rank.get(str(x), np.nan) for x in raw_labels],
        dtype=float,
    )

    valid = np.isfinite(ranks) & np.all(np.isfinite(X), axis=1)

    X = X[valid]
    ranks = ranks[valid]

    if slices is None:
        slice_ids = np.array(["slice"] * len(ranks))
        cross_slice = False
    else:
        slice_ids = np.asarray(slices)[valid]

    X = StandardScaler().fit_transform(X)

    n = X.shape[0]
    max_k = max(k_values)

    if cross_slice and len(np.unique(slice_ids)) < 2:
        raise ValueError("cross_slice=True requires at least two slices.")

    # Exact nearest neighbors. For very large data, replace this with FAISS/Annoy.
    n_neighbors = n if cross_slice else min(n, max_k + 1)
    nn = NearestNeighbors(
        n_neighbors=n_neighbors, 
        metric="euclidean",
        n_jobs=-1,
        )
    nn.fit(X)
    _, idx = nn.kneighbors(X)

    def build_neighbors(k):
        neighbors = []
        for i in range(n):
            cand = idx[i]
            cand = cand[cand != i]

            if cross_slice:
                cand = cand[slice_ids[cand] != slice_ids[i]]

            neighbors.append(cand[:k])
        return neighbors

    def macro_score(rank_vec, neighbors):
        local_scores = np.full(n, np.nan)

        for i, js in enumerate(neighbors):
            if len(js) == 0:
                continue

            sim = 1.0 - np.abs(rank_vec[i] - rank_vec[js]) / 6.0
            local_scores[i] = np.mean(sim)

        layer_scores = []
        for layer_rank in sorted(np.unique(rank_vec)):
            mask = (rank_vec == layer_rank) & np.isfinite(local_scores)
            if np.any(mask):
                layer_scores.append(np.mean(local_scores[mask]))

        return float(np.mean(layer_scores))

    def permute_within_slice(rank_vec):
        out = rank_vec.copy()
        for s in np.unique(slice_ids):
            mask = slice_ids == s
            out[mask] = rng.permutation(out[mask])
        return out

    records = []

    for k in k_values:
        print(f"Evaluating k={k}...")
        neighbors = build_neighbors(k)

        obs = macro_score(ranks, neighbors)

        if obs_only:
            records.append(
                {
                    "k": k,
                    "observed_OLNC": obs,
                    "null_OLNC": np.nan,
                    "corrected_OLNC": np.nan,
                    "permutation_p_value": np.nan,
                    "cross_slice": cross_slice,
                    "mean_available_neighbors": np.mean([len(x) for x in neighbors]),
                }
            )
            continue

        # null_scores = []
        # for _ in range(n_perm):
        #     perm_ranks = permute_within_slice(ranks)
        #     null_scores.append(macro_score(perm_ranks, neighbors))

        unique_slices = np.unique(slice_ids)
        slice_masks = [slice_ids == s for s in unique_slices]

        def perm_score(seed):
            local_rng = np.random.default_rng(seed)
            perm_ranks = ranks.copy()

            for mask in slice_masks:
                perm_ranks[mask] = local_rng.permutation(perm_ranks[mask])

            return macro_score(perm_ranks, neighbors)

        seeds = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=n_perm,
            dtype=np.uint32,
        )

        null_scores = Parallel(n_jobs=-1, prefer="processes")(
            delayed(perm_score)(int(seed)) for seed in seeds
        )

        null_scores = np.asarray(null_scores)
        null_mean = float(np.mean(null_scores))

        corrected = (obs - null_mean) / (1.0 - null_mean)
        p_value = (1.0 + np.sum(null_scores >= obs)) / (n_perm + 1.0)

        mean_available_neighbors = np.mean([len(x) for x in neighbors])

        records.append(
            {
                "k": k,
                "observed_OLNC": obs,
                "null_OLNC": null_mean,
                "corrected_OLNC": corrected,
                "permutation_p_value": p_value,
                "cross_slice": cross_slice,
                "mean_available_neighbors": mean_available_neighbors,
            }
        )

    df = pd.DataFrame(records)
    if obs_only:
        return df["observed_OLNC"].mean()

    summary = {
        "k": "mean",
        "observed_OLNC": df["observed_OLNC"].mean(),
        "null_OLNC": df["null_OLNC"].mean(),
        "corrected_OLNC": df["corrected_OLNC"].mean(),
        "permutation_p_value": np.nan,
        "cross_slice": cross_slice,
        "mean_available_neighbors": df["mean_available_neighbors"].mean(),
    }

    # df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    return summary["corrected_OLNC"]


def xlc_score(
    adata,
    label_key,
    embedding_keys,
    batch_key=None,
    k_values=(15, 30, 50),
    cross_slice=True,
    n_perm=100,
    label_to_rank=None,
    obs_only=False,
    random_state=0,
):
    if label_key not in adata.obs.columns:
        raise ValueError(f"label_key={label_key!r} not found in adata.obs")
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key={batch_key!r} not found in adata.obs")

    labels = adata.obs[label_key].to_numpy().astype(str)
    batch = adata.obs[batch_key].to_numpy().astype(str) if batch_key else None

    results = {}
    for key in embedding_keys:
        if key not in adata.obsm:
            print(f"Embedding key {key!r} not found in adata.obsm. Skipping.")
            results[key] = float("nan")
            continue

        X = adata.obsm[key]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        print(f"[{key}] Computing XLC score...")

        score = ordinal_layer_continuity(
            embedding=X,
            labels=labels,
            slices=batch,
            k_values=k_values,
            cross_slice=cross_slice,
            n_perm=n_perm,
            label_to_rank=label_to_rank,
            obs_only=obs_only,
            random_state=random_state,
        )
        results[key] = score

        print(f"  XLC score = {score:.4f}")

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["XLC_score"],
    )
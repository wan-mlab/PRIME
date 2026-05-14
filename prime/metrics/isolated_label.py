import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import logging
from sklearn.metrics import silhouette_samples

RESULTS_METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE_VALUE = "Aggregate score"
BIO_CONSERVATION_COL = "Bio conservation"
BATCH_CORRECTION_COL = "Batch correction"
TOTAL_SCORE_COL = "Total"
ISOLATED_LABELS_COL = "Isolated labels"

logger = logging.getLogger(__name__)

def _get_isolated_labels(
    labels: np.ndarray,
    batch: np.ndarray | None,
    iso_threshold: int | None,
) -> np.ndarray:
    if batch is None:
        return np.unique(labels)

    tmp = pd.DataFrame({"label": labels, "batch": batch}).drop_duplicates()
    batch_per_label = tmp.groupby("label")["batch"].count()

    if iso_threshold is None:
        iso_threshold = int(batch_per_label.min())

    logger.info("Isolated label threshold: <= %s batch(es) per label", iso_threshold)
    isolated = batch_per_label[batch_per_label <= iso_threshold].index.to_numpy()

    if isolated.size == 0:
        logger.warning("No isolated labels found with threshold=%s", iso_threshold)

    return isolated


def isolated_label_score_single(
    X: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray | None = None,
    rescale: bool = True,
    iso_threshold: int | None = None,
) -> float:
    isolated = _get_isolated_labels(labels, batch, iso_threshold)
    if isolated.size == 0:
        logger.warning("No isolated labels found. Returning NaN.")
        return float("nan")

    try:
        sil_all = silhouette_samples(X, labels, metric="euclidean")
    except ValueError as exc:
        logger.warning("Failed to compute isolated label score: %s", exc)
        return float("nan")

    if rescale:
        sil_all = (sil_all + 1.0) / 2.0

    per_label_scores = []
    for label in isolated:
        mask = labels == label
        per_label_scores.append(float(np.mean(sil_all[mask])))

    return float(np.mean(per_label_scores))


def compute_isolated_label_scores(
    adata: ad.AnnData,
    label_key: str,
    embedding_keys: list[str],
    batch_key: str | None = None,
    rescale: bool = True,
    iso_threshold: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    if label_key not in adata.obs.columns:
        raise ValueError(f"label_key={label_key!r} not found in adata.obs")
    if batch_key is not None and batch_key not in adata.obs.columns:
        raise ValueError(f"batch_key={batch_key!r} not found in adata.obs")

    labels = adata.obs[label_key].to_numpy().astype(str)
    batch = adata.obs[batch_key].to_numpy().astype(str) if batch_key else None

    results = {}
    for key in embedding_keys:
        if key not in adata.obsm:
            logger.warning("Embedding key %r not found in adata.obsm", key)
            results[key] = float("nan")
            continue

        X = adata.obsm[key]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)

        if verbose:
            print(f"[{key}] Computing isolated label score...")

        score = isolated_label_score_single(
            X=X,
            labels=labels,
            batch=batch,
            rescale=rescale,
            iso_threshold=iso_threshold,
        )
        results[key] = score

        if verbose:
            print(f"  score = {score:.4f}")

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=[ISOLATED_LABELS_COL],
    )


def _merge_isolated_label_scores(
    results_df: pd.DataFrame,
    isolated_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = results_df.copy()
    merged[ISOLATED_LABELS_COL] = np.nan

    method_rows = merged.index[merged.index != RESULTS_METRIC_TYPE_ROW]
    merged.loc[method_rows, ISOLATED_LABELS_COL] = (
        isolated_df.reindex(method_rows)[ISOLATED_LABELS_COL].to_numpy()
    )

    if RESULTS_METRIC_TYPE_ROW in merged.index:
        merged.loc[RESULTS_METRIC_TYPE_ROW, ISOLATED_LABELS_COL] = BIO_CONSERVATION_COL
        cols_without_new = [col for col in merged.columns if col != ISOLATED_LABELS_COL]
        metric_type_row = merged.loc[RESULTS_METRIC_TYPE_ROW, cols_without_new]
        aggregate_cols = metric_type_row.index[
            metric_type_row == AGGREGATE_SCORE_VALUE
        ].tolist()
        insert_at = (
            cols_without_new.index(aggregate_cols[0])
            if aggregate_cols
            else len(cols_without_new)
        )
        reordered_cols = (
            cols_without_new[:insert_at]
            + [ISOLATED_LABELS_COL]
            + cols_without_new[insert_at:]
        )
        merged = merged.loc[:, reordered_cols]

    return merged


def _recompute_aggregate_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    if RESULTS_METRIC_TYPE_ROW not in results_df.index:
        raise ValueError("results_df must include the Metric Type row.")

    updated = results_df.copy()
    metric_type_row = updated.loc[RESULTS_METRIC_TYPE_ROW]
    method_rows = updated.index[updated.index != RESULTS_METRIC_TYPE_ROW]

    bio_metric_cols = metric_type_row.index[
        metric_type_row == BIO_CONSERVATION_COL
    ].tolist()
    bio_metrics = updated.loc[method_rows, bio_metric_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    batch_scores = pd.to_numeric(
        updated.loc[method_rows, BATCH_CORRECTION_COL],
        errors="coerce",
    )

    updated.loc[method_rows, BIO_CONSERVATION_COL] = bio_metrics.mean(axis=1)
    updated.loc[method_rows, TOTAL_SCORE_COL] = (
        0.6 * pd.to_numeric(updated.loc[method_rows, BIO_CONSERVATION_COL], errors="coerce")
        + 0.4 * batch_scores
    )

    updated.loc[RESULTS_METRIC_TYPE_ROW, BATCH_CORRECTION_COL] = AGGREGATE_SCORE_VALUE
    updated.loc[RESULTS_METRIC_TYPE_ROW, BIO_CONSERVATION_COL] = AGGREGATE_SCORE_VALUE
    updated.loc[RESULTS_METRIC_TYPE_ROW, TOTAL_SCORE_COL] = AGGREGATE_SCORE_VALUE
    return updated

def add_isolated_label_scores_to_results(
        results_df: pd.DataFrame,
        isolated_df: pd.DataFrame,
) -> pd.DataFrame:
    results_df = _merge_isolated_label_scores(results_df, isolated_df)
    results_df = _recompute_aggregate_scores(results_df)
    return results_df


def re_compute_aggregate_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    return _recompute_aggregate_scores(results_df)

    
## usage example:
# isolated_results_df = compute_isolated_label_scores(
#     adata,
#     label_key="cell_type",
#     embedding_keys=benchmark_keys,
#     batch_key="batch",
#     rescale=True,
#     iso_threshold=None,
#     verbose=True,
# )
# results_df = _merge_isolated_label_scores(results_df, isolated_results_df)
# results_df = _recompute_aggregate_scores(results_df)

# revised_res_df_file = f"{data_path}/benchmark_results_with_isolated_labels.csv"
# results_df.to_csv(revised_res_df_file, index=True)

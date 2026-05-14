from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from matplotlib.text import Text
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar

METRIC_TYPE_ROW = "Metric Type"
AGGREGATE_SCORE = "Aggregate score"
DEFAULT_SORT_PRIORITY = ("Total", "Batch correction", "Bio conservation")

# Default group ordering for both fine-grained metric columns and aggregate
# score columns. Groups not listed here are appended after these, in the
# order they first appear in the input DataFrame.
GROUP_ORDER: tuple[str, ...] = (
    "Bio conservation",
    "Batch correction",
    "Spatial smoothness",
    "Total",
)

OUTPUT_DIR = Path("/mnt/nrdstor/wanlab/xinchaowu/SHARP_data/immune")
PDF_FILENAME = "benchmark_results_pub.pdf"
FIGSIZE = (7.0, 4.2)
PDF_FONT_FAMILY = "Arial"
PDF_FONTTYPE = 42
PS_FONTTYPE = 42
BBOX_INCHES = "tight"
PAD_INCHES = 0.02
CELL_CMAP = mpl.cm.PRGn
SCORE_CMAP = mpl.cm.YlGnBu
SORT_COL = "Total"


def _validate_results_frame(df: pd.DataFrame) -> None:
    if METRIC_TYPE_ROW not in df.index:
        raise ValueError(
            "Input DataFrame must include a 'Metric Type' row, as returned by "
            "Benchmarker.get_results()."
        )


def _resolve_score_columns(df: pd.DataFrame) -> list[str]:
    metric_type_row = df.loc[METRIC_TYPE_ROW]
    score_cols = metric_type_row.index[metric_type_row == AGGREGATE_SCORE].tolist()
    if not score_cols:
        raise ValueError(
            "Input DataFrame must include at least one aggregate score column where "
            "df.loc['Metric Type', col] == 'Aggregate score'."
        )
    return score_cols


def _resolve_sort_col(score_cols: list[str], sort_col: str | None) -> str:
    if sort_col is not None:
        if sort_col not in score_cols:
            raise ValueError(
                f"sort_col {sort_col!r} must be one of the aggregate score columns: "
                f"{score_cols!r}."
            )
        return sort_col

    for candidate in DEFAULT_SORT_PRIORITY:
        if candidate in score_cols:
            return candidate
    return score_cols[0]


def _reorder_columns_by_group(
    df: pd.DataFrame,
    group_order: Sequence[str],
) -> pd.DataFrame:
    """Reorder columns so that all fine-grained metric columns come first
    (ordered by their Metric Type group), followed by all aggregate score
    columns (also ordered by group).

    For fine-grained columns (Metric Type != "Aggregate score"), the
    grouping key is the Metric Type value. For aggregate score columns
    (Metric Type == "Aggregate score"), the grouping key is the column
    name itself.

    Final column order:
        [fine-grained cols ordered by group] + [aggregate cols ordered by group]

    Within each (is_aggregate, group) bucket, original column order is
    preserved (stable sort). Groups in ``group_order`` come first in the
    given order; groups not listed are appended in first-appearance order.
    """
    metric_type_row = df.loc[METRIC_TYPE_ROW]

    def _group_key(col: str) -> str:
        mt = metric_type_row[col]
        return col if mt == AGGREGATE_SCORE else mt

    def _is_aggregate(col: str) -> int:
        # 0 for fine-grained (comes first), 1 for aggregate (comes last).
        return 1 if metric_type_row[col] == AGGREGATE_SCORE else 0

    # Discover all groups in first-appearance order for deterministic
    # fallback positioning of unlisted groups.
    seen: list[str] = []
    for col in df.columns:
        key = _group_key(col)
        if key not in seen:
            seen.append(key)

    listed_present = [g for g in group_order if g in seen]
    unlisted = [g for g in seen if g not in listed_present]
    final_group_order = listed_present + unlisted
    group_rank = {g: i for i, g in enumerate(final_group_order)}

    # Sort by (is_aggregate, group_rank). Fine-grained (is_aggregate=0)
    # columns are placed before all aggregate columns. Within each of those
    # two halves, columns are arranged by group order. Stable sort keeps
    # the original within-group ordering of fine-grained sub-metrics.
    sorted_cols = sorted(
        df.columns,
        key=lambda c: (_is_aggregate(c), group_rank[_group_key(c)]),
    )
    return df.loc[:, sorted_cols]

def _build_cell_cmap(
    plot_df: pd.DataFrame,
    col: str,
    cell_cmap: Colormap,
    cell_num_stds: float,
):
    return normed_cmap(plot_df[col], cmap=cell_cmap, num_stds=cell_num_stds)


def _default_figsize(df: pd.DataFrame, num_methods: int) -> tuple[float, float]:
    return (len(df.columns) * 1.25, 3 + 0.3 * num_methods)


def _set_figure_text_font_family(fig: Figure, font_family: str) -> None:
    for text_artist in fig.findobj(match=Text):
        text_artist.set_fontfamily(font_family)


def plot_scib_results_table(
    df: pd.DataFrame,
    cell_cmap: Colormap = mpl.cm.PRGn,
    score_cmap: Colormap = mpl.cm.YlGnBu,
    cell_num_stds: float = 2.5,
    sort_col: str | None = None,
    group_order: Sequence[str] | None = None,
    show: bool = True,
    figsize: tuple[float, float] | None = None,
    svg_text_as_text: bool = True,
) -> tuple[Figure, Axes, Table]:
    """Reproduce scib-metrics' benchmark results table without using its plotting API.

    Columns are automatically reordered by group (Metric Type) so that, by
    default, "Bio conservation" comes first, then "Batch correction", then
    "Spatial smoothness", then "Total". Aggregate score bars are placed
    immediately after the fine-grained columns of the same group.

    Parameters
    ----------
    df
        Results table with the same structure as ``Benchmarker.get_results()``.
        Rows are methods plus one ``"Metric Type"`` row.
    group_order
        Ordering of metric groups. Both fine-grained columns and aggregate
        score columns are arranged by this. Defaults to ``GROUP_ORDER``.
        Groups not listed here are appended in first-appearance order.
    ... (other parameters unchanged)
    """

    _validate_results_frame(df)

    # Reorder columns by group BEFORE any downstream processing so that
    # column definitions, sorting, and rendering all see the desired layout.
    resolved_group_order = tuple(group_order) if group_order is not None else GROUP_ORDER
    results_df = _reorder_columns_by_group(df.copy(), resolved_group_order)

    score_cols = _resolve_score_columns(results_df)
    resolved_sort_col = _resolve_sort_col(score_cols, sort_col)

    plot_df = results_df.drop(index=METRIC_TYPE_ROW)
    plot_df = plot_df.sort_values(by=resolved_sort_col, ascending=False).astype(np.float64)
    plot_df["Method"] = plot_df.index

    metric_type_row = results_df.loc[METRIC_TYPE_ROW]
    other_cols = metric_type_row.index[metric_type_row != AGGREGATE_SCORE].tolist()
    num_methods = plot_df.shape[0]

    column_definitions = [
        ColumnDefinition(
            "Method",
            width=1.5,
            textprops={"ha": "left", "weight": "bold"},
        )
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            title=col.replace(" ", "\n", 1),
            width=1,
            textprops={
                "ha": "center",
                "bbox": {"boxstyle": "circle", "pad": 0.25},
            },
            cmap=_build_cell_cmap(plot_df, col, cell_cmap=cell_cmap, cell_num_stds=cell_num_stds),
            group=metric_type_row[col],
            formatter="{:.2f}",
        )
        for col in other_cols
    ]
    column_definitions += [
        ColumnDefinition(
            col,
            width=1,
            title=col.replace(" ", "\n", 1),
            plot_fn=bar,
            plot_kw={
                "cmap": score_cmap,
                "plot_bg_bar": False,
                "annotate": True,
                "height": 0.9,
                "formatter": "{:.2f}",
            },
            group=metric_type_row[col],
            border="left" if i == 0 else None,
        )
        for i, col in enumerate(score_cols)
    ]

    rc_context = mpl.rc_context(
        {"svg.fonttype": "none"}) if svg_text_as_text else nullcontext()
    with rc_context:
        fig, ax = plt.subplots(figsize=figsize or _default_figsize(results_df, num_methods))
        table = Table(
            plot_df,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=column_definitions,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": 10, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col="Method",
        ).autoset_fontcolors(colnames=plot_df.columns)

    if show:
        plt.show()

    return fig, ax, table



def save_scib_results_publication_pdf(
    fig: Figure,
    ax: Axes,
    save_path: str | Path,
    *,
    dpi: int = 300,
    font_family: str = PDF_FONT_FAMILY,
    pdf_fonttype: int = PDF_FONTTYPE,
    ps_fonttype: int = PS_FONTTYPE,
    bbox_inches: str = BBOX_INCHES,
    pad_inches: float = PAD_INCHES,
) -> Path:
    """Save a figure as a publication-quality PDF.

    Applies the right rcParams in a *local* context so that:
    - PDF/PS text remains editable (TrueType embedding, fonttype 42)
    - SVG text stays as <text> elements rather than being converted to paths
    - The given font family is applied uniformly across all text artists

    The rcParams are scoped to this call only; they do not leak into your
    global matplotlib state.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Update existing Text artists to the desired font family.
    # rcParams['font.family'] alone does NOT retroactively change Text
    # objects whose fontfamily was already set during figure construction,
    # so we walk the figure and set it explicitly.
    _set_figure_text_font_family(fig, font_family)

    with mpl.rc_context({
        "pdf.fonttype": pdf_fonttype,
        "ps.fonttype": ps_fonttype,
        "svg.fonttype": "none",
        "font.family": font_family,
    }):
        fig.savefig(
            save_path,
            facecolor=ax.get_facecolor(),
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )

    return save_path
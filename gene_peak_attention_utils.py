from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.mixture import GaussianMixture


ArrayLike = Union[np.ndarray, Sequence[float]]


@dataclass(frozen=True)
class GMMThresholdResult:
    threshold: float
    means: np.ndarray
    variances: np.ndarray
    weights: np.ndarray
    intersections: np.ndarray
    model: GaussianMixture


def load_attention_matrix(npz_path: Union[str, Path]) -> sp.spmatrix:
    """Load a sparse attention matrix from an ``.npz`` file."""
    return sp.load_npz(str(npz_path))


def _to_numpy_str(values: Iterable[str]) -> np.ndarray:
    return np.asarray(list(values), dtype=object)


def extract_peak_gene_connections(
    peak_gene_attention: sp.spmatrix,
    peaks: Iterable[str],
    genes: Iterable[str],
    *,
    index_layout: str = "peak_first",
    min_attention: float = 0.0,
    drop_self_loops: bool = True,
) -> pd.DataFrame:
    """
    Reproduce tutorial extraction of peak->gene edges from a square attention matrix.

    Parameters
    ----------
    peak_gene_attention
        Sparse square matrix output by MultiGATE.
    peaks
        Peak feature names in the same order as used by the model.
    genes
        Gene feature names in the same order as used by the model.
    index_layout
        ``"peak_first"`` (tutorial default) means rows/cols are [peaks, genes].
        ``"gene_first"`` means rows/cols are [genes, peaks].
    """
    if index_layout not in {"peak_first", "gene_first"}:
        raise ValueError("index_layout must be one of {'peak_first', 'gene_first'}.")

    peaks_np = _to_numpy_str(peaks)
    genes_np = _to_numpy_str(genes)
    n_peaks = peaks_np.size
    n_genes = genes_np.size

    coo = peak_gene_attention.tocoo(copy=False)
    row = coo.row
    col = coo.col
    attn = coo.data

    if drop_self_loops:
        mask = row != col
        row = row[mask]
        col = col[mask]
        attn = attn[mask]

    if index_layout == "peak_first":
        edge_mask = (row >= 0) & (row < n_peaks) & (col >= n_peaks) & (col < (n_peaks + n_genes))
        peak_idx = row[edge_mask]
        gene_idx = col[edge_mask] - n_peaks
    else:
        edge_mask = (row >= n_genes) & (row < (n_genes + n_peaks)) & (col >= 0) & (col < n_genes)
        peak_idx = row[edge_mask] - n_genes
        gene_idx = col[edge_mask]

    attn = attn[edge_mask]
    if min_attention > 0:
        keep = attn >= float(min_attention)
        peak_idx = peak_idx[keep]
        gene_idx = gene_idx[keep]
        attn = attn[keep]

    df = pd.DataFrame(
        {
            "Gene": genes_np[gene_idx],
            "Peak": peaks_np[peak_idx],
            "Attention": attn,
            "gene_idx": gene_idx.astype(int),
            "peak_idx": peak_idx.astype(int),
        }
    ).sort_values("Attention", ascending=False, ignore_index=True)
    return df


def extract_gene_name(gene_value: str) -> str:
    """Match tutorial behavior: keep text after the final underscore when present."""
    if pd.isna(gene_value):
        return np.nan
    gene_str = str(gene_value)
    return gene_str.split("_")[-1] if "_" in gene_str else gene_str


def parse_peak(peak_value: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Parse ``chr:start-end`` or ``chr-start-end`` peak IDs into coordinates."""
    import re
    if pd.isna(peak_value):
        return None, None, None
    try:
        parts = re.split(r"[:-]", str(peak_value))
        if len(parts) == 3:
            chrom, start, end = parts
            return chrom, int(start), int(end)
        return None, None, None
    except Exception:
        return None, None, None


def add_gene_and_peak_columns(
    peak_gene_df: pd.DataFrame,
    *,
    gene_col: str = "Gene",
    peak_col: str = "Peak",
) -> pd.DataFrame:
    """Add tutorial-style ``gene_name`` and peak coordinate columns."""
    out = peak_gene_df.copy()
    out["gene_name"] = out[gene_col].map(extract_gene_name)
    parsed = out[peak_col].map(parse_peak)
    out["peak_chr"] = parsed.map(lambda x: x[0])
    out["peak_start"] = parsed.map(lambda x: x[1])
    out["peak_end"] = parsed.map(lambda x: x[2])
    return out


def parse_gtf_file(
    gtf_file: Union[str, Path],
    *,
    feature: str = "gene",
) -> pd.DataFrame:
    """
    Parse GTF and return columns used in the attention tutorial.

    Returned columns: ``gene_name``, ``gene_id``, ``chr``, ``start``, ``end``,
    ``strand``, ``tss``.
    """
    gtf_columns = [
        "chr",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attributes",
    ]
    gtf = pd.read_csv(
        str(gtf_file),
        sep="\t",
        comment="#",
        header=None,
        names=gtf_columns,
        low_memory=False,
    )

    if feature is not None:
        gtf = gtf.loc[gtf["feature"] == feature].copy()

    attrs = gtf["attributes"].astype(str)
    gtf["gene_name"] = attrs.str.extract(r'gene_name "([^"]+)"', expand=False)
    gtf["gene_id"] = attrs.str.extract(r'gene_id "([^"]+)"', expand=False)
    gtf["gene_name"] = gtf["gene_name"].fillna(gtf["gene_id"])

    gtf["start"] = pd.to_numeric(gtf["start"], errors="coerce")
    gtf["end"] = pd.to_numeric(gtf["end"], errors="coerce")
    gtf["tss"] = np.where(gtf["strand"] == "+", gtf["start"], gtf["end"])

    out_cols = ["gene_name", "gene_id", "chr", "start", "end", "strand", "tss"]
    return gtf[out_cols].dropna(subset=["gene_name", "chr", "tss"]).copy()


def merge_with_gene_annotations(
    peak_gene_df: pd.DataFrame,
    gtf_df: pd.DataFrame,
    *,
    gene_name_col: str = "gene_name",
) -> pd.DataFrame:
    """Left-join extracted peak-gene pairs with parsed GTF coordinates."""
    return peak_gene_df.merge(
        gtf_df[["gene_name", "gene_id", "chr", "start", "end", "strand", "tss"]],
        left_on=gene_name_col,
        right_on="gene_name",
        how="left",
    )


def compute_gene_peak_distance(
    merged_df: pd.DataFrame,
    *,
    gene_chr_col: str = "chr",
    peak_chr_col: str = "peak_chr",
    peak_start_col: str = "peak_start",
    peak_end_col: str = "peak_end",
    tss_col: str = "tss",
    out_col: str = "distance",
) -> pd.DataFrame:
    """Compute tutorial distance-to-TSS (0 if peak overlaps TSS)."""
    out = merged_df.copy()
    same_chr = out[gene_chr_col].astype(str) == out[peak_chr_col].astype(str)
    peak_start = pd.to_numeric(out[peak_start_col], errors="coerce")
    peak_end = pd.to_numeric(out[peak_end_col], errors="coerce")
    tss = pd.to_numeric(out[tss_col], errors="coerce")

    distance = np.where(
        ~same_chr | tss.isna() | peak_start.isna() | peak_end.isna(),
        np.nan,
        np.where(
            peak_end < tss,
            tss - peak_end,
            np.where(peak_start > tss, peak_start - tss, 0),
        ),
    )
    out[out_col] = distance
    return out


def fit_attention_gmm(
    attention_values: ArrayLike,
    *,
    n_components: int = 2,
    random_state: int = 0,
) -> GaussianMixture:
    arr = np.asarray(attention_values, dtype=float).reshape(-1, 1)
    arr = arr[np.isfinite(arr).ravel()]
    if arr.size == 0:
        raise ValueError("No finite attention values were provided.")
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(arr.reshape(-1, 1))
    return gmm


def _gmm_intersections_1d(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Solve weighted Gaussian density intersections for two 1D components."""
    if means.size != 2:
        raise ValueError("Intersection solver currently supports exactly 2 components.")

    m1, m2 = means.astype(float)
    v1, v2 = variances.astype(float)
    w1, w2 = weights.astype(float)

    # log(w1/sqrt(v1)) - (x-m1)^2/(2v1) = log(w2/sqrt(v2)) - (x-m2)^2/(2v2)
    a = 1.0 / (2.0 * v2) - 1.0 / (2.0 * v1)
    b = m1 / v1 - m2 / v2
    c = (m2 ** 2) / (2.0 * v2) - (m1 ** 2) / (2.0 * v1) + np.log((w1 * np.sqrt(v2)) / (w2 * np.sqrt(v1)))

    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0):
            return np.array([], dtype=float)
        return np.array([-c / b], dtype=float)

    roots = np.roots([a, b, c])
    roots = roots[np.isreal(roots)].real
    return np.sort(roots)


def get_gmm_attention_threshold(
    attention_values: ArrayLike,
    *,
    random_state: int = 0,
) -> GMMThresholdResult:
    """
    Fit 2-component GMM and compute intersection threshold.

    Fallback matches tutorial behavior: if no valid in-range intersection is found,
    use the smaller component mean.
    """
    values = np.asarray(attention_values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite attention values were provided.")

    gmm = fit_attention_gmm(values, n_components=2, random_state=random_state)
    means = gmm.means_.ravel()
    variances = gmm.covariances_.ravel()
    weights = gmm.weights_.ravel()
    intersections = _gmm_intersections_1d(means, variances, weights)

    vmin = float(np.min(values))
    vmax = float(np.max(values))
    between = intersections[(intersections > vmin) & (intersections < vmax)]

    if between.size > 0:
        lo, hi = np.sort(means)
        between_means = between[(between >= lo) & (between <= hi)]
        threshold = float(between_means[0] if between_means.size > 0 else between[0])
    else:
        threshold = float(np.min(means))

    return GMMThresholdResult(
        threshold=threshold,
        means=means,
        variances=variances,
        weights=weights,
        intersections=intersections,
        model=gmm,
    )


def filter_by_attention_threshold(
    merged_df: pd.DataFrame,
    threshold: float,
    *,
    attention_col: str = "Attention",
) -> pd.DataFrame:
    """Keep rows with attention strictly above the selected threshold."""
    return merged_df.loc[pd.to_numeric(merged_df[attention_col], errors="coerce") > float(threshold)].copy()


def assign_regulatory_region(
    df: pd.DataFrame,
    *,
    distance_col: str = "distance",
    promoter_window_bp: int = 2000,
    out_col: str = "regulatory_region",
) -> pd.DataFrame:
    """
    Label each pair as promoter/distal using distance to TSS.

    - ``Promoter``: distance <= ``promoter_window_bp``
    - ``Distal``: distance > ``promoter_window_bp``
    - ``Unknown``: missing distance
    """
    out = df.copy()
    d = pd.to_numeric(out[distance_col], errors="coerce")
    out[out_col] = np.where(d.isna(), "Unknown", np.where(d <= promoter_window_bp, "Promoter", "Distal"))
    return out


def plot_attention_distribution(
    attention_values: ArrayLike,
    *,
    threshold: Optional[float] = None,
    gmm: Optional[GaussianMixture] = None,
    bins: int = 100,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot attention histogram and optionally overlay GMM density + threshold."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    values = np.asarray(attention_values, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, density=True, alpha=0.6, color="steelblue", label="Attention")

    if gmm is not None:
        x = np.linspace(values.min(), values.max(), 1000)
        pdf = np.zeros_like(x)
        means = gmm.means_.ravel()
        variances = gmm.covariances_.ravel()
        weights = gmm.weights_.ravel()
        for mean, var, weight in zip(means, variances, weights):
            comp_pdf = weight * norm.pdf(x, mean, np.sqrt(var))
            pdf += comp_pdf
            ax.plot(x, comp_pdf, linestyle="--", linewidth=1.5)
        ax.plot(x, pdf, color="black", linewidth=2.0, label="GMM")

    if threshold is not None:
        ax.axvline(float(threshold), color="red", linestyle="--", linewidth=2, label=f"Threshold={threshold:.4f}")

    ax.set_xlabel("Attention")
    ax.set_ylabel("Density")
    ax.set_title("Attention Distribution")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_distance_distribution(
    distances: ArrayLike,
    *,
    bins: int = 100,
    figsize: Tuple[int, int] = (8, 5),
):
    """Plot histogram of gene-peak genomic distances."""
    import matplotlib.pyplot as plt

    values = np.asarray(distances, dtype=float)
    values = values[np.isfinite(values)]

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(values, bins=bins, color="slateblue", alpha=0.8)
    ax.set_xlabel("Distance to TSS (bp)")
    ax.set_ylabel("Count")
    ax.set_title("Gene-Peak Distance Distribution")
    fig.tight_layout()
    return fig, ax


def save_attention_outputs(
    merged_df: pd.DataFrame,
    merged_df_threshold: pd.DataFrame,
    threshold: float,
    *,
    output_dir: Union[str, Path],
    merged_name: str = "merged_df.csv",
    thresholded_name: str = "merged_df_threshold.csv",
    threshold_name: str = "threshold.txt",
) -> None:
    """Save tutorial-style result tables and threshold value."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_df.to_csv(out_dir / merged_name, index=False)
    merged_df_threshold.to_csv(out_dir / thresholded_name, index=False)
    with open(out_dir / threshold_name, "w", encoding="utf-8") as f:
        f.write(f"{float(threshold)}\n")


def save_gene_peak_links_bedpe(
    gp_link_df: pd.DataFrame,
    output_path: Union[str, Path],
    *,
    score_col: str = "Attention",
) -> Path:
    """
    Save selected gene-peak links as BEDPE for CoolBox Arcs.

    The peak interval is BEDPE end 1. The gene endpoint is a 1 bp interval
    anchored at the GTF-style 1-based TSS, converted to BED-style
    ``[tss - 1, tss]`` coordinates.
    """
    required_cols = {
        "Gene",
        "Peak",
        "peak_chr",
        "peak_start",
        "peak_end",
        "chr",
        "tss",
        "strand",
        score_col,
    }
    missing = sorted(required_cols.difference(gp_link_df.columns))
    if missing:
        raise ValueError(f"gp_link_df is missing required columns: {missing}")
    if gp_link_df.empty:
        raise ValueError("gp_link_df is empty; no BEDPE links to save.")

    out = gp_link_df.copy()
    numeric_cols = ["peak_start", "peak_end", "tss", score_col]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    invalid = {
        col: int(out[col].isna().sum())
        for col in ["peak_chr", "peak_start", "peak_end", "chr", "tss", "strand", score_col]
        if out[col].isna().any()
    }
    for col in ["peak_chr", "chr", "strand"]:
        empty = out[col].astype(str).str.strip().eq("")
        if empty.any():
            invalid[col] = invalid.get(col, 0) + int(empty.sum())
    if invalid:
        raise ValueError(f"gp_link_df contains invalid BEDPE fields: {invalid}")

    if (out["peak_start"] < 0).any():
        raise ValueError("BEDPE peak_start values must be non-negative.")
    if (out["peak_end"] <= out["peak_start"]).any():
        raise ValueError("BEDPE peak_end values must be greater than peak_start.")
    if (out["tss"] <= 0).any():
        raise ValueError("TSS values must be positive 1-based coordinates.")
    if (out[score_col] < 0).any():
        raise ValueError(f"{score_col} values must be non-negative for CoolBox score scaling.")

    strand_values = set(out["strand"].astype(str))
    invalid_strands = sorted(strand_values.difference({"+", "-", "."}))
    if invalid_strands:
        raise ValueError(f"Unexpected strand values for BEDPE: {invalid_strands}")

    name = (
        out["Gene"].astype(str).str.replace(r"[\t\r\n]+", "_", regex=True)
        + "|"
        + out["Peak"].astype(str).str.replace(r"[\t\r\n]+", "_", regex=True)
    )
    bedpe = pd.DataFrame(
        {
            "chrom1": out["peak_chr"].astype(str),
            "start1": out["peak_start"].astype(np.int64),
            "end1": out["peak_end"].astype(np.int64),
            "chrom2": out["chr"].astype(str),
            "start2": (out["tss"] - 1).astype(np.int64),
            "end2": out["tss"].astype(np.int64),
            "name": name,
            "score": out[score_col].astype(float),
            "strand1": ".",
            "strand2": out["strand"].astype(str),
        }
    )
    if bedpe.empty:
        raise ValueError("No BEDPE rows were generated.")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bedpe.to_csv(out_path, sep="\t", header=False, index=False)
    return out_path


def _draw_signed_arcs_on_axes(
    ax,
    gp_link_df: pd.DataFrame,
    *,
    weight_col: str,
    pos_color: str,
    neg_color: str,
    max_abs_weight: Optional[float] = None,
    min_alpha: float = 0.10,
    max_alpha: float = 0.85,
    n_arc_points: int = 200,
) -> None:
    """Draw peak->TSS semi-elliptical arcs on a single matplotlib axes.

    Each arc's color is set by the sign of ``weight_col`` (positive: ``pos_color``,
    negative: ``neg_color``), and its transparency scales with ``|weight|`` so
    that the largest absolute weight reaches ``max_alpha`` and the smallest
    reaches ``min_alpha``.
    """
    if weight_col not in gp_link_df.columns:
        raise KeyError(f"gp_link_df is missing required column: {weight_col!r}")

    if gp_link_df.empty:
        return

    weights = gp_link_df[weight_col].astype(float).to_numpy()
    if max_abs_weight is None:
        max_abs_weight = float(np.nanmax(np.abs(weights))) if weights.size else 1.0
    if not np.isfinite(max_abs_weight) or max_abs_weight <= 0:
        max_abs_weight = 1.0

    peak_mids = (gp_link_df["peak_start"].astype(float) + gp_link_df["peak_end"].astype(float)) / 2.0
    tss_vals = gp_link_df["tss"].astype(float)
    diameters_all = (peak_mids - tss_vals).abs().to_numpy()
    max_diameter = float(diameters_all.max()) if diameters_all.size else 1.0
    if max_diameter <= 0:
        max_diameter = 1.0

    theta = np.linspace(0.0, np.pi, n_arc_points)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    for peak_mid, tss, weight in zip(peak_mids.to_numpy(), tss_vals.to_numpy(), weights):
        if not (np.isfinite(peak_mid) and np.isfinite(tss) and np.isfinite(weight)):
            continue
        color = pos_color if weight > 0 else neg_color
        alpha_raw = abs(weight) / max_abs_weight
        alpha = min_alpha + (max_alpha - min_alpha) * float(np.clip(alpha_raw, 0.0, 1.0))

        left, right = (peak_mid, tss) if peak_mid <= tss else (tss, peak_mid)
        center = (left + right) / 2.0
        diameter = right - left
        height = (diameter / max_diameter) ** 0.3 if diameter > 0 else 0.0

        xs = center + (diameter / 2.0) * cos_theta
        ys = height * sin_theta
        ax.fill(xs, ys, color=color, alpha=alpha, edgecolor="none", linewidth=0.0)


def plot_signed_arcs_stacked(
    gp_link_dfs,
    *,
    pls_cmp: str,
    gene: str,
    out_path: Union[str, Path],
    pos_color: str = "#1f77b4",
    neg_color: str = "#cc4c02",
    fig_width_inches: float = 12.0,
    panel_height_inches: float = 2.0,
    gene_panel_height_inches: float = 0.5,
    padding_bp: int = 10000,
    share_alpha_scale: bool = True,
    title: Optional[str] = None,
):
    """Stack one or more arc panels (e.g. source on top, target below) sharing a
    genomic x-axis, with a gene-body strip at the bottom.

    Parameters
    ----------
    gp_link_dfs
        Mapping of panel label → ``gp_link_df`` (the output of
        ``select_gene_peak_link_df``). Each df must contain
        ``peak_start``, ``peak_end``, ``tss``, ``peak_chr``, ``chr``,
        ``start``, ``end``, ``strand``, ``Gene``, and
        ``f"{pls_cmp}_peak_weight"``.
    pls_cmp
        PLS component label used to locate the peak weight column.
    gene
        Gene symbol the links anchor on (used in the gene-body strip).
    out_path
        Destination PDF/PNG path.
    pos_color, neg_color
        Colors for arcs associated with positive / negative PLS peak weights.
    share_alpha_scale
        If True, scale alpha jointly across all panels using the global max
        ``|weight|``; otherwise each panel rescales independently.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FuncFormatter

    if not gp_link_dfs:
        raise ValueError("gp_link_dfs must contain at least one entry.")

    weight_col = f"{pls_cmp}_peak_weight"

    chrom_values = set()
    lows: list[float] = []
    highs: list[float] = []
    for df in gp_link_dfs.values():
        if df.empty:
            continue
        if weight_col not in df.columns:
            raise KeyError(f"gp_link_df is missing required column: {weight_col!r}")
        chrom_values.update(df["peak_chr"].astype(str).unique())
        chrom_values.update(df["chr"].astype(str).unique())
        # Only arc anchors (peak ± TSS) drive the x-range so a long gene body
        # cannot saturate the figure. The gene-body strip is clipped to this
        # window below.
        lows.extend(df["peak_start"].astype(float).tolist())
        lows.extend(df["tss"].astype(float).tolist())
        highs.extend(df["peak_end"].astype(float).tolist())
        highs.extend(df["tss"].astype(float).tolist())

    if not lows or not highs:
        raise ValueError("All provided gp_link_dfs are empty; nothing to plot.")
    if len(chrom_values) > 1:
        raise ValueError(
            f"Expected all links on a single chromosome, got: {sorted(chrom_values)}"
        )
    chrom = next(iter(chrom_values))
    x_min = max(0.0, float(min(lows)) - float(padding_bp))
    x_max = float(max(highs)) + float(padding_bp)

    if share_alpha_scale:
        per_panel_max = [
            float(df[weight_col].abs().max()) if not df.empty else 0.0
            for df in gp_link_dfs.values()
        ]
        global_max = max(per_panel_max) if per_panel_max else 0.0
        max_abs_weight_per_panel = {
            label: (global_max if global_max > 0 else 1.0)
            for label in gp_link_dfs
        }
    else:
        max_abs_weight_per_panel = {
            label: (float(df[weight_col].abs().max()) if not df.empty else 1.0)
            for label, df in gp_link_dfs.items()
        }

    n_panels = len(gp_link_dfs)
    height_ratios = [1.0] * n_panels + [
        max(0.1, gene_panel_height_inches / max(0.01, panel_height_inches))
    ]
    fig_height = panel_height_inches * n_panels + gene_panel_height_inches + 0.9
    fig, axes = plt.subplots(
        n_panels + 1,
        1,
        figsize=(fig_width_inches, fig_height),
        gridspec_kw=dict(height_ratios=height_ratios, hspace=0.18),
        sharex=True,
        constrained_layout=True,
    )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    else:
        axes = list(axes)

    for ax, (label, df) in zip(axes[:n_panels], gp_link_dfs.items()):
        _draw_signed_arcs_on_axes(
            ax,
            df,
            weight_col=weight_col,
            pos_color=pos_color,
            neg_color=neg_color,
            max_abs_weight=max_abs_weight_per_panel[label],
        )
        ax.set_ylim(0, 1.10)
        ax.set_yticks([])
        ax.set_xlim(x_min, x_max)
        ax.set_title(str(label), loc="left", fontsize=11, pad=2)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="x", labelbottom=False)

    # Gene-body strip at the bottom (shared x-axis). The full gene span is
    # clipped to the visible window so a long gene body cannot dominate the
    # figure; truncated edges are marked with a small chevron.
    ax_gene = axes[-1]
    first_df = next(df for df in gp_link_dfs.values() if not df.empty)
    if "start" in first_df.columns and "end" in first_df.columns:
        gene_start = float(pd.to_numeric(first_df["start"], errors="coerce").iloc[0])
        gene_end = float(pd.to_numeric(first_df["end"], errors="coerce").iloc[0])
    else:
        gene_start = float(first_df["tss"].iloc[0])
        gene_end = gene_start + 1.0

    strand_vals = first_df["strand"].astype(str) if "strand" in first_df.columns else pd.Series([""])
    strand = strand_vals.iloc[0] if len(strand_vals) else ""

    clipped_start = max(gene_start, x_min)
    clipped_end = min(gene_end, x_max)
    truncated_left = gene_start < x_min
    truncated_right = gene_end > x_max
    rect_h = 0.5
    rect_y0 = 0.25
    if clipped_end > clipped_start:
        ax_gene.add_patch(
            Rectangle(
                (clipped_start, rect_y0),
                clipped_end - clipped_start,
                rect_h,
                facecolor="#cc4c02",
                edgecolor="black",
                alpha=0.6,
                linewidth=0.5,
            )
        )

    # Chevron markers at clipped edges to signal the gene continues off-axis.
    chevron_y = rect_y0 + rect_h / 2.0
    if truncated_left and clipped_end > clipped_start:
        ax_gene.annotate(
            "", xy=(x_min, chevron_y),
            xytext=(x_min + (x_max - x_min) * 0.012, chevron_y),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8),
        )
    if truncated_right and clipped_end > clipped_start:
        ax_gene.annotate(
            "", xy=(x_max, chevron_y),
            xytext=(x_max - (x_max - x_min) * 0.012, chevron_y),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8),
        )

    # Strand indicator: tiny arrow inside the visible portion of the gene body.
    if strand in {"+", "-"} and clipped_end > clipped_start:
        strand_marker = "▶" if strand == "+" else "◀"
        ax_gene.text(
            (clipped_start + clipped_end) / 2.0,
            rect_y0 + rect_h + 0.08,
            strand_marker,
            ha="center", va="bottom", fontsize=8, color="black",
        )

    label_margin = (x_max - x_min) * 0.01
    raw_label_x = (clipped_start + clipped_end) / 2.0
    label_x = float(np.clip(raw_label_x, x_min + label_margin, x_max - label_margin))
    ax_gene.text(
        label_x, rect_y0 + rect_h / 2.0, gene,
        ha="center", va="center", fontsize=10, fontweight="bold", color="black",
    )
    ax_gene.set_xlim(x_min, x_max)
    ax_gene.set_ylim(0, 1)
    ax_gene.set_yticks([])
    for spine in ("top", "right", "left"):
        ax_gene.spines[spine].set_visible(False)
    ax_gene.set_xlabel(f"{chrom} (bp)")
    ax_gene.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / 1000:,.0f} Kb"))

    legend_handles = [
        Patch(facecolor=pos_color, edgecolor="none", alpha=0.7, label=f"{weight_col} > 0"),
        Patch(facecolor=neg_color, edgecolor="none", alpha=0.7, label=f"{weight_col} < 0"),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=8,
        frameon=False,
    )

    if title:
        fig.suptitle(title, fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    return fig


__all__ = [
    "GMMThresholdResult",
    "add_gene_and_peak_columns",
    "assign_regulatory_region",
    "compute_gene_peak_distance",
    "extract_gene_name",
    "extract_peak_gene_connections",
    "filter_by_attention_threshold",
    "fit_attention_gmm",
    "get_gmm_attention_threshold",
    "load_attention_matrix",
    "merge_with_gene_annotations",
    "parse_gtf_file",
    "parse_peak",
    "plot_attention_distribution",
    "plot_distance_distribution",
    "plot_signed_arcs_stacked",
    "save_attention_outputs",
    "save_gene_peak_links_bedpe",
]

"""Visual inspection figures for the cross-domain bivariate-Moran shortlist."""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import cKDTree


ROOT = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/rna_atac_figures/fig4_linked_peak_candidates_20261018"
BASE = Path(os.environ['SPATIALJEPA_HOME'])
DATA = BASE / "data/aligned_data"
LINKS_FILE = BASE / "outputs/MultiGATE/attention_analysis/merged_df_threshold.csv"
SUMMARY_FILE = ROOT / "bivariate_gene_peak/bivariate_morans_all_linked_genes.csv"
OUT = ROOT / "bivariate_gene_peak/shortlist"
OUT.mkdir(exist_ok=True)

GENES = ["Mobp", "Mal", "Rgs9", "Phactr1"]
PEAK_PLOT_GENES = ["Mobp", "Mal", "Phactr1"]
# Mobp and Phactr1 use the Fig. 4a Leiden colours (tab10): cluster 4 and cluster 0.
LEIDEN_COLORS = plt.get_cmap("tab10").colors
COLORS = {
    "Mobp": LEIDEN_COLORS[4],
    "Mal": "#009E73",
    "Rgs9": "#D55E00",
    "Phactr1": LEIDEN_COLORS[0],
}


def read_dense_selected(path, genes):
    with h5py.File(path) as handle:
        var_names = handle["var/_index"].asstr()[:]
        lookup = {gene: index for index, gene in enumerate(var_names)}
        indices = np.array([lookup[gene] for gene in genes])
        order = np.argsort(indices)
        inverse = np.argsort(order)
        values = np.asarray(handle["X"][:, indices[order]], float)[:, inverse]
        xy = handle["obsm/spatial"][:]
    return values, xy


def read_sparse_atac(path):
    with h5py.File(path) as handle:
        var_names = handle["var/_index"].asstr()[:]
        x = handle["X"]
        matrix = sp.csr_matrix(
            (x["data"][:], x["indices"][:], x["indptr"][:]),
            shape=tuple(x.attrs["shape"]),
        )
        xy = handle["obsm/spatial"][:]
    return matrix, var_names, xy


def linked_scores(matrix, peak_names, links, genes):
    peak_index = {peak: index for index, peak in enumerate(peak_names)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    weight = sp.csr_matrix(
        (
            links.Attention.to_numpy(float),
            (links.Peak.map(peak_index), links.Gene.map(gene_index)),
        ),
        shape=(len(peak_names), len(genes)),
    )
    return (matrix @ weight).toarray()


def aggregate_by_source(values, xy, source_xy):
    lookup = {tuple(point): index for index, point in enumerate(source_xy)}
    assignment = np.array([lookup[tuple(point)] for point in xy])
    occupied, inverse, counts = np.unique(
        assignment, return_inverse=True, return_counts=True
    )
    aggregate = sp.csr_matrix(
        (
            1 / counts[inverse],
            (inverse, np.arange(len(values))),
        ),
        shape=(len(occupied), len(values)),
    )
    return occupied, aggregate @ values


def gaussian_smooth(values, xy, full_xy, sigma_spots=1.0):
    spacing = float(np.median(cKDTree(full_xy).query(full_xy, k=2)[0][:, 1]))
    sigma = sigma_spots * spacing
    tree = cKDTree(xy)
    distances = tree.sparse_distance_matrix(
        tree, 3 * sigma, output_type="coo_matrix"
    )
    graph = sp.csr_matrix(
        (
            np.exp(-0.5 * (distances.data / sigma) ** 2),
            (distances.row, distances.col),
        ),
        shape=(len(xy), len(xy)),
    )
    weights = sp.diags(1 / np.asarray(graph.sum(axis=1)).ravel()) @ graph
    return weights @ values


def r2_contour(source_xy):
    with h5py.File(DATA / "source_atac_aligned_with_latents.h5ad") as handle:
        group = handle["obs/RNA_clusters"]
        categories = group["categories"].asstr()[:]
        labels = categories[group["codes"][:]]
    spacing = float(np.median(cKDTree(source_xy).query(source_xy, k=2)[0][:, 1]))
    tree = cKDTree(source_xy)
    sigma = 1.5 * spacing
    distances = tree.sparse_distance_matrix(
        tree, 3 * sigma, output_type="coo_matrix"
    )
    weights = sp.csr_matrix(
        (
            np.exp(-0.5 * (distances.data / sigma) ** 2),
            (distances.row, distances.col),
        ),
        shape=(len(source_xy), len(source_xy)),
    )
    values = np.asarray(weights @ (labels == "R2")).ravel()
    values /= np.asarray(weights.sum(axis=1)).ravel()
    triangulation = mtri.Triangulation(source_xy[:, 0], source_xy[:, 1])
    vertices = source_xy[triangulation.triangles]
    edge_lengths = np.linalg.norm(vertices - np.roll(vertices, 1, axis=1), axis=2)
    triangulation.set_mask(edge_lengths.max(axis=1) > 1.8 * spacing)
    return triangulation, values


def robust_limits(source, target, zero_floor):
    pooled = np.concatenate([source, target])
    lower = 0.0 if zero_floor else float(np.percentile(pooled, 0.5))
    upper = float(np.percentile(pooled, 99.5))
    return lower, upper


def draw(ax, xy, values, cmap, limits, contour=None):
    order = np.argsort(values, kind="stable")
    art = ax.scatter(
        xy[order, 0],
        xy[order, 1],
        c=values[order],
        s=6,
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        linewidths=0,
        rasterized=True,
    )
    if contour is not None:
        triangulation, contour_values = contour
        lines = ax.tricontour(
            triangulation,
            contour_values,
            levels=[0.5],
            colors=["#31d8e7"],
            linewidths=0.8,
        )
        lines.set_path_effects(
            [pe.Stroke(linewidth=1.4, foreground="0.12"), pe.Normal()]
        )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return art


def save(fig, stem, dpi=190):
    for suffix in (".png", ".pdf", ".svg"):  # .svg: read by assemble_transfer_figure.py
        fig.savefig(OUT / f"{stem}{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def map_grid(summary, coordinates, expression, peaks, stem, subtitle, contour=None):
    fig, axes = plt.subplots(
        4, len(GENES), figsize=(2.55 * len(GENES), 9.2),
        squeeze=False, layout="constrained"
    )
    specs = [
        ("Source expression", "source", expression, "viridis"),
        ("Source linked peaks", "source", peaks, "magma"),
        ("Target expression", "target", expression, "viridis"),
        ("Target linked peaks", "target", peaks, "magma"),
    ]
    for column, gene in enumerate(GENES):
        expression_limits = robust_limits(
            expression["source"][:, column],
            expression["target"][:, column],
            np.min(np.concatenate([
                expression["source"][:, column],
                expression["target"][:, column],
            ])) >= 0,
        )
        peak_limits = robust_limits(
            peaks["source"][:, column], peaks["target"][:, column], True
        )
        for row, (label, domain, values, cmap) in enumerate(specs):
            limits = expression_limits if "expression" in label else peak_limits
            art = draw(
                axes[row, column],
                coordinates[domain],
                values[domain][:, column],
                cmap,
                limits,
                contour,
            )
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=10)
            fig.colorbar(
                art, ax=axes[row, column], fraction=0.035, pad=0.02
            ).ax.tick_params(labelsize=6)
        values = summary.loc[gene]
        peak_label = "peak" if int(values.n_peaks) == 1 else "peaks"
        rank_label = (
            str(int(values.product_rank))
            if np.isfinite(values.product_rank)
            else "n/a (negative I)"
        )
        axes[0, column].set_title(
            f"{gene} ({int(values.n_peaks)} {peak_label}; rank {rank_label})\n"
            f"$I$: {values.I_source:.3f} → {values.I_target:.3f}",
            fontsize=9.5,
        )
    fig.suptitle(
        "Cross-domain bivariate-Moran shortlist\n" + subtitle,
        fontsize=14,
    )
    fig.supxlabel(
        "Matched source–target scale within each expression or linked-peak pair"
        + ("; cyan = source R2 outline" if contour is not None else ""),
        fontsize=10,
    )
    save(fig, stem)


def spread(values, gap=0.006):
    order = np.argsort(values)
    placed = np.asarray(values, float).copy()
    for current, previous in zip(order[1:], order[:-1]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def paired_plot(
    summary, statistic, min_peaks=None, highlighted_genes=None, output_suffix=""
):
    plotted = (
        summary.loc[summary.n_peaks >= min_peaks]
        if min_peaks is not None
        else summary
    )
    if highlighted_genes is None:
        highlighted_genes = PEAK_PLOT_GENES
    highlighted = [gene for gene in highlighted_genes if gene in plotted.index]
    fig, ax = plt.subplots(figsize=(2.45, 3.1))
    for row in plotted.itertuples():
        ax.plot(
            [0, 1], [row.I_source, row.I_target],
            lw=0.3, color="0.7", alpha=0.38, zorder=1
        )
    for x, column in ((0, "I_source"), (1, "I_target")):
        ax.hlines(
            float(getattr(plotted[column], statistic)()),
            x - 0.18, x + 0.18, color="black", lw=1.3, zorder=3
        )
    label_y = spread(plotted.loc[highlighted, "I_target"].to_numpy())
    for gene, y_text in zip(highlighted, label_y):
        row = summary.loc[gene]
        color = COLORS[gene]
        ax.plot(
            [0, 1], [row.I_source, row.I_target],
            lw=1.5, color=color, zorder=4
        )
        ax.scatter(
            [0, 1], [row.I_source, row.I_target],
            s=22, color=color, edgecolor="black", linewidths=0.4, zorder=5
        )
        ax.annotate(
            gene,
            xy=(1, row.I_target),
            xytext=(1.08, y_text),
            textcoords="data",
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6},
            fontsize=7,
            fontweight="bold",
            color=color,
            ha="left",
            va="center",
            clip_on=False,
        )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["source", "target"], fontsize=8)
    ax.set_xlim(-0.24, 1.5)
    ax.set_ylabel(
        "Bivariate Moran's $I$\nexpression $\\rightarrow$ linked peaks",
        fontsize=8,
    )
    criterion = (
        f"; $\\geq${min_peaks} linked peaks" if min_peaks is not None else ""
    )
    ax.set_title(
        f"Black bars: {statistic}{criterion}\n$n={len(plotted):,}$ genes",
        fontsize=8,
    )
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    suffix = f"_ge{min_peaks}peaks" if min_peaks is not None else ""
    save(
        fig,
        f"shortlist_paired_bivariate_{statistic}{suffix}{output_suffix}",
        dpi=220,
    )


def main():
    summary = pd.read_csv(SUMMARY_FILE).set_index("gene")
    positive = (summary.I_source > 0) & (summary.I_target > 0)
    summary["cross_domain_product"] = summary.I_source * summary.I_target
    summary["product_rank"] = np.nan
    summary.loc[positive, "product_rank"] = (
        summary.loc[positive, "cross_domain_product"]
        .rank(method="min", ascending=False)
    )
    links = pd.read_csv(LINKS_FILE).drop_duplicates(["Gene", "Peak"])
    links = links.loc[links.Gene.isin(GENES)].copy()
    source_expression, source_rna_xy = read_dense_selected(
        DATA / "source_rna_aligned_with_latents.h5ad", GENES
    )
    target_expression, target_rna_xy = read_dense_selected(
        DATA / "target_rna_aligned_with_latents.h5ad", GENES
    )
    source_atac, source_peak_names, source_atac_xy = read_sparse_atac(
        DATA / "source_atac_aligned_with_latents.h5ad"
    )
    target_atac, target_peak_names, target_atac_xy = read_sparse_atac(
        DATA / "target_atac_aligned_with_latents.h5ad"
    )
    assert np.array_equal(source_rna_xy, source_atac_xy)
    source_peaks = linked_scores(
        source_atac, source_peak_names, links, GENES
    )
    target_peaks = linked_scores(
        target_atac, target_peak_names, links, GENES
    )

    individual_coordinates = {
        "source": source_atac_xy,
        # Target expression and peaks have modality-specific coordinates; the
        # grid function receives each row separately below via a dedicated pass.
        "target": target_rna_xy,
    }
    individual_expression = {
        "source": source_expression,
        "target": target_expression,
    }
    individual_peak_coordinates = {
        "source": source_atac_xy,
        "target": target_atac_xy,
    }

    # For individual profiles, expression and peaks use different target
    # coordinates. Draw this case explicitly rather than forcing one coordinate.
    original_map_grid = map_grid
    common_rna_occupied, target_expression_lattice = aggregate_by_source(
        target_expression, target_rna_xy, source_atac_xy
    )
    common_atac_occupied, target_peaks_lattice = aggregate_by_source(
        target_peaks, target_atac_xy, source_atac_xy
    )
    joint = np.intersect1d(common_rna_occupied, common_atac_occupied)
    rna_rows = {spot: row for row, spot in enumerate(common_rna_occupied)}
    atac_rows = {spot: row for row, spot in enumerate(common_atac_occupied)}
    rna_take = np.array([rna_rows[spot] for spot in joint])
    atac_take = np.array([atac_rows[spot] for spot in joint])
    lattice_expression = {
        "source": source_expression[joint],
        "target": target_expression_lattice[rna_take],
    }
    lattice_peaks = {
        "source": source_peaks[joint],
        "target": target_peaks_lattice[atac_take],
    }
    lattice_coordinates = {
        "source": source_atac_xy[joint],
        "target": source_atac_xy[joint],
    }

    paired_plot(summary, "mean")
    paired_plot(summary, "median")
    paired_plot(summary, "mean", min_peaks=3)
    paired_plot(summary, "median", min_peaks=3)
    paired_plot(
        summary,
        "mean",
        highlighted_genes=["Mobp", "Phactr1"],
        output_suffix="_no_mal",
    )
    original_map_grid(
        summary,
        lattice_coordinates,
        lattice_expression,
        lattice_peaks,
        "shortlist_common_lattice",
        "Common 8,945-spot lattice used for bivariate Moran's I",
    )
    contour = r2_contour(source_atac_xy)
    original_map_grid(
        summary,
        lattice_coordinates,
        lattice_expression,
        lattice_peaks,
        "shortlist_common_lattice_R2",
        "Common 8,945-spot lattice used for bivariate Moran's I",
        contour=contour,
    )

    # Individual-profile figure with modality-specific target coordinates.
    fig, axes = plt.subplots(
        4, len(GENES), figsize=(2.55 * len(GENES), 9.2),
        squeeze=False, layout="constrained"
    )
    for column, gene in enumerate(GENES):
        expression_limits = robust_limits(
            source_expression[:, column], target_expression[:, column],
            np.min(np.concatenate([
                source_expression[:, column], target_expression[:, column]
            ])) >= 0,
        )
        peak_limits = robust_limits(
            source_peaks[:, column], target_peaks[:, column], True
        )
        rows = [
            ("Source expression", source_rna_xy, source_expression[:, column],
             "viridis", expression_limits),
            ("Source linked peaks", source_atac_xy, source_peaks[:, column],
             "magma", peak_limits),
            ("Target expression", target_rna_xy, target_expression[:, column],
             "viridis", expression_limits),
            ("Target linked peaks", target_atac_xy, target_peaks[:, column],
             "magma", peak_limits),
        ]
        for row, (label, xy, values, cmap, limits) in enumerate(rows):
            art = draw(axes[row, column], xy, values, cmap, limits)
            if column == 0:
                axes[row, column].set_ylabel(label, fontsize=10)
            fig.colorbar(
                art, ax=axes[row, column], fraction=0.035, pad=0.02
            ).ax.tick_params(labelsize=6)
        values = summary.loc[gene]
        peak_label = "peak" if int(values.n_peaks) == 1 else "peaks"
        rank_label = (
            str(int(values.product_rank))
            if np.isfinite(values.product_rank)
            else "n/a (negative I)"
        )
        axes[0, column].set_title(
            f"{gene} ({int(values.n_peaks)} {peak_label}; rank {rank_label})\n"
            f"$I$: {values.I_source:.3f} → {values.I_target:.3f}",
            fontsize=9.5,
        )
    fig.suptitle(
        "Cross-domain bivariate-Moran shortlist\n"
        "Individual profiles at modality-specific transferred coordinates",
        fontsize=14,
    )
    fig.supxlabel(
        "Matched source–target scale within each expression or linked-peak pair",
        fontsize=10,
    )
    save(fig, "shortlist_individual_profiles")

    # Manuscript-style two-row peak-only display, preserving individual target
    # profiles and modality-specific transferred coordinates. Each gene has one
    # pooled source/target cap and is rescaled to 0--1.
    fig, axes = plt.subplots(
        2, len(PEAK_PLOT_GENES),
        figsize=(2.55 * len(PEAK_PLOT_GENES), 5.25),
        squeeze=False, layout="constrained"
    )
    for column, gene in enumerate(PEAK_PLOT_GENES):
        score_column = GENES.index(gene)
        n_peaks = int(summary.loc[gene, "n_peaks"])
        peak_label = "peak" if n_peaks == 1 else "peaks"
        cap = float(np.percentile(
            np.concatenate([
                source_peaks[:, score_column],
                target_peaks[:, score_column],
            ]),
            99.9,
        ))
        rows = [
            ("Source peaks", source_atac_xy, source_peaks[:, score_column]),
            ("Target peaks", target_atac_xy, target_peaks[:, score_column]),
        ]
        for row, (label, xy, values) in enumerate(rows):
            display = np.minimum(values, cap) / cap
            art = draw(axes[row, column], xy, display, "magma", (0, 1))
            axes[row, column].set_title(
                (f"{gene} ({n_peaks} {peak_label})\n"
                 if row == 0 else "")
                + label,
                fontsize=10.5,
            )
            fig.colorbar(
                art, ax=axes[row, column], fraction=0.035, pad=0.02
            ).ax.tick_params(labelsize=6)
    fig.suptitle("Attention-weighted linked-peak accessibility", fontsize=14)
    fig.supxlabel(
        "Individual profiles at transferred coordinates; shared source–target "
        "99.9th-percentile cap within each gene, rescaled to 0–1",
        fontsize=9.5,
    )
    save(fig, "shortlist_peaks_two_row", dpi=220)

    # Light Gaussian smoothing on the common target-occupied source lattice.
    # This avoids treating coincident target nuclei as independent spatial sites.
    smooth_xy = source_atac_xy[common_atac_occupied]
    smooth_source = gaussian_smooth(
        source_peaks[common_atac_occupied],
        smooth_xy,
        source_atac_xy,
        sigma_spots=1,
    )
    smooth_target = gaussian_smooth(
        target_peaks_lattice,
        smooth_xy,
        source_atac_xy,
        sigma_spots=1,
    )
    fig, axes = plt.subplots(
        2, len(PEAK_PLOT_GENES),
        figsize=(2.55 * len(PEAK_PLOT_GENES), 5.25),
        squeeze=False, layout="constrained"
    )
    for column, gene in enumerate(PEAK_PLOT_GENES):
        score_column = GENES.index(gene)
        n_peaks = int(summary.loc[gene, "n_peaks"])
        peak_label = "peak" if n_peaks == 1 else "peaks"
        cap = float(np.percentile(
            np.concatenate([
                smooth_source[:, score_column],
                smooth_target[:, score_column],
            ]),
            99,
        ))
        rows = [
            ("Source peaks", smooth_source[:, score_column]),
            ("Target peaks", smooth_target[:, score_column]),
        ]
        for row, (label, values) in enumerate(rows):
            display = np.minimum(values, cap) / cap
            art = draw(axes[row, column], smooth_xy, display, "magma", (0, 1))
            axes[row, column].set_title(
                (f"{gene} ({n_peaks} {peak_label})\n"
                 if row == 0 else "")
                + label,
                fontsize=10.5,
            )
            fig.colorbar(
                art, ax=axes[row, column], fraction=0.035, pad=0.02
            ).ax.tick_params(labelsize=6)
    fig.suptitle("Attention-weighted linked-peak accessibility", fontsize=14)
    fig.supxlabel(
        "Common target-occupied source lattice; Gaussian smoothing "
        r"$\sigma=1$ spot; shared source–target 99th-percentile cap "
        "within each gene, rescaled to 0–1",
        fontsize=9.5,
    )
    save(fig, "shortlist_peaks_two_row_smooth1", dpi=220)

    no_mal_genes = ["Mobp", "Phactr1"]
    fig, axes = plt.subplots(
        2, len(no_mal_genes),
        figsize=(2.55 * len(no_mal_genes), 5.25),
        squeeze=False, layout="constrained"
    )
    for column, gene in enumerate(no_mal_genes):
        score_column = GENES.index(gene)
        n_peaks = int(summary.loc[gene, "n_peaks"])
        peak_label = "peak" if n_peaks == 1 else "peaks"
        cap = float(np.percentile(
            np.concatenate([
                smooth_source[:, score_column],
                smooth_target[:, score_column],
            ]),
            99,
        ))
        rows = [
            ("Source peaks", smooth_source[:, score_column]),
            ("Target peaks", smooth_target[:, score_column]),
        ]
        for row, (label, values) in enumerate(rows):
            display = np.minimum(values, cap) / cap
            art = draw(axes[row, column], smooth_xy, display, "magma", (0, 1))
            axes[row, column].set_title(
                (f"{gene} ({n_peaks} {peak_label})\n"
                 if row == 0 else "")
                + label,
                fontsize=10.5,
            )
            fig.colorbar(
                art, ax=axes[row, column], fraction=0.035, pad=0.02
            ).ax.tick_params(labelsize=6)
    fig.suptitle("Attention-weighted linked-peak accessibility", fontsize=14)
    fig.supxlabel(
        "Common target-occupied source lattice; Gaussian smoothing "
        r"$\sigma=1$ spot; shared source–target 99th-percentile cap "
        "within each gene, rescaled to 0–1",
        fontsize=9.5,
    )
    save(fig, "shortlist_peaks_two_row_smooth1_no_mal", dpi=220)

    summary.loc[GENES].to_csv(OUT / "shortlist_metrics.csv")
    print(
        summary.loc[
            GENES,
            ["n_peaks", "I_source", "I_target", "delta_I",
             "cross_domain_product", "product_rank"],
        ].to_string()
    )


if __name__ == "__main__":
    main()

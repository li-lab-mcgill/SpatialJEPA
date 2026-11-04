"""Bivariate Moran's I between gene expression and linked-peak accessibility.

Source and target are evaluated on the same occupied source-spot lattice and
the same six-neighbor graph used for the univariate linked-peak analysis.
Target nuclei are averaged within their assigned source spot before scoring.
Every shared RNA gene with at least one retained attention link is included;
there is no minimum linked-peak count.
The directed statistic is expression against the spatial lag of linked peaks:

    I(gene, peaks) = mean(z_gene * W @ z_linked_peaks)

where both fields are standardized and W is row-standardized. The reverse
orientation and their mean are saved as diagnostics because row-standardizing
a symmetric adjacency matrix makes W itself slightly asymmetric.
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
from pathlib import Path
import json

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph


OUT = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/rna_atac_figures/fig4_linked_peak_candidates_20261018" / "bivariate_gene_peak"
OUT.mkdir(exist_ok=True)
ROOT = OUT.parent
DATA = Path(f"{os.environ['SPATIALJEPA_HOME']}/data/aligned_data")
LINKS_FILE = Path(
    f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/"
    "attention_analysis/merged_df_threshold.csv"
)
RNA_FILES = {
    "source": DATA / "source_rna_aligned_with_latents.h5ad",
    "target": DATA / "target_rna_aligned_with_latents.h5ad",
}
TOP_N = 5
HIGHLIGHT_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00"]


def read_metadata(path):
    with h5py.File(path) as handle:
        var_names = handle["var/_index"].asstr()[:]
        xy = handle["obsm/spatial"][:]
        obs_names = handle["obs/_index"].asstr()[:]
    return var_names, xy, obs_names


def aggregate_dense_rna(path, genes, aggregate, take_rows, batch_size=128):
    """Read dense RNA in bounded-memory column batches and aggregate rows."""
    with h5py.File(path) as handle:
        var_names = handle["var/_index"].asstr()[:]
        index = {gene: i for i, gene in enumerate(var_names)}
        keep = np.array([index[gene] for gene in genes])
        output = np.empty((len(take_rows), len(genes)), dtype=np.float32)
        for start in range(0, len(genes), batch_size):
            stop = min(start + batch_size, len(genes))
            batch_indices = keep[start:stop]
            order = np.argsort(batch_indices)
            inverse = np.argsort(order)
            values = np.asarray(
                handle["X"][:, batch_indices[order]], dtype=np.float32
            )[:, inverse]
            if aggregate is not None:
                values = aggregate @ values
            output[:, start:stop] = values[take_rows]
    return output


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


def link_weight_matrix(links, peak_names, genes):
    peak_index = {peak: index for index, peak in enumerate(peak_names)}
    gene_index = {gene: index for index, gene in enumerate(genes)}
    assert links.Peak.isin(peak_index).all()
    return sp.csr_matrix(
        (
            links.Attention.to_numpy(float),
            (links.Peak.map(peak_index), links.Gene.map(gene_index)),
        ),
        shape=(len(peak_names), len(genes)),
    )


def standardize(values):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    valid = np.isfinite(std) & (std > 0)
    z = np.full_like(values, np.nan, dtype=float)
    z[:, valid] = (values[:, valid] - mean[valid]) / std[valid]
    return z, valid


def bivariate_i(x, y, weights):
    zx, valid_x = standardize(x)
    zy, valid_y = standardize(y)
    valid = valid_x & valid_y
    result = np.full(x.shape[1], np.nan)
    result[valid] = np.mean(zx[:, valid] * (weights @ zy[:, valid]), axis=0)
    return result


def save(fig, stem, dpi=220):
    for suffix in (".png", ".pdf"):
        fig.savefig(OUT / f"{stem}{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def spread_label_positions(values, minimum_gap):
    """Greedily separate labels while preserving their endpoint order."""
    order = np.argsort(values)
    placed = np.asarray(values, float).copy()
    for current, previous in zip(order[1:], order[:-1]):
        placed[current] = max(placed[current], placed[previous] + minimum_gap)
    return placed


def paired_plot(summary, top_genes):
    fig, ax = plt.subplots(figsize=(2.35, 3.0))
    for row in summary.itertuples():
        ax.plot(
            [0, 1],
            [row.I_source, row.I_target],
            lw=0.3,
            color="0.68",
            alpha=0.42,
            zorder=1,
        )
    for x, column in ((0, "I_source"), (1, "I_target")):
        median = float(summary[column].median())
        ax.hlines(median, x - 0.18, x + 0.18, color="black", lw=1.3, zorder=3)

    target_values = summary.loc[top_genes, "I_target"].to_numpy()
    y_range = float(summary[["I_source", "I_target"]].max().max()
                    - summary[["I_source", "I_target"]].min().min())
    label_positions = spread_label_positions(target_values, max(0.018 * y_range, 0.004))
    for gene, color, label_y in zip(top_genes, HIGHLIGHT_COLORS, label_positions):
        row = summary.loc[gene]
        ax.plot(
            [0, 1],
            [row.I_source, row.I_target],
            lw=1.5,
            color=color,
            zorder=4,
        )
        ax.scatter(
            [0, 1],
            [row.I_source, row.I_target],
            s=22,
            color=color,
            edgecolor="black",
            linewidths=0.4,
            zorder=5,
        )
        ax.annotate(
            gene,
            xy=(1, row.I_target),
            xytext=(1.08, label_y),
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
    ax.set_xlim(-0.24, 1.48)
    ax.set_ylabel(
        "Bivariate Moran's $I$\nexpression $\\rightarrow$ linked peaks",
        fontsize=8,
    )
    ax.tick_params(axis="y", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, "paired_bivariate_morans")


def robust_limits(source, target, lower_zero=False):
    pooled = np.concatenate([source, target])
    low = 0.0 if lower_zero else float(np.percentile(pooled, 0.5))
    high = float(np.percentile(pooled, 99.5))
    if high <= low:
        high = float(np.max(pooled))
    return low, high


def draw_spatial(ax, xy, values, cmap, limits):
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
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return art


def spatial_grid(summary, top_genes, xy, expression, peaks):
    fig, axes = plt.subplots(
        4,
        len(top_genes),
        figsize=(2.55 * len(top_genes), 9.2),
        squeeze=False,
        layout="constrained",
    )
    row_specs = [
        ("Source expression", "source", expression, "viridis"),
        ("Source linked peaks", "source", peaks, "magma"),
        ("Target expression", "target", expression, "viridis"),
        ("Target linked peaks", "target", peaks, "magma"),
    ]
    for column, gene in enumerate(top_genes):
        gene_index = summary.index.get_loc(gene)
        expression_limits = robust_limits(
            expression["source"][:, gene_index],
            expression["target"][:, gene_index],
            lower_zero=np.min(
                np.concatenate([
                    expression["source"][:, gene_index],
                    expression["target"][:, gene_index],
                ])
            ) >= 0,
        )
        peak_limits = robust_limits(
            peaks["source"][:, gene_index],
            peaks["target"][:, gene_index],
            lower_zero=True,
        )
        for row_index, (row_label, domain, values, cmap) in enumerate(row_specs):
            limits = expression_limits if "expression" in row_label else peak_limits
            art = draw_spatial(
                axes[row_index, column],
                xy,
                values[domain][:, gene_index],
                cmap,
                limits,
            )
            if column == 0:
                axes[row_index, column].set_ylabel(row_label, fontsize=10)
            fig.colorbar(
                art,
                ax=axes[row_index, column],
                fraction=0.035,
                pad=0.02,
            ).ax.tick_params(labelsize=6)
        row = summary.loc[gene]
        axes[0, column].set_title(
            f"{gene}\n$I$: {row.I_source:.3f} → {row.I_target:.3f} "
            f"(Δ={row.delta_I:+.3f})",
            fontsize=10,
        )
    fig.suptitle(
        "Largest gains in expression–linked-peak spatial cross-correlation",
        fontsize=14,
    )
    fig.supxlabel(
        "Same occupied source spots and source–target scale within each row pair; "
        "target nuclei averaged per spot",
        fontsize=10,
    )
    save(fig, "top_gain_spatial_patterns", dpi=190)


def main():
    links = pd.read_csv(LINKS_FILE).drop_duplicates(["Gene", "Peak"])
    source_rna_vars, source_xy, source_names = read_metadata(RNA_FILES["source"])
    target_rna_vars, target_xy, target_names = read_metadata(RNA_FILES["target"])
    target_rna_set = set(target_rna_vars)
    linked_gene_set = set(links.Gene)
    genes = [
        gene for gene in source_rna_vars
        if gene in target_rna_set and gene in linked_gene_set
    ]
    links = links.loc[links.Gene.isin(genes)].copy()
    n_peaks = links.groupby("Gene").Peak.nunique().reindex(genes)

    source_atac, source_peak_names, source_atac_xy = read_sparse_atac(
        DATA / "source_atac_aligned_with_latents.h5ad"
    )
    target_atac, target_peak_names, target_atac_xy = read_sparse_atac(
        DATA / "target_atac_aligned_with_latents.h5ad"
    )
    assert np.array_equal(source_xy, source_atac_xy)
    source_link_weights = link_weight_matrix(links, source_peak_names, genes)
    target_link_weights = link_weight_matrix(links, target_peak_names, genes)

    # RNA and ATAC target profiles receive modality-specific transferred
    # coordinates. Aggregate each modality independently, then compare them on
    # the intersection of occupied source spots.
    source_lookup = {tuple(point): index for index, point in enumerate(source_xy)}
    rna_assignment = np.array([source_lookup[tuple(point)] for point in target_xy])
    rna_occupied, rna_inverse, rna_counts = np.unique(
        rna_assignment, return_inverse=True, return_counts=True
    )
    rna_aggregate = sp.csr_matrix(
        (
            1 / rna_counts[rna_inverse],
            (rna_inverse, np.arange(len(target_xy))),
        ),
        shape=(len(rna_occupied), len(target_xy)),
    )
    atac_assignment = np.array(
        [source_lookup[tuple(point)] for point in target_atac_xy]
    )
    atac_occupied, atac_inverse, atac_counts = np.unique(
        atac_assignment, return_inverse=True, return_counts=True
    )
    atac_aggregate = sp.csr_matrix(
        (
            1 / atac_counts[atac_inverse],
            (atac_inverse, np.arange(len(target_atac_xy))),
        ),
        shape=(len(atac_occupied), len(target_atac_xy)),
    )
    joint_occupied = np.intersect1d(atac_occupied, rna_occupied)
    atac_row = {spot: row for row, spot in enumerate(atac_occupied)}
    rna_row = {spot: row for row, spot in enumerate(rna_occupied)}
    atac_take = np.array([atac_row[spot] for spot in joint_occupied])
    rna_take = np.array([rna_row[spot] for spot in joint_occupied])

    expression = {
        "source": aggregate_dense_rna(
            RNA_FILES["source"], genes, None, joint_occupied
        ),
        "target": aggregate_dense_rna(
            RNA_FILES["target"], genes, rna_aggregate, rna_take
        ),
    }
    peaks = {
        "source": (source_atac[joint_occupied] @ source_link_weights).toarray(),
        "target": (
            atac_aggregate @ (target_atac @ target_link_weights)
        )[atac_take].toarray(),
    }

    adjacency = kneighbors_graph(
        source_xy[joint_occupied].astype(float),
        6,
        mode="connectivity",
        include_self=False,
    )
    adjacency = adjacency.maximum(adjacency.T)
    weights = sp.diags(1 / np.asarray(adjacency.sum(axis=1)).ravel()) @ adjacency

    rows = []
    for domain in ("source", "target"):
        forward = bivariate_i(expression[domain], peaks[domain], weights)
        reverse = bivariate_i(peaks[domain], expression[domain], weights)
        rows.append((forward, reverse))
    summary = pd.DataFrame(
        {
            "gene": genes,
            "n_peaks": n_peaks.to_numpy(int),
            "I_source": rows[0][0],
            "I_target": rows[1][0],
            "I_reverse_source": rows[0][1],
            "I_reverse_target": rows[1][1],
        }
    ).set_index("gene")
    summary["I_symmetric_source"] = (
        summary.I_source + summary.I_reverse_source
    ) / 2
    summary["I_symmetric_target"] = (
        summary.I_target + summary.I_reverse_target
    ) / 2
    summary["delta_I"] = summary.I_target - summary.I_source
    summary["delta_I_symmetric"] = (
        summary.I_symmetric_target - summary.I_symmetric_source
    )
    summary = summary.dropna().sort_values("delta_I", ascending=False)
    summary["gain_rank"] = np.arange(1, len(summary) + 1)
    summary.to_csv(OUT / "bivariate_morans_all_linked_genes.csv")

    top_genes = summary.index[:TOP_N].tolist()
    paired_plot(summary, top_genes)

    # Reorder matrices so columns match the gain-sorted summary used by the map code.
    original_index = {gene: index for index, gene in enumerate(genes)}
    selected_columns = [original_index[gene] for gene in summary.index]
    sorted_expression = {
        domain: values[:, selected_columns] for domain, values in expression.items()
    }
    sorted_peaks = {
        domain: values[:, selected_columns] for domain, values in peaks.items()
    }
    spatial_grid(
        summary,
        top_genes,
        source_xy[joint_occupied],
        sorted_expression,
        sorted_peaks,
    )

    previous = pd.read_csv(
        Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/"
             "attention_analysis/gene_peak_bivariate_moran_I.csv")
    ).set_index("gene")
    overlap = summary.index.intersection(previous.index)
    comparison = summary.loc[overlap, ["I_source", "I_target"]].join(
        previous.loc[overlap, ["I_source", "I_target"]],
        lsuffix="_common_lattice",
        rsuffix="_previous",
    )
    comparison.to_csv(OUT / "comparison_to_previous_20_gene_output.csv")

    metadata = {
        "n_genes": int(len(summary)),
        "n_joint_occupied_source_spots": int(len(joint_occupied)),
        "n_rna_occupied_source_spots": int(len(rna_occupied)),
        "n_atac_occupied_source_spots": int(len(atac_occupied)),
        "target_aggregation": (
            "RNA and ATAC target nuclei are averaged independently by their "
            "modality-specific transferred source spot, then restricted to spots "
            "occupied in both modalities."
        ),
        "weights": "Symmetrized 6-nearest-neighbor adjacency, then row-standardized.",
        "statistic": "mean(z_expression * (W @ z_linked_peak_accessibility))",
        "ranking": "I_target - I_source for the directed statistic above.",
        "top_genes": top_genes,
        "reverse_orientation_saved": True,
        "symmetric_average_saved": True,
    }
    (OUT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.head(20).to_string())
    print("\nPrevious-output comparison:")
    print(comparison.head(20).to_string())


if __name__ == "__main__":
    main()

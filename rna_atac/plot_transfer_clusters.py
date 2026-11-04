"""Manuscript Fig. 4a,c from independent source/target Leiden clusters (shared k=150, res=0.9).

Panel a uses shared row/column titles and the colours from run_alignment_and_spatial_plot
in MultiGATE scripts/analyze_rna_atac.py; panel c ports its DE-concordance block (scanpy t-test vs rest,
genes significant in both domains, hexbin of pooled -log p ranks, LOWESS per cluster, per-cluster
Spearman). Target clusters are renamed to the source cluster they overlap most (Hungarian
matching on target vs assigned-source labels), so both domains share one legend; see
transfer_leiden_sweep.py for the clustering.

Each panel is drawn at the physical size of its slot in the assembled figure
(overleaf-cibb-2026/figures/2_source_target_aligned.svg), so it drops in at 100% scale. Type
sizes follow PANEL_RC in analyze_rna_atac.py; dense scatter layers are rasterized, while text,
axes, hexbins and lines stay vector.
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
import argparse
import sys

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import combine_pvalues
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from transfer_target_leiden import load, match_clusters  # noqa: E402

MM = 1 / 25.4
## slot sizes in the assembled Fig. 4 (panel c = rank plot over Spearman stem plot)
PANEL_SIZE_IN = {"a": (74.5 * MM, 59.3 * MM), "c": (53.2 * MM, 60.6 * MM)}
## point sizes as they read in the assembled figure (analyze_rna_atac.py PANEL_FS/PANEL_RC)
FS = {"tick": 5.5, "label": 6.5, "legend": 5.5}
RC = {
    "font.size": FS["tick"], "axes.titlesize": FS["label"], "axes.labelsize": FS["label"],
    "xtick.labelsize": FS["tick"], "ytick.labelsize": FS["tick"], "legend.fontsize": FS["legend"],
    "legend.title_fontsize": FS["legend"], "axes.linewidth": 0.5, "axes.labelpad": 2.0,
    "axes.titlepad": 3.0, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 1.8, "ytick.major.size": 1.8, "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5, "lines.linewidth": 0.9, "patch.linewidth": 0.5,
    "legend.handlelength": 1.3, "legend.handletextpad": 0.45, "legend.labelspacing": 0.28,
    "legend.borderpad": 0.3, "legend.borderaxespad": 0.25, "legend.frameon": False,
    "svg.fonttype": "none", "pdf.fonttype": 42,
}
## panel a matches panel b (scanpy defaults at ~0.345x in its slot: 4.1 pt titles, 3.5 pt labels)
RC_A = {**RC, "font.size": 3.5, "axes.titlesize": 4.1, "axes.labelsize": 3.5,
        "legend.fontsize": 3.5, "axes.linewidth": 0.3, "axes.labelpad": 1.2, "axes.titlepad": 2.0,
        "legend.handletextpad": 0.2, "legend.labelspacing": 0.35}
TAB10 = plt.get_cmap("tab10").colors


def save(fig, out):
    # Fixed figure size (no tight bbox) so the panel keeps its slot dimensions.
    for ext in ["svg", "pdf", "png"]:
        fig.savefig(f"{out}.{ext}", dpi=600, facecolor="white")
    plt.close(fig)
    print("saved", out)


def points_per_unit(ax, fig):
    width_pt = ax.get_position().width * fig.get_size_inches()[0] * 72
    return width_pt / np.ptp(ax.get_xlim())


def plot_a(domains, categories, out, figsize=None, rc=None):
    """domains: [(labels, spatial, umap, side, title_suffix), ...].

    Domain names label rows; spatial/UMAP titles label columns. The legacy
    suffix is accepted for callers but no longer repeated on each target plot.
    """
    with plt.rc_context(RC_A if rc is None else rc):
        fig = plt.figure(figsize=PANEL_SIZE_IN["a"] if figsize is None else figsize)
        gs = fig.add_gridspec(2, 2, left=0.105, right=0.915, bottom=0.04, top=0.94,
                              wspace=0.1, hspace=0.2)
        spots = np.unique(domains[0][1], axis=0)  # RNA and ATAC profiles share spot coordinates
        spacing = np.median(cKDTree(spots).query(spots, k=2)[0][:, 1])
        for row, (labels, spatial, umap, side, suffix) in enumerate(domains):
            colors = np.array([TAB10[int(c)] for c in labels])
            for col, (xy, view) in enumerate([(spatial, "spatial"), (umap, "UMAP")]):
                ax = fig.add_subplot(gs[row, col])
                pad = 0.02 * np.ptp(xy, axis=0)
                ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0])
                ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
                if view == "spatial":  # marker diameter slightly above the spot pitch: no gaps
                    size = (1.4 * spacing * points_per_unit(ax, fig)) ** 2
                else:
                    size = 1.0
                ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=size, linewidths=0, rasterized=True)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel(f"{view}1"); ax.set_ylabel(f"{view}2")
                ax.set_title(("Spatial" if col == 0 else "UMAP") if row == 0 else "")
                if col == 0:
                    ax.text(-0.20, 0.5, side, transform=ax.transAxes,
                            rotation=90, ha="center", va="center",
                            fontsize=plt.rcParams["axes.titlesize"])
                if view == "UMAP":
                    handles = [Line2D([], [], color=TAB10[int(c)], marker="o", ls="", markersize=2.0, label=c)
                               for c in categories]
                    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5))
        save(fig, out)


def rank_genes_df(rna, labels, categories):
    a = ad.AnnData(X=rna["X"], var=pd.DataFrame(index=rna["var"].index))
    a.obs["leiden"] = pd.Categorical(labels, categories=categories)
    a.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(a, groupby="leiden", method="t-test", reference="rest")
    return sc.get.rank_genes_groups_df(a, group=None)


def plot_c(source_rna, target_rna, source_labels, target_labels, categories, out):
    merged = pd.merge(
        rank_genes_df(source_rna, source_labels, categories).drop(columns=["scores", "logfoldchanges", "pvals"]),
        rank_genes_df(target_rna, target_labels, categories).drop(columns=["scores", "logfoldchanges", "pvals"]),
        on=["group", "names"], how="outer", suffixes=("_source", "_target"))
    merged = merged[merged["pvals_adj_source"].le(0.05) & merged["pvals_adj_target"].le(0.05)].copy()
    corr = merged.groupby("group")[["pvals_adj_source", "pvals_adj_target"]].corr(method="spearman")
    corr = corr.groupby(level=0).apply(lambda df: df.iloc[0, 1])
    print("Spearman per leiden:", corr.round(3).to_dict(), "n genes:", len(merged))
    ranked = merged[["pvals_adj_source", "pvals_adj_target"]].apply(lambda x: -np.log(x)).rank()
    ranked["leiden"] = merged["group"].astype(str)
    order = [str(c) for c in categories]

    with plt.rc_context(RC):
        fig = plt.figure(figsize=PANEL_SIZE_IN["c"])
        gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1], left=0.2, right=0.8, bottom=0.11,
                              top=0.985, hspace=0.42)
        ax = fig.add_subplot(gs[0])
        ax.hexbin(ranked.pvals_adj_source, ranked.pvals_adj_target, gridsize=10, mincnt=3,
                  cmap="Greys", alpha=0.5, linewidths=0.3)
        for leiden in order:
            df = ranked[ranked.leiden == leiden]
            fit = lowess(df.pvals_adj_target, df.pvals_adj_source)
            ax.plot(fit[:, 0], fit[:, 1], color=TAB10[int(leiden)], lw=1.1)
        legend = ax.legend(handles=[Line2D([0], [0], color=TAB10[int(l)], lw=1.1, label=l) for l in order],
                  title="leiden", loc="center left", bbox_to_anchor=(1.0, 0.5))
        ax.set(xlabel="source gene rank", ylabel="target gene rank")
        ax.spines[["top", "right"]].set_visible(False)

        ax = fig.add_subplot(gs[1])
        x = np.arange(len(order)); y = corr.loc[order].to_numpy()
        ax.axhline(0, color="tab:red", lw=0.6)
        ax.vlines(x, 0, y, color="grey", lw=0.6)
        ax.scatter(x, y, s=12, c=[TAB10[int(l)] for l in order], edgecolors="black", linewidths=0.5, zorder=3)
        ax.set_xticks(x, order)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylim(min(0, y.min()) - 0.03, y.max() + 0.06)
        ax.set(xlabel="leiden", ylabel="Spearman corr.")
        # Stem plot spans to the right edge of the rank plot's legend (no whitespace beside it).
        fig.canvas.draw()
        right = legend.get_window_extent().transformed(fig.transFigure.inverted()).x1
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0, right - pos.x0, pos.height])
        save(fig, out)

    # analyze_rna_atac.py checks these markers are top genes of some Leiden cluster (Fig. 4b).
    merged["combined_pvals"] = merged.apply(lambda r: combine_pvalues(
        [r["pvals_adj_source"], r["pvals_adj_target"]], method="pearson")[1], axis=1)
    top = merged.groupby("group").apply(lambda df: df.nsmallest(1, "combined_pvals", keep="all"))["names"].unique()
    print("Fig. 4b markers that are not a top Leiden gene:",
          [g for g in ["Pde10a", "Rgs9", "Gng7", "Mobp", "Mal"] if g not in top])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(f"{os.environ['SPATIALJEPA_HOME']}/data/aligned_data"))
    parser.add_argument("--labels", type=Path, default=Path(
        f"{os.environ['SPATIALJEPA_HOME']}/outputs/rna_atac_transfer_clusters/shared_params"))
    parser.add_argument("--name", default="shared_k150_r0p9")
    args = parser.parse_args()
    keys = ["X_umap", "spatial"]
    d = {f"{s}_{m}": load(args.data / f"{s}_{m}_aligned_with_latents.h5ad", keys, expression=m == "rna")
         for s in ["source", "target"] for m in ["rna", "atac"]}
    src, tgt = [d["source_rna"], d["source_atac"]], [d["target_rna"], d["target_atac"]]
    source_names = np.concatenate([x["obs"].index.to_numpy() + s for x, s in zip(src, ["_rna", "_atac"])])
    mapping = np.concatenate([x["obs"].source_obs_names.astype(str).to_numpy() for x in tgt])
    positions = pd.Index(source_names).get_indexer(mapping)

    s_labels = pd.read_csv(args.labels / f"{args.name}_source_labels.csv", dtype=str)["leiden"].to_numpy()
    t_labels = pd.read_csv(args.labels / f"{args.name}_target_labels.csv", dtype=str)["leiden"].to_numpy()
    _, pairs = match_clusters(s_labels[positions], t_labels)
    assert len(pairs) == len(np.unique(t_labels)) == len(np.unique(s_labels)), "unmatched clusters"
    t_labels = np.array([pairs[c] for c in t_labels])
    categories = sorted(np.unique(s_labels), key=int)

    cat = lambda pair, key: np.concatenate([x["obsm"][key] for x in pair])
    plot_a([(s_labels, cat(src, "spatial"), cat(src, "X_umap"), "Source", ""),
            (t_labels, cat(tgt, "spatial"), cat(tgt, "X_umap"), "Target", ", aligned to source")],
           categories, args.labels / f"fig4a_{args.name}_manuscript")
    plot_c(src[0], tgt[0], s_labels[:len(src[0]["obs"])], t_labels[:len(tgt[0]["obs"])], categories,
           args.labels / f"fig4c_{args.name}_manuscript")


if __name__ == "__main__":
    main()

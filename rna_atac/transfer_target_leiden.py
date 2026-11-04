"""Recreate SpatialJEPA Fig. 4a,c with target Leiden clusters fitted on target embeddings.

Published Fig. 4a/c (MultiGATE scripts/analyze_rna_atac.py, run_alignment_and_spatial_plot
with model = source_mgate) clusters the source and overwrites each target label with the label
of the target profile's nearest source profile. Here the target is clustered on its own
embeddings, and those clusters are compared with the transferred partition.

Inputs are the May 7 *_aligned_with_latents.h5ad files. Source labels are not stored there, so
they are recovered from the target files: every target profile stores its assigned source
profile (source_obs_names) and the label it inherited from it. Source profiles that were never
assigned (1.5%) take the majority label of their 15 nearest assigned source profiles.

Target clusters are paired with source clusters by maximum overlap with the transferred labels
(Hungarian matching on the contingency table), never by expression.
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
import json

import h5py
import numpy as np
import pandas as pd
import anndata as ad
from anndata.io import read_elem
import scanpy as sc
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TAB10 = plt.get_cmap("tab10").colors  # palette of the published panels
EXTRA = [plt.get_cmap("tab20").colors[i] for i in (1, 3, 5, 7, 9, 11, 13, 15, 17, 19)]


def log(*args):
    print(*args, flush=True)


def load(path, keys, expression=False):
    with h5py.File(path) as f:
        return dict(obs=read_elem(f["obs"]), var=read_elem(f["var"]),
                    obsm={k: f["obsm"][k][:] for k in keys},
                    X=f["X"][:] if expression else None)


def recover_source_labels(source_names, mapping, transferred, z_source, k=15):
    lookup = pd.DataFrame({"source": mapping, "label": transferred}).drop_duplicates()
    assert lookup.source.is_unique, "a source profile carries more than one transferred label"
    labels = pd.Series(lookup.label.to_numpy(), index=lookup.source).reindex(source_names)
    covered = labels.notna().to_numpy()
    nn = NearestNeighbors(n_neighbors=k).fit(z_source[covered])
    _, idx = nn.kneighbors(z_source[~covered])
    votes = labels.to_numpy()[covered][idx]
    labels[~covered] = [pd.Series(v).mode().iloc[0] for v in votes]
    return labels.to_numpy().astype(str), covered


def leiden(z, neighbors, path, resolution=0.5, seed=0):
    """Same calls as run_alignment_and_spatial_plot / compute_concat_umap."""
    if path.exists():
        return pd.read_csv(path, dtype=str)["leiden"].to_numpy()
    a = ad.AnnData(X=z)
    sc.pp.neighbors(a, n_neighbors=neighbors, use_rep="X", knn=True, method="umap",
                    metric="euclidean", random_state=seed)
    sc.tl.leiden(a, resolution=resolution, random_state=seed)
    labels = a.obs["leiden"].astype(str).to_numpy()
    pd.DataFrame({"leiden": labels}).to_csv(path, index=False)
    return labels


def sweep(z, ks, resolutions, transferred, path, seed=0):
    """Leiden cluster counts over a k x resolution grid; one neighbour graph per k."""
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for k in ks:
        a = ad.AnnData(X=z)
        sc.pp.neighbors(a, n_neighbors=k, use_rep="X", knn=True, method="umap",
                        metric="euclidean", random_state=seed)
        for r in resolutions:
            sc.tl.leiden(a, resolution=r, random_state=seed, key_added="leiden")
            labels = a.obs["leiden"].astype(str).to_numpy()
            rows.append(dict(k=k, resolution=r, n_clusters=len(np.unique(labels)),
                             ari_vs_transferred=adjusted_rand_score(transferred, labels)))
            log(rows[-1])
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def match_clusters(transferred, labels):
    ct = pd.crosstab(pd.Series(transferred, name="transferred"), pd.Series(labels, name="independent"))
    ct = ct.reindex(index=sorted(ct.index, key=int), columns=sorted(ct.columns, key=int))
    r, c = linear_sum_assignment(-ct.to_numpy())
    return ct, {str(ct.columns[j]): str(ct.index[i]) for i, j in zip(r, c)}


def rank_genes(x, labels, genes):
    """sc.tl.rank_genes_groups exactly as in the published script (t-test vs rest)."""
    a = ad.AnnData(X=x, var=pd.DataFrame(index=genes))
    a.obs["leiden"] = pd.Categorical(labels, categories=sorted(np.unique(labels), key=int))
    a.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(a, groupby="leiden", method="t-test", reference="rest")
    return sc.get.rank_genes_groups_df(a, group=None)[["group", "names", "pvals_adj"]]


def concordance(source_de, target_de, pairs):
    """Published Fig. 4c statistics; target groups are renamed to their matched source IDs."""
    t = target_de[target_de.group.astype(str).isin(pairs)].copy()
    t["target_group"] = t.group.astype(str)
    t["group"] = t.target_group.map(pairs)
    s = source_de.assign(group=source_de.group.astype(str))
    m = s.merge(t, on=["group", "names"], suffixes=("_source", "_target"))
    m = m[m.pvals_adj_source.le(.05) & m.pvals_adj_target.le(.05)].copy()
    rho = m.groupby("group")[["pvals_adj_source", "pvals_adj_target"]].corr(method="spearman")
    rho = rho.groupby(level=0).apply(lambda df: df.iloc[0, 1])
    ranked = m[["pvals_adj_source", "pvals_adj_target"]].apply(lambda v: -np.log(v)).rank()
    ranked["group"] = m.group
    summary = pd.DataFrame({"source_cluster": rho.index, "spearman_r": rho.values})
    summary["target_cluster"] = summary.source_cluster.map({v: k for k, v in pairs.items()})
    summary["n_shared_significant"] = summary.source_cluster.map(m.group.value_counts())
    summary = summary.sort_values("source_cluster", key=lambda c: c.astype(int)).reset_index(drop=True)
    return summary, ranked, m


def all_pairs(source_de, target_de):
    """Spearman for every source x target cluster pair, independent of the overlap matching."""
    s = {g: d.set_index("names").pvals_adj for g, d in source_de.groupby(source_de.group.astype(str))}
    t = {g: d.set_index("names").pvals_adj for g, d in target_de.groupby(target_de.group.astype(str))}
    out = pd.DataFrame(index=sorted(s, key=int), columns=sorted(t, key=int), dtype=float)
    for i, a in s.items():
        for j, b in t.items():
            ab = pd.concat([a, b], axis=1, join="inner").dropna()
            ab = ab[(ab.iloc[:, 0] <= .05) & (ab.iloc[:, 1] <= .05)]
            out.loc[i, j] = spearmanr(ab.iloc[:, 0], ab.iloc[:, 1]).statistic if len(ab) > 2 else np.nan
    return out


def target_palette(labels, pairs):
    names, colors, extra = {}, {}, iter(EXTRA)
    for c in sorted(np.unique(labels), key=int):
        if c in pairs:
            names[c], colors[c] = f"T{c} (S{pairs[c]})", TAB10[int(pairs[c])]
        else:
            names[c], colors[c] = f"T{c} (unmatched)", next(extra)
    return names, colors


def save(fig, path):
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_a(source_xy, target_xy, source_labels, target_labels, target_names, target_colors, title, path):
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    rows = [(source_xy, source_labels, {c: TAB10[int(c)] for c in np.unique(source_labels)},
             {c: c for c in np.unique(source_labels)}, "Source"),
            (target_xy, target_labels, target_colors, target_names, "Target")]
    for r, (xy, labels, colors, names, side) in enumerate(rows):
        for col, (key, view, size) in enumerate([("spatial", "spatial", 6), ("X_umap", "UMAP", 3)]):
            ax = axs[r, col]
            ax.scatter(xy[key][:, 0], xy[key][:, 1], c=[colors[c] for c in labels], s=size,
                       linewidths=0, rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(f"{view}1"); ax.set_ylabel(f"{view}2")
            ax.set_title(f"{side} {view}" if side == "Source" else f"{side} {view}, {title}", fontsize=10)
        handles = [Line2D([], [], color=colors[c], marker="o", ls="", label=names[c])
                   for c in sorted(np.unique(labels), key=int)]
        axs[r, 1].legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, fontsize=8)
    fig.tight_layout()
    save(fig, path)


def plot_c(summary, ranked, title, path):
    fig, axs = plt.subplots(2, 1, figsize=(6, 7), gridspec_kw={"height_ratios": [2.2, 1]})
    axs[0].hexbin(ranked.pvals_adj_source, ranked.pvals_adj_target, gridsize=10, mincnt=3,
                  cmap="Greys", alpha=.5)
    handles = []
    for row in summary.itertuples():
        d = ranked[ranked.group == row.source_cluster]
        fit = lowess(d.pvals_adj_target, d.pvals_adj_source)
        color = TAB10[int(row.source_cluster)]
        axs[0].plot(fit[:, 0], fit[:, 1], color=color, lw=3)
        label = row.source_cluster if row.target_cluster == row.source_cluster and title.startswith("transferred") \
            else f"S{row.source_cluster}–T{row.target_cluster}"
        handles.append(Line2D([0], [0], color=color, lw=3, label=label))
    axs[0].legend(handles=handles, title="leiden", loc="center left", bbox_to_anchor=(1.02, .5), frameon=False)
    axs[0].set(xlabel="source gene rank", ylabel="target gene rank", title=title)
    x = np.arange(len(summary))
    axs[1].axhline(0, color="tab:red", lw=1)
    axs[1].vlines(x, 0, summary.spearman_r, color="grey")
    axs[1].scatter(x, summary.spearman_r, s=80, c=[TAB10[int(c)] for c in summary.source_cluster],
                   edgecolors="black", linewidths=1.5, zorder=3)
    axs[1].set_xticks(x, summary.source_cluster if title.startswith("transferred") else
                      [f"S{r.source_cluster}\nT{r.target_cluster}" for r in summary.itertuples()])
    axs[1].set(ylabel="Spearman corr.", ylim=(min(0, summary.spearman_r.min()) - .03, .55))
    fig.tight_layout()
    save(fig, path)


def plot_contingency(ct, pairs, title, path):
    frac = ct.div(ct.sum(0), axis=1)  # columns: independent target clusters
    fig, ax = plt.subplots(figsize=(1 + .55 * ct.shape[1], 4))
    im = ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            ax.text(j, i, f"{frac.iat[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=7,
                    color="white" if frac.iat[i, j] > .6 else "black",
                    fontweight="bold" if pairs.get(str(frac.columns[j])) == str(frac.index[i]) else "normal")
    ax.set_xticks(range(frac.shape[1]), [f"T{c}" for c in frac.columns])
    ax.set_yticks(range(frac.shape[0]), [f"S{c}" for c in frac.index])
    ax.set(xlabel="Independent target cluster", ylabel="Transferred (source) label", title=title)
    fig.colorbar(im, ax=ax, label="Fraction of target cluster")
    fig.tight_layout()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(f"{os.environ['SPATIALJEPA_HOME']}/data/aligned_data"))
    parser.add_argument("--out", type=Path, default=Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/rna_atac_transfer_clusters"))
    args = parser.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=True)
    keys = ["MultiGATE", "MultiGATE_source_aligned", "X_umap", "spatial"]
    d = {f"{s}_{m}": load(args.data / f"{s}_{m}_aligned_with_latents.h5ad", keys, expression=m == "rna")
         for s in ["source", "target"] for m in ["rna", "atac"]}
    src, tgt = [d["source_rna"], d["source_atac"]], [d["target_rna"], d["target_atac"]]
    assert src[0]["var"].index.equals(tgt[0]["var"].index)
    genes = src[0]["var"].index.to_numpy()
    nsrc, ntgt = len(src[0]["obs"]), len(tgt[0]["obs"])
    cat = lambda pair, key: np.concatenate([x["obsm"][key] for x in pair])
    source_xy = {k: cat(src, k) for k in ["spatial", "X_umap"]}
    target_xy = {k: cat(tgt, k) for k in ["spatial", "X_umap"]}
    z_source, z_ot, z_raw = cat(src, "MultiGATE"), cat(tgt, "MultiGATE_source_aligned"), cat(tgt, "MultiGATE")
    source_names = np.concatenate([x["obs"].index.to_numpy() + s for x, s in zip(src, ["_rna", "_atac"])])
    transferred = np.concatenate([x["obs"].leiden.astype(str).to_numpy() for x in tgt])
    mapping = np.concatenate([x["obs"].source_obs_names.astype(str).to_numpy() for x in tgt])

    source_labels, covered = recover_source_labels(source_names, mapping, transferred, z_source)
    pd.DataFrame({"obs": source_names, "leiden": source_labels, "recovered_exactly": covered}).to_csv(
        out / "source_labels.csv", index=False)
    nn_dist = NearestNeighbors(n_neighbors=1).fit(z_source).kneighbors(z_ot)[0][:, 0]
    src_dist = NearestNeighbors(n_neighbors=2).fit(z_source).kneighbors(z_source)[0][:, 1]
    provenance = dict(scanpy=sc.__version__, anndata=ad.__version__, n_source_profiles=len(source_names),
                      n_target_profiles=len(transferred), n_genes=len(genes),
                      source_labels_recovered_exactly=float(covered.mean()),
                      median_ot_target_to_nearest_source=float(np.median(nn_dist)),
                      median_source_to_nearest_source=float(np.median(src_dist)))
    log(json.dumps(provenance, indent=2))

    source_de = rank_genes(src[0]["X"], source_labels[:nsrc], genes)
    analyses = [("transferred", None, None, "source labels transferred (published)")]
    analyses += [(f"independent_{e}_k{k}", z, k, f"independent Leiden ({lbl}, k={k})")
                 for e, z, lbl in [("preot", z_raw, "before OT"), ("ot", z_ot, "after OT")] for k in (100, 10)]
    analyses = [(name, z, k, .5, title) for name, z, k, title in analyses]
    # Source Leiden used k=100, resolution 0.5. For k=100 and k=150, take the smallest resolution
    # increase that gives the target (after OT) as many clusters as the source.
    n_source = len(np.unique(source_labels))
    grid = sweep(z_ot, [50, 100, 150], np.round(np.arange(.5, 1.01, .05), 2), transferred,
                 out / "ot_resolution_sweep.csv")
    for k in (100, 150):
        hits = grid[(grid.k == k) & (grid.n_clusters == n_source) & (grid.resolution > .5)]
        if len(hits):
            r = float(hits.resolution.min())
            analyses.append((f"independent_ot_k{k}_r{r:.2f}".replace(".", "p"), z_ot, k, r,
                             f"independent Leiden (after OT, k={k}, res={r:.2f})"))
    summaries, metrics = [], []
    for name, z, k, res, title in analyses:
        labels = transferred if z is None else leiden(z, k, out / f"{name}_labels.csv", resolution=res)
        ct, pairs = match_clusters(transferred, labels)
        if z is None:
            pairs = {c: c for c in np.unique(labels)}
        names, colors = target_palette(labels, pairs)
        if z is None:
            names = {c: c for c in names}
        matched = sum(ct.loc[s, t] for t, s in pairs.items())
        metrics.append(dict(analysis=name, n_clusters=ct.shape[1], n_matched=len(pairs),
                            ari_vs_transferred=adjusted_rand_score(transferred, labels),
                            nmi_vs_transferred=normalized_mutual_info_score(transferred, labels),
                            fraction_in_matched_cluster=matched / len(labels),
                            cluster_sizes=json.dumps({c: int(n) for c, n in ct.sum(0).items()})))
        target_de = rank_genes(tgt[0]["X"], labels[:ntgt], genes)
        summary, ranked, merged = concordance(source_de, target_de, pairs)
        summary.insert(0, "analysis", name)
        summaries.append(summary)
        log(f"\n== {name}: {metrics[-1]}\n{summary.to_string(index=False)}")
        ct.to_csv(out / f"{name}_contingency.csv")
        merged.to_csv(out / f"{name}_shared_significant_genes.csv", index=False)
        all_pairs(source_de, target_de).to_csv(out / f"{name}_all_pairs_spearman.csv")
        plot_a(source_xy, target_xy, source_labels, labels, names, colors, title, out / f"fig4a_{name}")
        plot_c(summary, ranked, title, out / f"fig4c_{name}")
        if z is not None:
            plot_contingency(ct, pairs, title, out / f"contingency_{name}")
    pd.DataFrame(metrics).to_csv(out / "clustering_comparison.csv", index=False)
    results = pd.concat(summaries, ignore_index=True)
    results.to_csv(out / "concordance_comparison.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    for i, (name, g) in enumerate(results.groupby("analysis", sort=False)):
        ax.plot(g.source_cluster.astype(int) + (i - 2) * .08, g.spearman_r, "o", label=name,
                color="black" if name == "transferred" else plt.get_cmap("Set2")(i))
    ax.axhline(0, color="grey", lw=.7)
    ax.set(xlabel="Source cluster (target cluster matched by overlap)", ylabel="Spearman corr.")
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, .5))
    fig.tight_layout(); save(fig, out / "fig4c_comparison")
    log("DONE", out.resolve())


if __name__ == "__main__":
    main()

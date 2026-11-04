"""Search Leiden parameters shared by source and OT-aligned target for Fig. 4a/c.

Source and target (after OT) are clustered independently with the same (k, resolution).
Overlap is the ARI between each target profile's label and the label of its assigned source
profile (source_obs_names, the nearest-source mapping used in the published figure). The best
setting with equal cluster counts is then run through the Fig. 4a/c analysis, with the source
row showing the re-clustered (not published) source labels.
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
import sys

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from transfer_target_leiden import (  # noqa: E402
    log, load, match_clusters, rank_genes, concordance, all_pairs, target_palette,
    plot_a, plot_c, plot_contingency, save,
)

KS = [15, 30, 50, 100, 150]
RESOLUTIONS = [.3, .4, .5, .6, .7, .8, .9, 1.]


def cluster_grid(z, path, seed=0):
    """Leiden labels for every (k, resolution); one neighbour graph per k."""
    if path.exists():
        return pd.read_csv(path, dtype=str)
    cols = {}
    for k in KS:
        a = ad.AnnData(X=z)
        sc.pp.neighbors(a, n_neighbors=k, use_rep="X", knn=True, method="umap",
                        metric="euclidean", random_state=seed)
        for r in RESOLUTIONS:
            sc.tl.leiden(a, resolution=r, random_state=seed, key_added="leiden")
            cols[f"k{k}_r{r:.1f}"] = a.obs["leiden"].astype(str).to_numpy()
            log(path.name, k, r, len(np.unique(cols[f"k{k}_r{r:.1f}"])))
    df = pd.DataFrame(cols)
    df.to_csv(path, index=False)
    return df


def plot_grid(grid, path):
    ari = grid.pivot(index="k", columns="resolution", values="ari_shared")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(ari.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i, k in enumerate(ari.index):
        for j, r in enumerate(ari.columns):
            row = grid[(grid.k == k) & (grid.resolution == r)].iloc[0]
            ax.text(j, i, f"{row.ari_shared:.2f}\n{row.n_source}/{row.n_target}", ha="center",
                    va="center", fontsize=7, color="white" if row.ari_shared < .6 else "black",
                    fontweight="bold" if row.n_source == row.n_target else "normal")
    ax.set_xticks(range(len(ari.columns)), [f"{r:.1f}" for r in ari.columns])
    ax.set_yticks(range(len(ari.index)), ari.index)
    ax.set(xlabel="Leiden resolution", ylabel="k neighbours",
           title="Source vs target (after OT) Leiden overlap, shared parameters\n"
                 "ARI; source/target cluster counts (bold: equal)")
    fig.colorbar(im, ax=ax, label="ARI")
    fig.tight_layout()
    save(fig, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(f"{os.environ['SPATIALJEPA_HOME']}/data/aligned_data"))
    parser.add_argument("--out", type=Path, default=Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/rna_atac_transfer_clusters"))
    args = parser.parse_args()
    out = args.out / "shared_params"; out.mkdir(parents=True, exist_ok=True)
    keys = ["MultiGATE", "MultiGATE_source_aligned", "X_umap", "spatial"]
    d = {f"{s}_{m}": load(args.data / f"{s}_{m}_aligned_with_latents.h5ad", keys, expression=m == "rna")
         for s in ["source", "target"] for m in ["rna", "atac"]}
    src, tgt = [d["source_rna"], d["source_atac"]], [d["target_rna"], d["target_atac"]]
    genes = src[0]["var"].index.to_numpy()
    nsrc, ntgt = len(src[0]["obs"]), len(tgt[0]["obs"])
    cat = lambda pair, key: np.concatenate([x["obsm"][key] for x in pair])
    source_xy = {k: cat(src, k) for k in ["spatial", "X_umap"]}
    target_xy = {k: cat(tgt, k) for k in ["spatial", "X_umap"]}
    source_names = np.concatenate([x["obs"].index.to_numpy() + s for x, s in zip(src, ["_rna", "_atac"])])
    mapping = np.concatenate([x["obs"].source_obs_names.astype(str).to_numpy() for x in tgt])
    positions = pd.Index(source_names).get_indexer(mapping)
    assert (positions >= 0).all()
    published = pd.read_csv(args.out / "source_labels.csv", dtype=str)["leiden"].to_numpy()

    source_grid = cluster_grid(cat(src, "MultiGATE"), out / "source_grid_labels.csv")
    target_grid = cluster_grid(cat(tgt, "MultiGATE_source_aligned"), out / "target_grid_labels.csv")
    rows = []
    for col in source_grid.columns:
        s, t = source_grid[col].to_numpy(), target_grid[col].to_numpy()
        k, r = col[1:].split("_r")
        rows.append(dict(k=int(k), resolution=float(r), n_source=len(np.unique(s)), n_target=len(np.unique(t)),
                         ari_shared=adjusted_rand_score(s[positions], t),
                         source_ari_vs_published=adjusted_rand_score(published, s)))
    grid = pd.DataFrame(rows)
    grid.to_csv(out / "shared_params_grid.csv", index=False)
    plot_grid(grid, out / "shared_params_grid")
    log(grid.round(3).to_string(index=False))

    # Equal cluster counts, within the 10-colour palette of the published panels.
    ok = grid[(grid.n_source == grid.n_target) & grid.n_source.between(6, 10)]
    best = ok.sort_values("ari_shared", ascending=False).iloc[0]
    col = f"k{int(best.k)}_r{best.resolution:.1f}"
    name = f"shared_{col.replace('.', 'p')}"
    title = f"independent Leiden, shared k={int(best.k)}, res={best.resolution:.1f}"
    s_labels, t_labels = source_grid[col].to_numpy(), target_grid[col].to_numpy()
    mapped = s_labels[positions]
    ct, pairs = match_clusters(mapped, t_labels)
    names, colors = target_palette(t_labels, pairs)
    source_de = rank_genes(src[0]["X"], s_labels[:nsrc], genes)
    target_de = rank_genes(tgt[0]["X"], t_labels[:ntgt], genes)
    summary, ranked, merged = concordance(source_de, target_de, pairs)
    summary.insert(0, "analysis", name)
    summary.to_csv(out / f"{name}_concordance.csv", index=False)
    merged.to_csv(out / f"{name}_shared_significant_genes.csv", index=False)
    ct.to_csv(out / f"{name}_contingency.csv")
    all_pairs(source_de, target_de).to_csv(out / f"{name}_all_pairs_spearman.csv")
    pd.DataFrame({"source_obs": source_names, "leiden": s_labels}).to_csv(out / f"{name}_source_labels.csv", index=False)
    pd.DataFrame({"leiden": t_labels}).to_csv(out / f"{name}_target_labels.csv", index=False)
    pc = pd.crosstab(pd.Series(published, name="published"), pd.Series(s_labels, name="reclustered"))
    pc.to_csv(out / f"{name}_source_vs_published_contingency.csv")
    plot_a(source_xy, target_xy, s_labels, t_labels, names, colors, title, out / f"fig4a_{name}")
    plot_c(summary, ranked, title, out / f"fig4c_{name}")
    plot_contingency(ct, pairs, title, out / f"contingency_{name}")
    selected = dict(best.to_dict(), name=name, fraction_in_matched_cluster=float(
        sum(ct.loc[s, t] for t, s in pairs.items()) / len(t_labels)), mean_spearman=float(summary.spearman_r.mean()))
    (out / "selected.json").write_text(json.dumps(selected, indent=2))
    log(f"\nSELECTED {json.dumps(selected)}\n{summary.to_string(index=False)}")
    log("DONE", out.resolve())


if __name__ == "__main__":
    main()

"""Supplementary Fig. 3: position recovery on the RNA-side, ATAC-side and joint latents.

Main Fig. 3 scores every model on its RNA-side latent (MultiGATE RNA encoder,
MultiVI X_multivi_rna; analyze_rna_atac.py passes source_rna and the MultiVI
RNA modality). This repeats the same estimator on the ATAC-side latents
(MultiGATE ATAC encoder, MultiVI X_multivi_atac) and averages the per-spot ECDF
areas within the clusters of the matching modality: RNA latents within
RNA_clusters, ATAC latents within ATAC_clusters. It also scores the joint
latents: MultiGATE's own joint embedding (the "_clip_all" of Train_MultiGATE.py,
the mean of the L2-normalised RNA and ATAC embeddings), computed here for the
teacher, student and non-spatial ablation, and MultiVI's joint X_multivi; these
are averaged within RNA_clusters. The contiguity scatter (former Supplementary
Fig. S3b) is redrawn from the same statistics.

compute_spatial_structure_stats is loaded from analyze_rna_atac.py without
running that script. The teacher is the full-graph teacher latent saved with
the 2026-05-05 co-embed run's metric inputs (obsm MultiGATE_full_teacher).
obsm MultiGATE_teacher in data/aligned_data is not the teacher: analyze_rna_atac.py
sets it by running the student weights (source_mgate) with the full spatial graph
(RNA area 0.86). The RNA row reproduces the published areas, which the run
checks. All rows share one xmax, so their areas are on one scale.

    python plot_position_recovery_by_latent.py [--output DIR] [--draw-only]
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
import ast
import json
import pickle
import warnings

import h5py
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(os.environ['SPATIALJEPA_HOME'])
DATA = BASE / "data/aligned_data"
ORIGINAL = Path(os.environ["SPATIALJEPA_RNA_ATAC"]) / "analyze_rna_atac.py"
TEACHER = BASE / "outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap/metric_inputs"
DEFAULT_OUT = BASE / "outputs/rna_atac_figures/fig3_supp_ecdf_by_latent"
ARMS = ["teacher", "student", "non-spatial", "MultiVI"]
LABEL = {"teacher": "SpatialJEPA teacher", "student": "SpatialJEPA student",
         "non-spatial": "MultiGATE non-spatial", "MultiVI": "MultiVI"}
MODALITIES = ("rna", "atac", "joint")
CLUSTERS = {"rna": "RNA_clusters", "atac": "ATAC_clusters", "joint": "RNA_clusters"}
PUBLISHED_RNA = {"teacher": 0.75, "student": 0.57, "non-spatial": 0.26, "MultiVI": 0.33}


def original_functions():
    """compute_spatial_structure_stats and the constant it reads, without the pipeline."""
    tree = ast.parse(ORIGINAL.read_text())
    keep = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == "compute_spatial_structure_stats")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SPATIAL_STRUCTURE_ARMS"
                                                  for t in n.targets))]
    assert len(keep) == 2
    ns = {"np": np, "pd": pd, "warnings": warnings}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(ORIGINAL), "exec"), ns)
    return ns["compute_spatial_structure_stats"]


def read_obsm(path, keys):
    with h5py.File(path, "r") as h:
        idx = h["obs"][h["obs"].attrs["_index"]][:].astype(str)
        return idx, {k: h["obsm"][k][:] for k in keys}


def read_labels(path, cols):
    with h5py.File(path, "r") as h:
        out = {}
        for c in cols:
            g = h["obs"][c]
            out[c] = np.asarray(g["categories"][:].astype(str))[g["codes"][:]]
        return out


def l2(Z):
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def build(modality):
    """AnnData pair shaped like (source_rna, source_multivi_rna) for one modality, or 'joint'."""
    if modality == "joint":
        (names, R), (_, A) = (latents(m) for m in ("rna", "atac"))
        Z = {k: (l2(R[k]) + l2(A[k])) / 2.0 for k in ("MultiGATE", "MultiGATE_nonspatial", "MultiGATE_teacher")}
        Z["spatial"] = R["spatial"]
        _, stored = read_obsm(DATA / "source_rna_aligned_with_latents.h5ad", ["MultiGATE_clip_all"])
        assert np.allclose(Z["MultiGATE"], stored["MultiGATE_clip_all"], atol=1e-5)    # MultiGATE's own joint
    else:
        names, Z = latents(modality)
    rna_names, rna = read_obsm(DATA / "source_rna_aligned_with_latents.h5ad", ["X_pca"])
    assert (names == rna_names).all()
    labels = read_labels(DATA / "source_rna_aligned_with_latents.h5ad", ["RNA_clusters", "ATAC_clusters"])
    src = ad.AnnData(obs=pd.DataFrame(labels, index=names))
    src.obsm["MultiGATE_full_teacher"] = Z["MultiGATE_teacher"]
    src.obsm["MultiGATE"] = Z["MultiGATE"]
    src.obsm["MultiGATE_nonspatial"] = Z["MultiGATE_nonspatial"]
    src.obsm["spatial"] = Z["spatial"]
    src.obsm["X_pca"] = rna["X_pca"]          # molecular baseline: RNA PCs in both rows (printed only)
    with h5py.File(DATA / "multivi_mdata.h5mu", "r") as h:
        g = h if modality == "joint" else h["mod"][modality]
        mv_names = g["obs"][g["obs"].attrs["_index"]][:].astype(str)
        mv = ad.AnnData(obs=pd.DataFrame(index=mv_names))
        mv.obsm["X_multivi"] = g["obsm"]["X_multivi" if modality == "joint" else f"X_multivi_{modality}"][:]
    return src, mv


def latents(modality):
    names, Z = read_obsm(DATA / f"source_{modality}_aligned_with_latents.h5ad",
                         ["MultiGATE", "MultiGATE_nonspatial", "spatial"])
    t_names, T = read_obsm(TEACHER / f"teacher_source__{modality}.h5ad", ["MultiGATE_full_teacher"])
    assert (pd.Index(t_names).get_indexer(names) >= 0).all()
    Z["MultiGATE_teacher"] = T["MultiGATE_full_teacher"][pd.Index(t_names).get_indexer(names)]
    return names, Z


def areas(errors, xmax):
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {a: np.nanmean(1.0 - np.log10(np.clip(E, 1.0, xmax)) / np.log10(xmax), axis=1)
                for a, E in errors.items()}


def contiguity_domains(s, cluster_key, k=6):
    """Per-domain contiguity and student/teacher kNN agreement, as compute_spatial_structure_stats does
    for RNA_clusters (6 spatial neighbours; the function's own k-overlap per spot), for any label set."""
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors

    lab = s["labels"][cluster_key].to_numpy()
    nb = NearestNeighbors(n_neighbors=k + 1).fit(s["xy"]).kneighbors(s["xy"], return_distance=False)[:, 1:]
    d = pd.DataFrame({"contiguity": (lab[nb] == lab[:, None]).mean(axis=1),
                      "teacher": s["overlap"]["teacher"], "student": s["overlap"]["student"], "domain": lab})
    d = d.groupby("domain").mean()
    d["distil_gap"] = d["teacher"] - d["student"]
    r = spearmanr(d["contiguity"], d["distil_gap"])
    return dict(domains=d, rho=float(r.statistic), pval=float(r.pvalue))


def compute(out):
    fn = original_functions()
    stats = {}
    for modality in MODALITIES:
        print(f"== {modality} latents", flush=True)
        stats[modality] = fn(*build(modality))
    xmax = max(s["xmax"] for s in stats.values())
    rows, res = [], {}
    for modality, s in stats.items():
        A = areas({a: s["errors"][a] for a in ARMS}, xmax)
        lab = s["labels"][CLUSTERS[modality]]
        by_cl = {a: pd.Series(A[a], index=s["obs_names"]).groupby(lab).mean() for a in ARMS}
        delta = by_cl["student"] - by_cl["MultiVI"]
        res[modality] = dict(area=A, by_cluster=pd.DataFrame(by_cl), delta=delta,
                             contiguity_domains={c: contiguity_domains(s, c) for c in ("RNA_clusters", "ATAC_clusters")},
                             cluster_key=lab.name
                             if hasattr(lab, "name") else None, labels=lab.to_numpy(),
                             floor=s["floor_auc"], domains=s["domains"], rho=s["rho"], pval=s["pval"],
                             hit2={a: float(np.nanmean(s["hit"][2.0][a])) for a in ARMS})
        for a in ARMS:
            rows.append(dict(latent=modality.upper(), arm=a, ecdf_area=float(np.nanmean(A[a])),
                             within_2_spot_widths=res[modality]["hit2"][a],
                             cluster_mean_min=float(by_cl[a].min()), cluster_mean_max=float(by_cl[a].max())))
        print(f"{modality}: student - MultiVI by cluster {delta.min():+.2f} to {delta.max():+.2f}; "
              f"student ahead in {(delta > 0).sum()}/{len(delta)} clusters", flush=True)
    xy, obs_names = stats["rna"]["xy"], stats["rna"]["obs_names"]
    assert all((stats[m]["obs_names"] == obs_names).all() for m in MODALITIES)
    T = pd.DataFrame(rows)
    T.to_csv(out / "ecdf_area_by_latent.csv", index=False)
    pd.concat({m: r["by_cluster"].assign(student_minus_MultiVI=r["delta"]) for m, r in res.items()},
              names=["latent", "cluster"]).to_csv(out / "ecdf_area_by_cluster.csv")
    for m in MODALITIES:
        stats[m]["domains"].to_csv(out / f"domains_{m}_latent.csv")
    with open(out / "stats.pkl", "wb") as h:
        pickle.dump(dict(res=res, xy=xy, obs_names=obs_names, xmax=xmax), h)
    check = {a: round(float(T.query("latent == 'RNA' and arm == @a").ecdf_area.iloc[0]), 2) for a in ARMS}
    json.dump(dict(xmax=xmax, xmax_by_latent={m: s["xmax"] for m, s in stats.items()},
                   rna_matches_published=check == PUBLISHED_RNA, rna_areas=check, published=PUBLISHED_RNA,
                   teacher_key=f"{TEACHER}/teacher_source__<modality>.h5ad obsm MultiGATE_full_teacher",
                   joint="mean of L2-normalised RNA and ATAC latents (MultiGATE _clip_all); MultiVI top-level X_multivi",
                   multivi_keys={"rna": "X_multivi_rna", "atac": "X_multivi_atac", "joint": "X_multivi"}),
              open(out / "provenance.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False))
    print(f"xmax {xmax:.2f}; RNA areas match published Fig. 3: {check == PUBLISHED_RNA} ({check})")


def draw(out):
    """Maps for the RNA-side and ATAC-side latents (main Fig. 3 shows the joint latents); contiguity
    scatters by RNA_clusters (RNA-side and joint latents) and by ATAC_clusters (ATAC-side and joint)."""
    S = pickle.load(open(out / "stats.pkl", "rb"))
    res, xy = S["res"], S["xy"]
    rows = ["rna", "atac"]
    title = {"rna": "RNA-side latents", "atac": "ATAC-side latents", "joint": "joint latents"}
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8})
    W, H = 10.4, 8.9                                    # inches; the bottom row is placed in inches below
    fig = plt.figure(figsize=(W, H))
    inch = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])
    gs = fig.add_gridspec(len(rows), 8, width_ratios=[1, 1, 1, 1, 0.06, 0.28, 1, 0.06],
                          left=0.065, right=0.985, top=0.975, bottom=4.62 / H, hspace=0.2, wspace=0.08)
    seq, div = plt.get_cmap("viridis"), plt.get_cmap("PRGn")
    dmax = max(float(np.abs(res[m]["delta"]).max()) for m in rows)
    for row, m in enumerate(rows):
        r = res[m]
        for c, a in enumerate(ARMS):
            ax = fig.add_subplot(gs[row, c])
            v = r["by_cluster"][a].reindex(r["labels"]).to_numpy()
            sa = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=2.6, lw=0, cmap=seq, vmin=0, vmax=1, rasterized=True)
            ax.set_title(f"{LABEL[a]}\narea {np.nanmean(r['area'][a]):.2f}", fontsize=8)
            ax.set_aspect("equal"); ax.axis("off")
            if c == 0:
                ax.text(-0.14, 1.22, "ab"[row], transform=ax.transAxes, fontsize=13, fontweight="bold")
                ax.text(-0.06, 0.5, f"{title[m]}\n({CLUSTERS[m]})",
                        transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=8)
        cb = fig.colorbar(sa, cax=fig.add_subplot(gs[row, 4])); cb.ax.tick_params(labelsize=6)
        cb.ax.axhline(r["floor"], color="0.25", lw=1.0)
        cb.set_label("cluster-mean ECDF area", fontsize=7)
        ax = fig.add_subplot(gs[row, 6])
        v = r["delta"].reindex(r["labels"]).to_numpy()
        sd = ax.scatter(xy[:, 0], xy[:, 1], c=v, s=2.6, lw=0, cmap=div, vmin=-dmax, vmax=dmax, rasterized=True)
        ax.set_title(f"student $-$ MultiVI\n{r['delta'].min():+.2f} to {r['delta'].max():+.2f}", fontsize=8)
        ax.set_aspect("equal"); ax.axis("off")
        cb = fig.colorbar(sd, cax=fig.add_subplot(gs[row, 7])); cb.ax.tick_params(labelsize=6)
        cb.set_label("$\\Delta$ ECDF area", fontsize=7)
    # (c) the two cluster annotations on the section, as in main Fig. 2a (scanpy's default palette,
    # stored with the MultiVI data, in natural cluster order)
    # c: two stacked maps (with legends) as tall as the d and e squares beside them
    with h5py.File(DATA / "multivi_mdata.h5mu", "r") as h:
        palettes = {k: [c.decode() for c in h["uns"][f"{k}_colors"][:]] for k in ("RNA_clusters", "ATAC_clusters")}
    for k, (m, key, name) in enumerate((("rna", "RNA_clusters", "RNA clusters"), ("atac", "ATAC_clusters", "ATAC clusters"))):
        lab = res[m]["labels"].astype(str)
        cats = sorted(np.unique(lab), key=lambda c: int(c[1:]))
        col = dict(zip(cats, palettes[key]))
        ax = inch(0.15, 2.2 - 2.1 * k, 2.3, 1.9)
        ax.scatter(xy[:, 0], xy[:, 1], c=[col[c] for c in lab], s=4.5, lw=0, rasterized=True)   # spots overlap: no grid
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(name, fontsize=8.5)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.2, color=col[c], label=c) for c in cats],
                  loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5.5, frameon=False,
                  handletextpad=0.1, labelspacing=0.15, borderaxespad=0.2)
        if k == 0:
            fig.text(0.05 / W, 4.3 / H, "c", fontsize=13, fontweight="bold")
    # (d, e) contiguity vs student-minus-teacher kNN agreement; rho is for teacher - student, the axis
    # shows student - teacher, hence the sign flip in the legend
    style = {"rna": ("#1b7837", "o"), "atac": ("#2166ac", "s"), "joint": ("#b35806", "^")}
    for k, (key, latents) in enumerate((("RNA_clusters", ("rna", "joint")), ("ATAC_clusters", ("atac", "joint")))):
        ax = inch(3.7 + 3.7 * k, 0.5, 3.05, 3.05)
        for m in latents:
            cd = res[m]["contiguity_domains"][key]
            d = cd["domains"]
            ax.scatter(d["contiguity"], -d["distil_gap"], s=24, color=style[m][0], marker=style[m][1],
                       label=f"{title[m].replace(' latents', '')}: $r_s$ = {-cd['rho']:+.2f}, p = {cd['pval']:.3f}")
            if m != "joint":
                for cl, rr in d.iterrows():
                    name = str(cl)
                    if name in ("R6", "R10"):          # coincide with R8 at contiguity ~0.02
                        continue
                    ax.annotate("R6, R8, R10" if name == "R8" else name, (rr["contiguity"], -rr["distil_gap"]),
                                fontsize=5.5, xytext=(4, -8 if name == "R8" else 2), textcoords="offset points",
                                color="0.3")
        ax.set_xlabel(f"neighbourhood contiguity ({key})", fontsize=8)
        ax.set_ylabel("student $-$ teacher\nkNN agreement ($k$ = 10)", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.0), borderaxespad=0.1,
                  title="Spearman", title_fontsize=6.5, alignment="left")
        ax.spines[["top", "right"]].set_visible(False)
        fig.text((2.95 + 3.7 * k) / W, 4.3 / H, "de"[k], fontsize=13, fontweight="bold")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / f"S_Fig3_ecdf_by_latent.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}/S_Fig3_ecdf_by_latent.png")
    for m in ("rna", "atac", "joint"):
        for key, cd in res[m]["contiguity_domains"].items():
            print(f"{m:5s} {key:13s} Spearman (student - teacher vs contiguity) {-cd['rho']:+.3f}, p = {cd['pval']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--draw-only", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if not a.draw_only:
        compute(a.output)
    draw(a.output)


if __name__ == "__main__":
    main()

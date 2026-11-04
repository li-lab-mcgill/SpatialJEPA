"""Supplemental figure: held-out Multiome placement scored against the published P22 annotations.

Same pipeline and models as main-text Fig. 2f, but the reference labels change to the published RNA and
ATAC clusters of Zhang et al. (2023). Placed clusters
are each model's own Leiden clusters of its Multiome embedding at K=10, the resolution scored against
the published labels by heldout_multiome_alignment.py. Dashes are in-sample references: each
model's own P22 embedding clustered the same way (K=10) and scored against the same reference.

Usage:
    python plot_heldout_vs_published_annotations.py OUT_DIR [--no-insample]

--no-insample draws only the placed-cluster ARIs already computed by heldout_multiome_alignment.py
(no clustering is run and no in-sample dashes are drawn).
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).parent))
import heldout_multiome_alignment as al  # noqa: E402
from plot_heldout_multiome_placement import MODELS, COUPLINGS, DARK, RC, LEIDEN  # noqa: E402

PANEL_MODELS = ("student", "teacher_nograph", "nonspatial", "multivi")
REFS = {"RNA_clusters": "published RNA clusters", "ATAC_clusters": "published ATAC clusters"}
# The teacher-domain reference is main-text Fig. 2f; it is not repeated here.
CACHE = LEIDEN / "in_sample_vs_published_K10.csv"
TEACHER_COLOUR = "#66c2a5"  # main-text Fig. 2e (Set2): teacher
SOURCE_H5AD = Path(f"{os.environ['SPATIALJEPA_HOME']}/data/aligned_data/source_rna_aligned_with_latents.h5ad")


def teacher_vs_published():
    """ARI of the teacher's own P22 domains (K=10, no placement) with the published annotations,
    and of the two published annotations with each other."""
    import h5py
    from anndata.io import read_elem
    obs = np.load(LEIDEN / "backprojected_maps_student.npz", allow_pickle=True)["p22_obs"].astype(str)
    teacher = np.load(LEIDEN / "teacher_reference.npz", allow_pickle=True)["teacher_K10"].astype(str)
    with h5py.File(SOURCE_H5AD) as f:
        names = np.asarray(read_elem(f["obs"][f["obs"].attrs["_index"]])).astype(str)
        pub = {c: np.asarray(read_elem(f["obs"][c])).astype(str) for c in ("RNA_clusters", "ATAC_clusters")}
    pos = pd.Series(np.arange(len(names)), index=names)[obs].to_numpy()
    pub = {c: v[pos] for c, v in pub.items()}
    return ({c: adjusted_rand_score(teacher, v) for c, v in pub.items()},
            adjusted_rand_score(pub["RNA_clusters"], pub["ATAC_clusters"]))


def in_sample_scores():
    """ARI of each model's own P22 Leiden clusters (K=10) with each reference; cached."""
    if CACHE.exists():
        return pd.read_csv(CACHE)
    al.CLUSTERING = "leiden"
    d = al.load_all()
    teacher = np.load(LEIDEN / "teacher_reference.npz", allow_pickle=True)["teacher_K10"].astype(str)
    refs = {"teacher_K10": teacher, **{k: np.asarray(v).astype(str) for k, v in d["p22_labels"].items()}}
    rows = []
    for m in PANEL_MODELS:
        own = al.cluster_k(d["models"][m]["p22"]["joint"], (10,))[10]
        for ref, lab in refs.items():
            rows.append(dict(model=m, reference=ref, in_sample_ari=adjusted_rand_score(lab, own)))
        print(m, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(CACHE, index=False)
    return out


def placed_scores():
    long = pd.concat([pd.read_csv(f) for f in sorted(LEIDEN.glob("alignment_long_*.csv"))])
    x = long[(long.readout == "backproject") & (long.view == "joint") & (long.metric == "ari")
             & (long["labels"] == "own_K10") & long.reference.isin(REFS) & long.model.isin(PANEL_MODELS)
             & long.method.isin(["knn", "ot"])]
    return x.rename(columns={"value": "placed_ari"})[["model", "method", "reference", "placed_ari"]]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    dashes = "--no-insample" not in sys.argv
    ins = in_sample_scores() if dashes else None
    plc = placed_scores()
    tea, inter = teacher_vs_published()
    cols = ("teacher",) + PANEL_MODELS
    labels = {"teacher": "teacher", **{m: MODELS[m][0] for m in PANEL_MODELS}}
    rc = {**RC, "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.7), sharey=True, gridspec_kw=dict(wspace=0.22))
        for j, (ref, title) in enumerate(REFS.items()):
            ax = axes[j]
            if ref == "teacher_K10":
                ax.text(0, 0.04, "= reference", ha="center", va="bottom", fontsize=7, color=DARK, rotation=90)
            else:
                ax.scatter(0, tea[ref], marker="D", s=26, color=TEACHER_COLOUR, edgecolor="black", lw=0.5, zorder=3)
                ax.axhline(inter, color=DARK, lw=0.8, ls=":", zorder=1)
            for i, m in enumerate(PANEL_MODELS, start=1):
                if dashes:
                    v = ins[(ins.model == m) & (ins.reference == ref)].in_sample_ari.iloc[0]
                    ax.plot([i - 0.3, i + 0.3], [v, v], color=DARK, lw=1.0)
                for k, c in enumerate(("knn", "ot")):
                    y = plc[(plc.model == m) & (plc.method == c) & (plc.reference == ref)].placed_ari.iloc[0]
                    ax.scatter(i + (k - 0.5) * 0.26, y, marker=COUPLINGS[c][1], s=26, color=MODELS[m][1],
                               edgecolor="black", lw=0.5, zorder=3)
            ax.set_xticks(range(len(cols)), [labels[m] for m in cols], rotation=30,
                          ha="right", rotation_mode="anchor")
            ax.set_xlim(-0.6, len(cols) - 0.4)
            ax.set_ylim(0, 1.02)
            ax.set_title(title, fontsize=9, pad=8)
            ax.text(-0.3 if j == 0 else -0.12, 1.06, "ab"[j], transform=ax.transAxes, fontsize=11, fontweight="bold",
                    va="bottom", ha="left")
        axes[0].set_ylabel("ARI with reference")
        handles = [Line2D([], [], marker=COUPLINGS[c][1], ls="", color="white", mec="black", label=COUPLINGS[c][0])
                   for c in ("knn", "ot")] + ([Line2D([], [], color=DARK, lw=1, label="in-sample")] if dashes else []) + [
            Line2D([], [], marker="D", ls="", color=TEACHER_COLOUR, mec="black", label="teacher domains\n(no placement)"),
            Line2D([], [], color=DARK, lw=0.8, ls=":", label="RNA vs ATAC\nannotations")]
        axes[-1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5, handlelength=1.0, borderaxespad=0,
                       labelspacing=0.3)
        for ext in ("png", "svg"):
            fig.savefig(out / f"FigS_heldout_published_reference.{ext}", dpi=300, bbox_inches="tight",
                        facecolor="white")
        plt.close(fig)
    vals = plc.merge(ins, on=["model", "reference"]) if dashes else plc
    vals = pd.concat([vals, pd.DataFrame([dict(model="teacher_own_domains_K10", method="none", reference=r, placed_ari=v)
                                          for r, v in tea.items()] + [dict(model="RNA_vs_ATAC_annotations",
                                          method="none", reference="published", placed_ari=inter)])])
    vals.to_csv(out / "FigS_heldout_published_reference_values.csv", index=False)
    print(vals.pivot_table(index=["reference", "model"], columns="method", values="placed_ari").round(3))
    if dashes:
        print(ins.pivot_table(index="model", columns="reference", values="in_sample_ari").round(3))


if __name__ == "__main__":
    main()

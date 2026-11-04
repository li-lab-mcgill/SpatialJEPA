"""Main-text Fig. 2f row (Leiden version of former Supplemental Fig. S5a,d-f).

draw_supp() (reduced supplementary figure) is retained but not called: that figure was dropped.

Fig2f_heldout_multiome_row
  (i)   ARI between Multiome Leiden clusters placed on P22 and the spatial teacher's Leiden domains
        (mean over K = 5/10/20), 15-NN and balanced-OT couplings; dash = in-sample (model's own P22
        embedding clustered), the ceiling for each model.
  (ii)  Teacher's Leiden domains on P22 (K = 10), the reference partition.
  (iii-v) P22 spots whose placed Multiome cluster (15-NN, K = 10, matched one-to-one to the teacher
        domains) equals the spot's teacher domain, coloured by that domain; grey otherwise.

S_heldout_multiome_p22
  (a) CellTypist labels placed on P22: NMI with the teacher's domains (all couplings).
  (b) Spatial coherence (kappa, 6-NN grid) of the placed CellTypist map; dash = teacher domains.
  (c) ARI between placed Multiome clusters (K = 10) and the published P22 RNA/ATAC clusters.
  (d) Known RNA-ATAC pairing: median W1 between a nucleus's RNA- and ATAC-view placements
      (filled) and between its RNA view and a same-CellTypist partner's ATAC view (open), 15-NN.

Inputs: heldout_alignment_leiden/ (heldout_multiome_alignment.py --clustering leiden, then
--summarize) and heldout_alignment/placement_w1_summary.csv (W1 placement does not depend on the
clustering). Env: nichecompass_liana. Usage: python plot_heldout_multiome_placement.py <out_dir>
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment

RUN = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/rna_atac_analysis/20260505_190037")
LEIDEN = RUN / "heldout_alignment_leiden"
KMEANS = RUN / "heldout_alignment"
PITCH = 23.0  # P22 coordinate units per spot width

# main-text Fig. 2e colours (Set2): teacher, student, non-spatial, MultiVI
MODELS = {"student": ("student", "#fc8d62"),
          "teacher_nograph": ("teacher, no graph", "#b3e2cd"),
          "nonspatial": ("non-spatial", "#8da0cb"),
          "multivi": ("MultiVI", "#e78ac3"),
          "pca": ("PCA / LSI", "#4d4d4d")}
COUPLINGS = {"knn": ("15-NN", "o"), "ot": ("balanced OT", "s"), "uot": ("unbalanced OT", "D")}
DARK = "#4d4d4d"
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
      "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
      "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False}


def panel_label(ax, letter, x=-0.3, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def save(fig, out, name, values=None):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if values is not None:
        values.to_csv(out / f"{name}_values.csv", index=False)
    print("wrote", out / f"{name}.png", flush=True)


def load_maps(src=LEIDEN):
    maps = {}
    for f in sorted(src.glob("backprojected_maps*.npz")):
        z = np.load(f, allow_pickle=True)
        maps.update({k: z[k] for k in z.files})
    return maps


FIG4A_LABELS = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/rna_atac_transfer_clusters/shared_params/shared_k150_r0p9_source_labels.csv")


def fig4a_colour_index(teacher, p22_obs):
    """Map each teacher cluster to the Fig. 4a source Leiden cluster it overlaps most (one-to-one, RNA rows),
    so teacher clusters take that cluster's tab10 colour."""
    lab = pd.read_csv(FIG4A_LABELS, dtype=str)
    rna = lab[lab.source_obs.str.endswith("_rna")]
    f4 = pd.Series(rna.leiden.to_numpy(), index=rna.source_obs.str[:-4]).reindex(p22_obs).to_numpy()
    C = pd.crosstab(np.asarray(teacher), f4)
    r, c = linear_sum_assignment(-C.values)
    return {C.index[i]: int(C.columns[j]) for i, j in zip(r, c)}


def matched_domain(own, teacher):
    """Rename placed clusters to the teacher domains they overlap most (one-to-one); -1 if unmatched."""
    a, ua = pd.factorize(np.asarray(own))
    b, ub = pd.factorize(np.asarray(teacher))
    C = np.zeros((len(ua), len(ub)))
    np.add.at(C, (a, b), 1)
    r, c = linear_sum_assignment(-C)
    mapping = dict(zip(r, c))
    placed = np.array([mapping.get(i, -1) for i in a])
    return np.where(placed == b, b, -1), ub


def xticks(ax, models):
    ax.set_xticks(range(len(models)), [MODELS[m][0].replace("\n", " ") for m in models],
                  rotation=50, ha="right", rotation_mode="anchor", fontsize=6.5)


def draw_row(out, src=None, name="Fig2f_heldout_multiome_row", ref_label="teacher domains"):
    src = src or LEIDEN
    t = pd.read_csv(src / "alignment_summary.csv", index_col=[0, 1])
    maps = load_maps(src)
    xy = maps["coords"][:, :2]
    teacher = np.asarray(maps["teacher_K10"]).astype(str)
    models = [m for m in MODELS if (m, "knn") in t.index]
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(7.2, 2.0))
        gs = fig.add_gridspec(1, 5, width_ratios=[0.95, 1, 1, 1, 1], wspace=0.06)

        ax = fig.add_subplot(gs[0, 0])
        for i, m in enumerate(models):
            ax.plot([i - 0.3, i + 0.3], [t.loc[(m, "knn"), "in_sample_ari"]] * 2, color=DARK, lw=1.0)
            for k, c in enumerate(("knn", "ot")):
                ax.scatter(i + (k - 0.5) * 0.24, t.loc[(m, c), "backproject_ari"], marker=COUPLINGS[c][1], s=24,
                           color=MODELS[m][1], edgecolor="black", lw=0.5, zorder=3)
        xticks(ax, models)
        ax.set(ylabel=f"ARI with {ref_label}", ylim=(0, 1.02))
        ax.set_title("Multiome clusters\nplaced on P22", fontsize=7.5)
        handles = [Line2D([], [], marker=COUPLINGS[c][1], ls="", color="white", mec="black", label=COUPLINGS[c][0])
                   for c in ("knn", "ot")] + [Line2D([], [], color=DARK, lw=1, label="in-sample")]
        ax.legend(handles=handles, loc="upper right", fontsize=5.5, handlelength=1.0, borderaxespad=0, labelspacing=0.3)
        panel_label(ax, "f", x=-0.42, y=1.02)

        cmap = plt.get_cmap("tab10")
        dom, ud = pd.factorize(teacher)
        cidx = fig4a_colour_index(teacher, maps["p22_obs"].astype(str))  # teacher cluster -> Fig. 4a colour
        ax = fig.add_subplot(gs[0, 1])
        ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(cidx[ud[i]] % 10) for i in dom], s=2.2, lw=0, rasterized=True)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{ref_label}\n(P22 reference)", fontsize=7.5)
        rows = []
        for j, m in enumerate(("student", "nonspatial", "multivi")):
            key = f"{m}__knn__own_K10"
            if key not in maps:
                continue
            placed, ub = matched_domain(maps[key], teacher)
            order = {lab: i for i, lab in enumerate(ud)}
            colors = [cmap(cidx[ub[p]] % 10) if p >= 0 else "#e6e6e6" for p in placed]
            ax = fig.add_subplot(gs[0, 2 + j])
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=2.2, lw=0, rasterized=True)
            ax.set_aspect("equal"); ax.axis("off")
            frac = (placed >= 0).mean()
            ax.set_title(f"{MODELS[m][0]}\n{frac:.0%} of spots agree", fontsize=7.5)
            rows.append(dict(model=m, frac_spots_matching_teacher_domain=frac))
        vals = t.reset_index().rename(columns={"level_0": "model", "level_1": "coupling"})
        save(fig, out, name, vals.merge(pd.DataFrame(rows), on="model", how="left"))


def draw_f(out, src, name, ref_label):
    """Panel f only, as laid out in main-text Fig. 2: no PCA/LSI baseline, labels rotated like panel e."""
    t = pd.read_csv(src / "alignment_summary.csv", index_col=[0, 1])
    models = [m for m in ("student", "teacher_nograph", "nonspatial", "multivi") if (m, "knn") in t.index]
    rc = {**RC, "font.size": 9.5, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5}  # +1.5 pt
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(1.43, 2.6))  # ~25% narrower than 1.9 in
        for i, m in enumerate(models):
            ax.plot([i - 0.3, i + 0.3], [t.loc[(m, "knn"), "in_sample_ari"]] * 2, color=DARK, lw=1.0)
            for k, c in enumerate(("knn", "ot")):
                ax.scatter(i + (k - 0.5) * 0.26, t.loc[(m, c), "backproject_ari"], marker=COUPLINGS[c][1], s=26,
                           color=MODELS[m][1], edgecolor="black", lw=0.5, zorder=3)
        ax.set_xticks(range(len(models)), [MODELS[m][0] for m in models], rotation=30, ha="right",
                      rotation_mode="anchor", fontsize=8.5)
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set(ylabel=f"ARI with {ref_label}", ylim=(0, 1.02))
        ax.set_title("Multiome clusters\nplaced on P22", fontsize=9.5)
        handles = [Line2D([], [], marker=COUPLINGS[c][1], ls="", color="white", mec="black", label=COUPLINGS[c][0])
                   for c in ("knn", "ot")] + [Line2D([], [], color=DARK, lw=1, label="in-sample")]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5, handlelength=1.0, borderaxespad=0, labelspacing=0.3)
        ax.text(-0.62, 1.02, "f", transform=ax.transAxes, fontsize=11.5, fontweight="bold", va="bottom", ha="left")
        save(fig, out, name, t.loc[models].reset_index())


def draw_maps(out, src, name, ref_label):
    """Panel g only: teacher reference and agreement maps, coloured as main-text Fig. 4a."""
    maps = load_maps(src)
    xy = maps["coords"][:, :2]
    teacher = np.asarray(maps["teacher_K10"]).astype(str)
    cmap = plt.get_cmap("tab10")
    dom, ud = pd.factorize(teacher)
    cidx = fig4a_colour_index(teacher, maps["p22_obs"].astype(str))
    rows = []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.04))
        axes[0].scatter(xy[:, 0], xy[:, 1], c=[cmap(cidx[ud[i]] % 10) for i in dom], s=2.2, lw=0, rasterized=True)
        axes[0].set_title(f"{ref_label}\n(P22 reference)", fontsize=8)
        for ax, m in zip(axes[1:], ("student", "nonspatial", "multivi")):
            placed, ub = matched_domain(maps[f"{m}__knn__own_K10"], teacher)
            ax.scatter(xy[:, 0], xy[:, 1], c=[cmap(cidx[ub[p]] % 10) if p >= 0 else "#e6e6e6" for p in placed],
                       s=2.2, lw=0, rasterized=True)
            frac = (placed >= 0).mean()
            ax.set_title(f"{MODELS[m][0]}\n{frac:.0%} of spots agree", fontsize=8)
            rows.append(dict(model=m, frac_spots_agreeing=frac))
        for ax in axes:
            ax.set_aspect("equal"); ax.axis("off")
        panel_label(axes[0], "g", x=-0.05, y=1.12)
        save(fig, out, name, pd.DataFrame(rows))


def draw_supp(out):
    t = pd.read_csv(LEIDEN / "alignment_summary.csv", index_col=[0, 1])
    p = pd.read_csv(KMEANS / "placement_w1_summary.csv", index_col=[0, 1])
    models = [m for m in MODELS if (m, "knn") in t.index]
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1), gridspec_kw=dict(wspace=0.55))
        for ax, col, ylab, title, letter in (
                (axes[0], "celltypist_nmi", "NMI with teacher domains", "CellTypist labels on P22", "a"),
                (axes[1], "kappa_celltypist", "spatial coherence (κ)", "Placed CellTypist map", "b"),
                (axes[2], "published_ari", "ARI with published clusters", "Placed clusters vs\npublished annotation", "c")):
            for i, m in enumerate(models):
                for k, c in enumerate(COUPLINGS):
                    ax.scatter(i + (k - 1) * 0.2, t.loc[(m, c), col], marker=COUPLINGS[c][1], s=20,
                               color=MODELS[m][1], edgecolor="black", lw=0.5, zorder=3)
            xticks(ax, models)
            ax.set(ylabel=ylab, ylim=(0, None))
            ax.set_title(title, fontsize=7.5)
            panel_label(ax, letter)
        handles = [Line2D([], [], marker=mk, ls="", color="white", mec="black", label=lab) for lab, mk in COUPLINGS.values()]
        axes[2].legend(handles=handles, loc="lower right", fontsize=6)

        ax = axes[3]
        pm = [m for m in models if (m, "knn") in p.index]
        for i, m in enumerate(pm):
            r = p.loc[(m, "knn")]
            own, part = r.paired / PITCH, r.same_celltype / PITCH
            ax.plot([i, i], [own, part], color=DARK, lw=0.8, zorder=1)
            ax.scatter(i, own, s=24, color=MODELS[m][1], edgecolor="black", lw=0.5, zorder=3)
            ax.scatter(i, part, s=24, facecolor="white", edgecolor=MODELS[m][1], lw=1.0, zorder=3)
        xticks(ax, pm)
        ax.set(ylabel="median W1 (spot widths)")
        ax.set_title("Known RNA–ATAC pairing (15-NN)", fontsize=7.5)
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color=DARK, label="own ATAC view"),
                           Line2D([], [], marker="o", ls="", mfc="white", mec=DARK, label="same-type partner")],
                  loc="upper right", fontsize=6)
        panel_label(ax, "d")
        save(fig, out, "S_heldout_multiome_p22_leiden", t.reset_index())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--clustering", choices=("leiden", "kmeans"), default="leiden")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.clustering == "kmeans":
        draw_row(args.out_dir, KMEANS, "Fig2f_heldout_multiome_row_kmeans", "teacher clusters")
        draw_maps(args.out_dir, KMEANS, "Fig2g_maps_kmeans", "teacher clusters")
        draw_f(args.out_dir, KMEANS, "Fig2f_ari_kmeans", "teacher clusters")
    else:
        draw_row(args.out_dir)
        draw_maps(args.out_dir, LEIDEN, "Fig2g_maps_leiden", "teacher clusters")
        draw_f(args.out_dir, LEIDEN, "Fig2f_ari_leiden", "teacher clusters")


if __name__ == "__main__":
    main()

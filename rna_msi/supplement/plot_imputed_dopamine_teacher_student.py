try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
# Fig. 5 candidate panels (2026-10-24): a compact panel i and a teacher-imputed dopamine map for row 3.
#
#   panel i      Student PLS C2 on the multiome nuclei, top-10 genes and top-10 metabolites by |weight|,
#                as one two-sided lollipop chart: genes extend left, metabolites right, rank on the shared
#                vertical axis (1 at the top). Stem length = |weight|; stem and marker colour = sign;
#                marker shape = feature type (gene circle, annotated metabolite diamond, unannotated m/z
#                square). Weights are those of the current panel i: corr(nucleus gene, student C2 score)
#                and the OT-recovered metabolite weights (msi_svd_intact/multiome/weights_target_*.csv.gz).
#   maps         teacher- (and student-) imputed msi:Dopamine on the P22 spatial target (the supplementary teacher map),
#                drawn like panels g and h: same map and colour-bar boxes as panel g of the top-10 layout
#                (264.0 x 258.8 pt map; 12.4 x 248.0 pt bar, 5.4 pt to its right), viridis on the
#                98th-percentile clip with gamma 2 (5__ FIG_dopamine_overview's factor_map_clip / _gamma),
#                markers s=10 on the #ededed backdrop, y mirrored as for g, 5__'s font sizes.
#   S5           the teacher map and the same map from the student's decoder side by side (Supplementary Fig. S5),
#                each on its own 2-98th percentile clip, as the two decoders' outputs are on different scales.
# Usage: plot_imputed_dopamine_teacher_student.py <out_dir> [<S5 path stem>]
#   env FIG567_RUN: model run of msi_components_common.RUNS (default "fixed s3")
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
BASE = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from msi_components_common import RUNS  # noqa: E402
RUN = os.getenv("FIG567_RUN", "fixed s3")
COMP = (BASE / "rna_msi_runs/shared_components/msi_svd_intact"
        / ("" if RUN == "fixed s3" else f"runs/{RUN.replace(' ', '_')}") / "multiome")
PROJ = BASE / f"spatialjepa_projection/V11L12-109_B1{RUNS[RUN]}"
FACTOR, N = "C2", 10
POS, NEG = "#b2182b", "#2166ac"
RC = {"font.size": 11, "axes.titlesize": 12, "svg.fonttype": "none", "pdf.fonttype": 42}
PT = 1 / 72


def top(path):
    w = pd.read_csv(path, index_col=0)[FACTOR]
    return w.reindex(w.abs().sort_values(ascending=False).index)[:N]


def lollipop():
    genes, mets = top(COMP / "weights_target_RNA.csv.gz"), top(COMP / "weights_target_MSI_OT.csv.gz")
    names = [m.replace("msi:", "") for m in mets.index]
    annotated = [not n.replace(".", "").isdigit() for n in names]
    lim = 0.05 * np.ceil(max(genes.abs().max(), mets.abs().max()) / 0.05) + 0.05
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(270 * PT, 330 * PT))
        # two mirrored axes around a central rank column, as in a population pyramid
        axl = fig.add_axes([0.21, 0.24, 0.27, 0.62])
        axr = fig.add_axes([0.55, 0.24, 0.27, 0.62])   # not sharey: each side keeps its own tick labels
        y = np.arange(1, N + 1)
        for ax, w, side in ((axl, genes, -1), (axr, mets, 1)):
            col = [POS if v > 0 else NEG for v in w]
            ax.hlines(y, 0, side * w.abs(), color=col, lw=1.6, zorder=2)
            shapes = ["o"] * N if side < 0 else ["D" if a else "s" for a in annotated]
            for yi, v, c, mk in zip(y, w, col, shapes):
                ax.scatter(side * abs(v), yi, s=34 if mk != "D" else 30, marker=mk, color=c,
                           edgecolor="black", linewidth=0.4, zorder=3)
            ax.set_xlim((-lim, 0) if side < 0 else (0, lim))
            ax.grid(axis="x", color="0.85", lw=0.6, zorder=0)
            ax.tick_params(axis="x", labelsize=7, length=2)
            ax.spines[["top", "left" if side > 0 else "right"]].set_visible(False)
        ticks = np.arange(0, lim + 1e-9, 0.2)
        axl.set_xticks(-ticks, [f"{t:.1f}" for t in ticks]); axr.set_xticks(ticks, [f"{t:.1f}" for t in ticks])
        for ax in (axl, axr):
            ax.set_ylim(N + 0.6, 0.4)
        axl.set_yticks(y, list(genes.index), fontsize=7.5, style="italic")
        axr.yaxis.tick_right()
        axr.set_yticks(y, names, fontsize=7.5)
        axl.tick_params(axis="y", length=2); axr.tick_params(axis="y", length=2)
        for yi in y:                                   # rank column between the two halves
            fig.text(0.515, axl.transData.transform((0, yi))[1] / fig.bbox.height, str(yi),
                     ha="center", va="center", fontsize=7, color="0.35")
        fig.text(0.515, 0.885, "rank", ha="center", fontsize=7, color="0.35")
        axl.set_xlabel("gene |weight|", fontsize=8); axr.set_xlabel("metabolite |weight|", fontsize=8)
        axl.set_title("genes", fontsize=9); axr.set_title("metabolites", fontsize=9)
        fig.suptitle(f"Student PLS {FACTOR} top-{N} features\n(dissociated multiome target)", fontsize=10, y=0.995)
        handles = [Line2D([], [], marker="o", ls="", color="0.5", mec="black", mew=0.4, ms=5.5, label="gene"),
                   Line2D([], [], marker="D", ls="", color="0.5", mec="black", mew=0.4, ms=5, label="annotated metabolite"),
                   Line2D([], [], marker="s", ls="", color="0.5", mec="black", mew=0.4, ms=5.5, label="unannotated m/z"),
                   Line2D([], [], color=POS, lw=2, label="positive weight"),
                   Line2D([], [], color=NEG, lw=2, label="negative weight")]
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5, frameon=False,
                   bbox_to_anchor=(0.52, 0.0), columnspacing=1.0, handletextpad=0.4)
        for ext in ("svg", "png"):
            fig.savefig(OUT / f"Fig5i_lollipop.{ext}", dpi=300, facecolor="white")
        plt.close(fig)
    table = pd.concat([pd.DataFrame({"rank": y, "feature": genes.index, "type": "gene", "weight": genes.values}),
                       pd.DataFrame({"rank": y, "feature": names, "type": np.where(annotated, "annotated metabolite",
                                                                                     "unannotated m/z"),
                                     "weight": mets.values})])
    table.to_csv(OUT / "Fig5i_lollipop_values.csv", index=False)
    return table


def colorbar_ticks(norm, nbins=5):
    """5__ _fig_spatial_scatter's tick thinning: round values spaced by position along the bar."""
    cand = MaxNLocator(nbins=nbins * 2).tick_values(norm.vmin, norm.vmax)
    cand = cand[(cand >= norm.vmin) & (cand <= norm.vmax)]
    ticks, last = [], -np.inf
    for v, p in zip(cand, np.asarray(norm(cand), dtype=float)):
        if p - last >= 1.0 / (nbins + 2):
            ticks.append(v); last = p
    return ticks


def imputed_dopamine(which):
    """Imputed msi:Dopamine on the P22 spots and their (x, -y) coordinates, from the teacher or student decoder."""
    d = np.load(PROJ / f"spatial_target_{which}_msi_imputed.npz", allow_pickle=True)
    feats = [str(f) for f in d["msi_feature_names"]]
    obs = pd.Index([str(o) for o in d["obs_names"]])
    values = np.asarray(d[f"{which}_msi"][:, feats.index("msi:Dopamine")], dtype=float)
    with h5py.File(PROJ / "spatial_target_rna_atac_msi.h5mu", "r") as h:
        idx = h["obs"][h["obs"].attrs["_index"]][:].astype(str)
        xy = h["obsm"]["spatial"][:]
    pos = pd.Index(idx).get_indexer(obs)
    assert (pos >= 0).all()
    return values, np.asarray(xy, dtype=float)[pos] * np.array([1, -1])     # panel g's orientation


def draw_map(fig, box, values, xy, title):
    """One map in panel g's style; box = (x0, y0, W, H) in pt, the map and colour bar placed as in panel g."""
    x0, y0, W, H = box
    vmin, vmax = np.percentile(values, [2, 98])
    norm = PowerNorm(2.0, vmin=vmin, vmax=vmax, clip=True)
    mw, mh = 264.0, 258.77
    ax = fig.add_axes([x0 / W, y0 / H, mw / W, mh / H])
    cax = fig.add_axes([(x0 + mw + 5.39) / W, (y0 + 5.38) / H, 12.4 / W, 248.01 / H])
    order = np.argsort(values)
    art = ax.scatter(xy[order, 0], xy[order, 1], c=values[order], cmap="viridis", norm=norm, s=10,
                     linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim"); ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([]); ax.set_facecolor("#ededed")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title)
    cb = fig.colorbar(art, cax=cax)
    cb.set_label("imputed dopamine", fontsize=9); cb.ax.tick_params(labelsize=8)
    cb.set_ticks(colorbar_ticks(norm))
    return ax, [float(vmin), float(vmax)]


def single_maps():
    """Teacher and student maps as separate panels, each on panel g's canvas."""
    W, H = 340.0, 310.0
    info = {}
    for which in ("teacher", "student"):
        values, xy = imputed_dopamine(which)
        with plt.rc_context(RC):
            fig = plt.figure(figsize=(W * PT, H * PT))
            _, clip = draw_map(fig, (4.0, 4.0, W, H), values, xy, f"Imputed dopamine — {which}\n(P22 spatial target)")
            for ext in ("svg", "png"):
                fig.savefig(OUT / f"Fig5_{which}_imputed_dopamine.{ext}", dpi=300, facecolor="white")
            plt.close(fig)
        info[which] = dict(n_spots=len(values), clip=clip)
    return info


def supplementary_s5(path):
    """Supplementary Fig. S5: (a) teacher- and (b) student-imputed dopamine side by side, each with its own bar."""
    W, H, step = 700.0, 300.0, 345.0
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(W * PT, H * PT))
        for k, which in enumerate(("teacher", "student")):
            values, xy = imputed_dopamine(which)
            ax, _ = draw_map(fig, (24.0 + k * step, 4.0, W, H), values, xy, f"{which}-imputed dopamine")
            ax.text(-0.04, 1.06, "ab"[k], transform=ax.transAxes, fontsize=14, fontweight="bold", ha="right")
        for ext in ("svg", "png"):
            fig.savefig(path.with_suffix(f".{ext}"), dpi=300, facecolor="white")
        plt.close(fig)
    s5_numbers(path.parent / "s5_imputed_dopamine_by_cluster.csv")


def s5_numbers(path):
    """The numbers in the S5 text: per-spot teacher-student correlation and per-cluster means of the z-scored maps."""
    from scipy.stats import pearsonr, spearmanr
    obs = pd.Index([str(o) for o in np.load(PROJ / "spatial_target_teacher_msi_imputed.npz", allow_pickle=True)["obs_names"]])
    v = pd.DataFrame({w: imputed_dopamine(w)[0] for w in ("teacher", "student")}, index=obs)
    z = (v - v.mean()) / v.std()
    with h5py.File(PROJ / "spatial_target_rna_atac_msi.h5mu", "r") as h:
        idx = pd.Index(h["obs"][h["obs"].attrs["_index"]][:].astype(str))
        labels = {k: pd.Categorical.from_codes(h["obs"][k]["codes"][:], h["obs"][k]["categories"][:].astype(str))
                  for k in ("RNA_clusters", "ATAC_clusters")}
    tables = []
    for k, lab in labels.items():
        t = z.assign(cluster=pd.Series(lab, index=idx).reindex(obs).values).groupby("cluster", observed=True)
        t = t.mean().join(t.size().rename("n_spots")).sort_values("teacher", ascending=False)
        tables.append(t.rename(columns={"teacher": "teacher_z_mean", "student": "student_z_mean"})
                      .reset_index().assign(clustering=k))
    r, rho = pearsonr(v["teacher"], v["student"])[0], spearmanr(v["teacher"], v["student"])[0]
    with open(path, "w") as f:
        f.write(f"# teacher vs student per spot (n = {len(obs)}): pearson {r:.3f}, spearman {rho:.3f}\n")
        pd.concat(tables)[["clustering", "cluster", "n_spots", "teacher_z_mean", "student_z_mean"]].round(3).to_csv(f, index=False)


if __name__ == "__main__":
    t = lollipop()
    print(t.round(3).to_string(index=False))
    print(single_maps())
    if len(sys.argv) > 2:
        supplementary_s5(Path(sys.argv[2]))

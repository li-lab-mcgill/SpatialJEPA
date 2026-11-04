"""Supplementary Fig. S1: integration metrics (a-b) + per-model UMAPs (c-f).

a/b: bars drawn as in plot_scib_barplot (analyze_rna_atac.py): Set2 colour per
     model in the same order, hatching for in-sample scores, 95% intervals, and
     y axes zoomed to the occupied range. Metrics: 1-FOSCTTM, iLISI, bio_cons.
c/d: source UMAPs of each model's joint RNA+ATAC embedding, coloured by modality
     (c) and by RNA cluster (d).
e/f: the same for the target, coloured by modality (e) and CellTypist prediction (f).
Built at the supplement's text width (6.5 in), so fonts print at nominal size.
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

WORK = f"{os.environ['SPATIALJEPA_HOME']}/outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap_v2"
UMAPS = os.path.join(WORK, "s1_umaps")
FIG_BASE = f"{os.environ['SPATIALJEPA_MANUSCRIPT']}/figures/supplementary/S1_P22_scIB_v3"

SRC = ["teacher_source", "source", "source_nonspatial", "source_multivi"]
TGT = ["target", "target_ot", "target_multivi_zero_shot"]
LABELS = {
    "teacher_source": "teacher_source", "source": "student_source",
    "source_nonspatial": "nonspatial_source", "source_multivi": "multivi_source",
    "target": "student_target_zero_shot", "target_ot": "student_target_OT",
    "target_multivi_zero_shot": "multiVI_target_zero_shot",
}
IN_SAMPLE = {"teacher_source", "source", "source_nonspatial", "source_multivi"}
METRICS = [("foscttm", "1-FOSCTTM"), ("ilisi", "iLISI"), ("bio_conservation", "bio_cons.")]
MODEL_COLORS = dict(zip(SRC + TGT, sns.color_palette("Set2", n_colors=7)))

# scanpy default_20, assigned in the label orders used by main Fig. 2
DEFAULT_20 = ['#1f77b4', '#ff7f0e', '#279e68', '#d62728', '#aa40fc', '#8c564b', '#e377c2',
              '#b5bd61', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
              '#c49c94', '#f7b6d2', '#dbdb8d', '#9edae5', '#ad494a', '#8c6d31']
MODALITY_COLORS = {"atac": DEFAULT_20[0], "rna": DEFAULT_20[1]}

FS_TICK, FS_TITLE, FS_LEGEND, FS_LETTER = 5.5, 6.5, 5.5, 9
mpl.rcParams.update({"font.size": FS_TICK, "axes.linewidth": 0.4, "hatch.linewidth": 0.6})

# v2 metrics, with MultiVI zero-shot re-scored on all target nuclei (metrics_bootstrap_v3)
metrics = pd.read_csv(os.path.join(os.path.dirname(WORK), "metrics_bootstrap_v3", "metrics_with_ci.csv"))
# UMAP settings for panels c-f (S1_UMAP_SETTINGS):
#   "target" (manuscript): every UMAP uses the Fig. 2d settings (n_neighbors=30,
#       min_dist=0.1, spread=1.5); target panels on the Fig. 2d nuclei.
#       Source caches: umap_source_embeddings.py; target caches:
#       umap_target_embeddings.py.
#   "source" / "mixed": alternatives explored before choosing "target" (Fig. 2b/c
#       settings everywhere / Fig. 2b/c for source and OT panels, Fig. 2d for the
#       zero-shot student and MultiVI); written to previews/ only.
UMAP_MODE = os.environ.get("S1_UMAP_SETTINGS", "target")
UMAP_FILES = {
    "target": {**{d: d + "__fig2d_settings" for d in SRC},
               "target": "target__fig2d", "target_ot": "target_ot__fig2d",
               "target_multivi_zero_shot": "target_multivi_zero_shot__fig2d_full_target"},
    "source": {"target": "target__subset_src_settings", "target_ot": "target_ot__subset_src_settings",
               "target_multivi_zero_shot": "target_multivi_zero_shot__subset_src_settings_full_target"},
    "mixed": {"target": "target__fig2d", "target_ot": "target_ot__subset_src_settings",
              "target_multivi_zero_shot": "target_multivi_zero_shot__fig2d_full_target"},
}[UMAP_MODE]
if UMAP_MODE != "target":
    FIG_BASE = os.path.join(os.path.dirname(WORK), "metrics_bootstrap_v3", "previews",
                            f"S1_preview_{UMAP_MODE}_settings")
    os.makedirs(os.path.dirname(FIG_BASE), exist_ok=True)
umaps = {d: pd.read_csv(os.path.join(UMAPS, UMAP_FILES.get(d, d) + ".csv.gz"), index_col=0)
         for d in SRC + TGT}
# UMAP axis signs are arbitrary (spectral initialization), so panels are
# oriented for comparison (the student source panel is sign-aligned to the
# teacher in its cache):
# - OT-aligned points are near-copies of source spots: mirror x so the OT panel
#   faces the student_source panel
umaps["target_ot"]["umap1"] *= -1
# - zero-shot student: this recomputation came out with both axes inverted
#   relative to Fig. 2d (same nuclei and settings), so flip both to match it
if UMAP_MODE != "source":
    umaps["target"][["umap1", "umap2"]] *= -1


def natural(labels):
    return sorted(labels, key=lambda s: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)])


def label_palette(domains, order=sorted):
    """Colours in the order main Fig. 2 uses: R0..R10 naturally, CellTypist alphabetically."""
    cats = order({l for d in domains for l in umaps[d]["label"].unique()})
    return dict(zip(cats, DEFAULT_20))


# ── layout, in inches ────────────────────────────────────────────────────
W = 6.5
LEFT = 0.38                      # room for panel letters + row labels
BAR_W, BAR_H, BAR_TICK = 0.62, 0.80, 0.28
UMAP_S, UMAP_GAP = 1.15, 0.08
LEG_TOP = 0.42                   # bar legend band
BAR_TOP = LEG_TOP + 0.20         # metric titles
UMAP_TOP = BAR_TOP + BAR_H + 0.42
ROW_PITCH = UMAP_S + 0.24
H = UMAP_TOP + 4 * ROW_PITCH - 0.10

fig = plt.figure(figsize=(W, H))


def axes_in(x, y_top, w, h):
    """Axes from inch coordinates measured from the top-left corner."""
    return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])


def letter(x, y_top, text):
    fig.text(x / W, 1 - y_top / H, text, fontsize=FS_LETTER, fontweight="bold", va="top", ha="left")


# ── a/b: bars ────────────────────────────────────────────────────────────
def draw_bars(ax, metric, domains):
    extrema = []
    for i, d in enumerate(domains):
        r = metrics[(metrics.domain == d) & (metrics.metric_name == metric)].iloc[0]
        ax.bar(i, r.value, width=0.8, color=MODEL_COLORS[d],
               edgecolor="#555555" if d in IN_SAMPLE else "#cccccc", linewidth=0.4,
               hatch="//" if d in IN_SAMPLE else None)
        ax.errorbar(i, r.value, yerr=[[max(r.value - r.ci_low, 0.0)], [max(r.ci_high - r.value, 0.0)]],
                    fmt="none", ecolor="#444444", elinewidth=0.5, capsize=1.0, zorder=3)
        extrema += [r.value, r.ci_low, r.ci_high]
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(domains) - 0.3)
    ax.tick_params(axis="y", labelsize=FS_TICK, width=0.4, length=1.8, pad=1)
    extrema = [v for v in extrema if np.isfinite(v) and v > 0]  # same zoom rule as the cell
    lo, hi = min(extrema), max(extrema)
    margin = (hi - lo) * 0.2 or lo * 0.1
    ax.set_ylim(max(0, lo - margin), hi + margin)


# source panels span 0.76-3.18 in; the target block starts after a gap wide
# enough for its letter, y label and tick labels
for group, domains, x0, x_letter, name in [("a", SRC, LEFT + 0.10 + BAR_TICK, 0.02, "source"),
                                            ("b", TGT, 3.88, 3.36, "target")]:
    letter(x_letter, BAR_TOP - 0.18, group)
    for j, (metric, title) in enumerate(METRICS):
        ax = axes_in(x0 + j * (BAR_W + BAR_TICK), BAR_TOP, BAR_W, BAR_H)
        draw_bars(ax, metric, domains)
        ax.set_title(title, fontsize=FS_TITLE, pad=3)
        if j == 0:
            ax.set_ylabel(name, fontsize=FS_TITLE, fontweight="bold", labelpad=2)

handles = [mpl.patches.Patch(facecolor=MODEL_COLORS[d], edgecolor="#555555" if d in IN_SAMPLE else "#cccccc",
                             hatch="//" if d in IN_SAMPLE else None, linewidth=0.4, label=LABELS[d])
           for d in SRC + TGT]
handles.append(mpl.patches.Patch(facecolor="white", edgecolor="#555555", hatch="//", linewidth=0.4,
                                 label="scored on training cells"))
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.02 / H), ncol=4,
           fontsize=FS_LEGEND, frameon=True, handlelength=1.6, columnspacing=1.2, borderpad=0.4)


# ── c-f: UMAPs ───────────────────────────────────────────────────────────
def draw_umap(ax, d, colors):
    u = umaps[d]
    order = np.random.default_rng(0).permutation(len(u))  # no modality drawn on top
    size = float(np.clip(6000 / len(u), 0.1, 1.0))
    ax.scatter(u["umap1"].values[order], u["umap2"].values[order], c=colors[order],
               s=size, linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.3); sp.set_color("#999999")
    ax.set_title(LABELS[d], fontsize=FS_TICK, pad=2)


def legend_block(x, y_top, entries, title, ncol):
    ax = axes_in(x, y_top, W - x - 0.02, UMAP_S)
    ax.axis("off")
    hs = [mpl.lines.Line2D([], [], marker="o", ls="", markersize=3.2, markerfacecolor=c,
                           markeredgewidth=0, label=l) for l, c in entries]
    ax.legend(handles=hs, title=title, loc="center left", ncol=ncol, fontsize=FS_LEGEND,
              title_fontsize=FS_LEGEND, frameon=False, handletextpad=0.2, columnspacing=0.8,
              labelspacing=0.35, borderaxespad=0.0)


src_pal, tgt_pal = label_palette(SRC, order=natural), label_palette(TGT)
rows = [
    ("c", "source", SRC, "modality", MODALITY_COLORS, [("RNA", MODALITY_COLORS["rna"]), ("ATAC", MODALITY_COLORS["atac"])], "Modality", 1),
    ("d", "source", SRC, "label", src_pal, [(l, src_pal[l]) for l in natural(src_pal)], "RNA clusters", 2),
    ("e", "target", TGT, "modality", MODALITY_COLORS, [("RNA", MODALITY_COLORS["rna"]), ("ATAC", MODALITY_COLORS["atac"])], "Modality", 1),
    ("f", "target", TGT, "label", tgt_pal, [(l, tgt_pal[l]) for l in sorted(tgt_pal)], "CellTypist predictions", 2),
]
for r, (let, name, domains, key, pal, entries, leg_title, ncol) in enumerate(rows):
    y_top = UMAP_TOP + r * ROW_PITCH
    letter(0.02, y_top - 0.16, let)
    for c, d in enumerate(domains):
        ax = axes_in(LEFT + c * (UMAP_S + UMAP_GAP), y_top, UMAP_S, UMAP_S)
        draw_umap(ax, d, umaps[d][key].map(pal).values)
        if c == 0:
            ax.set_ylabel(name, fontsize=FS_TITLE, fontweight="bold", labelpad=2)
    legend_block(LEFT + len(domains) * (UMAP_S + UMAP_GAP) + 0.05, y_top, entries, leg_title, ncol)

with plt.rc_context({"svg.fonttype": "none"}):
    fig.savefig(FIG_BASE + ".svg", format="svg", dpi=600)
fig.savefig(FIG_BASE + ".png", format="png", dpi=450)
print("Saved:", FIG_BASE + ".svg", FIG_BASE + ".png", f"({W} x {H:.2f} in)")

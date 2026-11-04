try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
# Draft panel: source dopamine weight of a latent direction vs NMF-22 convergence of its target gene
# weights. Grey: 5,000 random latent directions (oriented dopamine >= 0); dots: the six MSI components.
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RUN = (sys.argv[3] if len(sys.argv) > 3 else "fixed s3").replace(" ", "_")   # as dopamine_direction_null*.py
INK, MUTED, CLOUD, DA_RED = "#222222", "#6b6b6b", "#c9c9c9", "#b2182b"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "font.family": "DejaVu Sans"})

sj = pickle.load(open(D / f"direction_null_{RUN}.pkl", "rb"))
pca = pickle.load(open(D / f"direction_null_rnaPCA_{RUN}.pkl", "rb"))


def facet(ax, res, model, title, show_x=True, show_y=True):
    p = res["pts"][model]
    x_r, y_r = res["dw_rand"], p["rand"]
    ax.scatter(x_r, y_r, s=3, c=CLOUD, lw=0, rasterized=True, zorder=1)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (3, 3)), zorder=0)
    placed = []
    order = np.argsort(-np.asarray(p["comp"]))
    for k in order:
        c = res["comps"][k]
        da = c == res["c_da"]
        x, y = res["dw_comp"][k], p["comp"][k]
        ax.scatter(x, y, s=46 if da else 30, c=DA_RED if da else INK, edgecolor="white", lw=1.2, zorder=3)
        ly = y
        dots = [(res["dw_comp"][j], p["comp"][j]) for j in range(len(res["comps"])) if j != k]
        while (any(abs(x - px) < 0.09 and abs(ly - py) < 0.035 for px, py in placed)
               or any(x < px < x + 0.11 and abs(ly - py) < 0.03 for px, py in dots)):
            ly -= 0.035                      # nudge the label down until it clears earlier ones
        placed.append((x, ly))
        ax.annotate(c, (x, y), xytext=(x + 0.025, ly), fontsize=8, ha="left", va="center",
                    color=INK, fontweight="bold" if da else "normal", zorder=4,
                    arrowprops=None if ly == y else dict(arrowstyle="-", color=MUTED, lw=0.5,
                                                         shrinkA=0, shrinkB=3))
    k = res["comps"].index(res["c_da"])
    pct = (y_r < p["comp"][k]).mean() * 100
    r = np.corrcoef(x_r, y_r)[0, 1]
    pct_s = f"{pct:.1f}" if pct >= 99 else f"{pct:.0f}"
    ax.text(0.97, 0.04, f"5,000 random directions (grey)\nr = {r:.2f}; {res['c_da']} above {pct_s}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=MUTED)
    ax.set_title(title, fontsize=8.5, color=INK, pad=4)
    ax.set_xlim(-0.03, 0.72)
    ax.set_ylim(0.15, 0.93)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5, width=0.6)
    if show_x:
        ax.set_xlabel("dopamine weight in source (measured MSI)")
    else:
        ax.tick_params(labelbottom=False)
    if show_y:
        ax.set_ylabel("NMF-22 convergence (AUROC)")
    else:
        ax.tick_params(labelleft=False)


# main-figure draft: two facets stacked (fits the panel-i slot of Fig. 5)
fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.8), sharex=True)
facet(axes[0], sj, "teacher @ P22", "Teacher (P22 spatial target)", show_x=False)
facet(axes[1], sj, "student @ multiome", "Student (dissociated multiome target)")
fig.tight_layout(h_pad=1.2)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"draft_dopamine_dose_panel.{ext}", dpi=300)
plt.close(fig)

# supplement draft: SpatialJEPA (top) vs the RNA-PCA control (bottom), P22 and multiome
fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.2), sharex=True, sharey=True)
facet(axes[0, 0], sj, "teacher @ P22", "Teacher latent, P22", show_x=False)
facet(axes[0, 1], sj, "student @ multiome", "Student latent, multiome", show_x=False, show_y=False)
facet(axes[1, 0], pca, "rnaPCA @ P22", "RNA-PCA control, P22")
facet(axes[1, 1], pca, "rnaPCA @ multiome", "RNA-PCA control, multiome", show_y=False)
fig.tight_layout(h_pad=1.2, w_pad=1.0)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"draft_dopamine_dose_supplement.{ext}", dpi=300)
print("wrote", OUT)

"""Draw the two supplementary figures that accompany Fig. 7 (seed0).

Supp A, metabolite transfer: (a) metabolite weights of C1 on each held-out source section, recovered
by OT from the other two sections vs measured; (b) transferred metabolites by CaH subclass (mean of
the OT-paired values, z per metabolite, over the 40,000 Fig. 7 nuclei).
Supp B (PLS C2) moved to plot_component_c2.py (Fig. 7 layout) and plot_metabolite_transfer_c2.py (its metabolite transfer).
Inputs: supp_metab.pkl (metabolite_transfer_supplement.py), c2.pkl (component_c2.py).
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer"
OUT = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer")
OUT.mkdir(parents=True, exist_ok=True)
MSI_C, RNA_C, INK, MUTED = "#1f77b4", "#ff7f0e", "#1f1f1f", "#6b6b6b"     # Fig. 7a view colours
SEC_C = ["#4c78a8", "#f58518", "#54a24b"]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
M = pickle.load(open(HERE / "supp_metab.pkl", "rb"))
C = pickle.load(open(HERE / "c2.pkl", "rb"))
names = pd.Index(M["metabolites"]).str.replace("msi:", "", regex=False)
da = names.get_loc("Dopamine")
SECS = sorted(M["held_out"])


def tag(ax, s):
    ax.text(-0.2, 1.12, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")


def heldout_scatter(ax, k, title):
    for i, s in enumerate(SECS):
        h = M["held_out"][s]
        j = int(np.argmax(h["r_U_vs_full"][:, k]))           # the held-out component matching full-data Ck
        x, y = h["measured"][:, j], h["ot"][:, j]
        sg = np.sign(np.corrcoef(h["ot"][:, j], C["W_msi_ot"].to_numpy() if k == 1 else x)[0, 1]) if k == 1 else 1
        x, y = x * sg, y * sg
        r = np.corrcoef(x, y)[0, 1]
        ax.scatter(x, y, s=5, color=SEC_C[i], alpha=0.55, linewidths=0, label=f"{s.split('_')[1]} held out (r {r:.2f})")
        if k == 0:
            ax.scatter(x[da], y[da], s=40, facecolor="none", edgecolor=INK, linewidth=1.0, zorder=3)
    lim = 1.0
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel("weight from measured MSI\n(held-out section)"); ax.set_ylabel("weight recovered by OT\n(other two sections)")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", handletextpad=0.2, markerscale=2)


# ------------------------------------------------------------------ Supp A
sub = np.asarray(M["subclass"])[M["pick"]]
Y = pd.DataFrame(M["paired"], columns=names)
Y = (Y - Y.mean()) / Y.std()
W1 = pd.read_csv(Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/human_cah_transfer/seed0/cah/pls_msi_ot_reg0.001_sec/weights_target_MSI_OT.csv.gz"), index_col=0)["PLS C1"]
W1.index = W1.index.str.replace("msi:", "", regex=False)
cols = ["Dopamine", "Serotonin", "3-MT", "Histidine", "GABA", "Norepinephrine", "Taurine",
        "449.1863", "450.18976", "570.23167", "572.24122"]
counts = pd.Series(sub).value_counts()
keep = counts[counts >= 50].index
H = Y[cols].groupby(sub).mean().loc[keep]
c1 = C["summary"]  # unused; order rows by C1 mean from the snRNA scores
order = pd.read_parquet(HERE.parent / "astro_c1_adnc" / "cells.parquet", columns=["Subclass", "C1_seed0"]).groupby("Subclass", observed=True)["C1_seed0"].mean()
order.index = order.index.astype(str)
H = H.loc[order.reindex(H.index).sort_values(ascending=False).index]

fig = plt.figure(figsize=(12, 9.2), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
for i, s in enumerate(SECS):
    ax = fig.add_subplot(gs[0, i])
    h = M["held_out"][s]; j = int(np.argmax(h["r_U_vs_full"][:, 0]))
    x, y = h["measured"][:, j], h["ot"][:, j]
    ax.scatter(x, y, s=6, color=MSI_C, alpha=0.5, linewidths=0)
    ax.scatter(x[da], y[da], s=45, facecolor="none", edgecolor=INK, linewidth=1.1, zorder=3)
    ax.annotate("dopamine", (x[da], y[da]), xytext=(6, -12), textcoords="offset points", fontsize=8)
    ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    rk = lambda v: int((v > v[da]).sum()) + 1
    ax.set_title(f"{s} held out\nr = {np.corrcoef(x, y)[0, 1]:.2f}; dopamine rank {rk(y)} (measured {rk(x)})", fontsize=10)
    ax.set_xlabel("C1 weight, measured MSI")
    if i == 0:
        ax.set_ylabel("C1 weight recovered by OT\nfrom the other two sections")
        tag(ax, "a")
ax = fig.add_subplot(gs[1, :])
v = np.nanpercentile(np.abs(H.to_numpy()), 98)
im = ax.imshow(H.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
ax.set_yticks(range(len(cols)), [f"{c}  ({W1[c]:+.2f})" for c in cols])
ax.set_xticks(range(len(H)), H.index, rotation=60, ha="right")
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("transferred metabolite\n(subclass mean, z)")
ax.set_title("Metabolites transferred to CaH nuclei, by subclass (subclasses ordered by mean C1; C1 metabolite weight in brackets)", fontsize=10)
tag(ax, "b")
for ext in ("png", "svg"):
    fig.savefig(OUT / f"supp_metabolite_transfer.{ext}", dpi=300)
plt.close(fig)

# Supp B (PLS C2) is drawn by plot_component_c2.py and plot_metabolite_transfer_c2.py.
print("wrote", OUT)

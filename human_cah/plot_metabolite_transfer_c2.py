"""Supplementary figure: metabolite transfer for PLS C2, the C2 counterpart of supp_metabolite_transfer.

(a) C2 metabolite weights on each held-out source section: recovered by OT from the other two sections
    vs measured (held-out component matched to the full-data C2 by |cos|, signs aligned to it);
    norepinephrine ringed.
(b) Metabolites transferred to CaH nuclei by subclass (mean of OT-paired values, z per metabolite, over
    the 40,000 Fig. 7 nuclei), subclasses ordered by mean C2, C2 weight in brackets.
Inputs: supp_metab.pkl (metabolite_transfer_supplement.py), c2.pkl (component_c2.py). Writes supp_metabolite_transfer_c2.{png,svg}.
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

HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer"
BASE = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/human_cah_transfer/seed0/cah")
OUT = BASE / "supp_metabolite_transfer"
MSI_C, INK, MUTED = "#1f77b4", "#1f1f1f", "#6b6b6b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
M = pickle.load(open(HERE / "supp_metab.pkl", "rb"))
C = pickle.load(open(HERE / "c2.pkl", "rb"))
R = pickle.load(open(BASE / "pls_msi_ot_reg0.001_sec/results.pkl", "rb"))
names = pd.Index(M["metabolites"]).str.replace("msi:", "", regex=False)
ne = names.get_loc("Norepinephrine")
K = R["comps"].index("PLS C2")


def tag(ax, s):
    ax.text(-0.2, 1.12, s, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom", ha="right")


fig = plt.figure(figsize=(12, 9.2), constrained_layout=True)
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
for i, s in enumerate(sorted(M["held_out"])):
    h = M["held_out"][s]
    j = int(np.argmax(h["r_U_vs_full"][:, K]))
    sg = np.sign(h["U"][:, j] @ R["U"][:, K])
    x, y = h["measured"][:, j] * sg, h["ot"][:, j] * sg
    ax = fig.add_subplot(gs[0, i])
    ax.scatter(x, y, s=6, color=MSI_C, alpha=0.5, linewidths=0)
    ax.scatter(x[ne], y[ne], s=45, facecolor="none", edgecolor=INK, linewidth=1.1, zorder=3)
    ax.annotate("norepinephrine", (x[ne], y[ne]), xytext=(6, -12), textcoords="offset points", fontsize=8)
    ax.plot([-1, 1], [-1, 1], color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_aspect("equal")
    rk = lambda v: int((v > v[ne]).sum()) + 1
    ax.set_title(f"{s} held out\nr = {np.corrcoef(x, y)[0, 1]:.2f}; |cos| with C2 {h['r_U_vs_full'][j, K]:.2f}\n"
                 f"norepinephrine rank {rk(y)} (measured {rk(x)})", fontsize=9.5)
    ax.set_xlabel("C2 weight, measured MSI")
    if i == 0:
        ax.set_ylabel("C2 weight recovered by OT\nfrom the other two sections")
        tag(ax, "a")

sub = np.asarray(M["subclass"])[M["pick"]]
Y = pd.DataFrame(M["paired"], columns=names)
Y = (Y - Y.mean()) / Y.std()
W2 = R["W"]["target MSI (OT)"]["PLS C2"]; W2.index = W2.index.str.replace("msi:", "", regex=False)
cols = ["Norepinephrine", "Dopamine", "Serotonin", "3-MT", "Histidine", "GABA", "Taurine",
        "489.94697", "574.23222", "573.22368", "552.24744", "555.18875"]
counts = pd.Series(sub).value_counts()
H = Y[cols].groupby(sub).mean().loc[counts[counts >= 50].index]
order = R["scores"]["PLS C2"].groupby(pd.read_parquet(HERE.parent / "astro_c1_adnc" / "cells.parquet", columns=["Subclass"])["Subclass"].astype(str).to_numpy()).mean()
H = H.loc[order.reindex(H.index).sort_values(ascending=False).index]
ax = fig.add_subplot(gs[1, :])
v = np.nanpercentile(np.abs(H.to_numpy()), 98)
im = ax.imshow(H.T.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
ax.set_yticks(range(len(cols)), [f"{c}  ({W2[c]:+.2f})" for c in cols])
ax.set_xticks(range(len(H)), H.index, rotation=60, ha="right")
ax.spines[:].set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01); cb.set_label("transferred metabolite\n(subclass mean, z)")
ax.set_title("Metabolites transferred to CaH nuclei, by subclass (subclasses ordered by mean C2; C2 metabolite weight in brackets)", fontsize=10)
tag(ax, "b")

for ext in ("png", "svg"):
    fig.savefig(OUT / f"supp_metabolite_transfer_c2.{ext}", dpi=300)
print("wrote", OUT / "supp_metabolite_transfer_c2.png")

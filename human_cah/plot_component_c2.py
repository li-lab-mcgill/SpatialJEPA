"""Supplementary figure on PLS C2 in the layout of Fig. 7 (cah_cell_runner.fig7_board), seed0.

    python plot_component_c2.py

a  top C2 RNA and OT-recovered MSI weights (Fig. 7a function)
b  RNA UMAPs coloured by C2 and by subclass (Fig. 7b; labels for C2's poles instead of MSN/oligodendrocyte)
c  C2 by subclass, ordered by mean (Fig. 7c violin function)
d-f  one row per Xenium section (SECTIONS), each with
   local ependymal density (Fig. 7d recipe: KDE of ependymal cells, bw 0.25 x Scott, min-max)
   C2 projected onto every cell with the snRNA gene weights (Fig. 7f recipe), 30-NN mean
   astrocyte C2: ridge on the 464 Xenium genes fitted within snRNA astrocytes, count-adjusted per section,
   30-NN astrocyte mean (astrocyte_c2_xenium.py; the recipe of Fig. 7 astrocyte panel g)
The MOFA file is opened only for its feature and nucleus index (as fig7_early_figure.py), from fix_s0.
Writes supp_pls_c2.{png,svg}.
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
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.environ["SPATIALJEPA_HUMAN_CAH"])
import cah_components as F  # noqa: E402

D = F.D
HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer"
SECTIONS = [("H21.33.002.CX18.CaH.02.01", "Fig. 7 section"),
            ("H21.33.002.CX18.CaH.02.02", "adjacent to the Fig. 7 section"),
            ("H20.33.026.CX18.CaH.03.02", "")]
RUN, FACTOR, SMOOTH = "seed0", "PLS C2", 30
LABELS = ("Astrocyte", "Ependymal", "OPC", "Oligodendrocyte")
OUT = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/human_cah_transfer/seed0/cah/supp_metabolite_transfer")
FIX_MOFA = f"{os.environ['SPATIALJEPA_HOME']}/outputs/human_cah_transfer/fix_s0/cah/snrna_mofa.h5mu"
XENIUM_PATH = f"{os.environ['SPATIALJEPA_HOME']}/data/SEA_AD/sea_ad_cah/CaH_Xenium_final.2026-01-07.h5ad"

p = F.paths(RUN, "msi", "ot", 0.001, "section")
R = pickle.load(open(p["pls"] / "results.pkl", "rb"))
ns = {"__name__": "fig_c2", "__file__": str(D.DOWNSTREAM_SCRIPT)}
D.run_cells(D.DOWNSTREAM_SCRIPT, ("load data", "load data and MOFA model"), ns, patches={"load data and MOFA model": [
    ('CAH_OUTPUT_ROOT = Path(\n    __import__("os").environ["SPATIALJEPA_HOME"], "outputs/sea_ad_lipid_gps_minimal/cah"\n)',
     f'CAH_OUTPUT_ROOT = Path({str(p["cah"])!r})'),
    ('mofa = mfx.mofa_model(CAH_OUTPUT_ROOT / "snrna_mofa.h5mu")', f"mofa = mfx.mofa_model({FIX_MOFA!r})")]})
real, rna = ns["mofa"], ns["scrna_msi_mudata"].mod["rna"]
weights = {}
for view, block in (("rna", "target RNA"), ("msi", "target MSI (OT)")):     # as cah_components.stage_figure
    idx = real.get_weights(views=[view], df=True).index
    frame = R["W"][block]; frame = frame[~frame.index.duplicated()]
    frame = frame.reindex(idx.str.replace(r"^rna:", "", regex=True) if view == "rna" else idx)
    frame.index = idx
    weights[view] = frame.dropna()
scores = R["scores"].reindex(rna.obs_names.astype(str))
assert not scores.isna().any().any()
ns["mofa"] = F.PLSModel(scores, weights)
ns["mofa_adata"] = sc.AnnData(scores.to_numpy(), obs=ns["scrna_msi_mudata"].mod["msi"].obs, obsm=rna.obsm,
                              var=pd.DataFrame(index=R["comps"]))
cells = ("Reusable Factor 1 multi-panel figure", "Transfer-characterisation figure")
D.run_cells(D.DOWNSTREAM_SCRIPT, cells, ns, defs_only=cells)

w_rna = R["W"]["target RNA"][FACTOR]; w_rna = w_rna[~w_rna.index.duplicated()]
w_rna = w_rna.groupby(w_rna.index.str.upper()).first()


def project_spatial_onto_factor(spatial, *_, key="mofa_factor1", **__):
    """Fig. 7f recipe (cah_components.stage_figure) for C2: snRNA gene weights on per-section z-scored expression."""
    X = spatial.X.toarray() if sp.issparse(spatial.X) else np.asarray(spatial.X, dtype=float)
    Z = (X - X.mean(0)) / X.std(0).clip(1e-6)
    genes = pd.Index(spatial.var_names.astype(str)).str.upper()
    shared = genes.isin(w_rna.index)
    wv = w_rna.reindex(genes[shared]).to_numpy()
    spatial.obs[key] = Z[:, shared] @ wv / np.abs(wv).sum()
    return spatial


ns["project_spatial_onto_factor"] = project_spatial_onto_factor
ns["cah_xenium"] = sc.read_h5ad(XENIUM_PATH)                 # read once; load_xenium_factor_section reuses it
A2 = pickle.load(open(HERE / "astro_c2_xenium.pkl", "rb"))["xa"]["c2_adj"]
sections = []
for sid, note in SECTIONS:
    s = ns["load_xenium_factor_section"](section_id=sid)
    s.obs["dens_epend"] = ns["local_category_density"](s, "Ependymal_Subclass")
    astro = s.obs["Subclass"].astype(str).str.startswith("Astrocyte").to_numpy()
    xy = np.asarray(s.obsm["X_spatial"])[:, :2][astro]
    val = A2.reindex(s.obs_names[astro]).to_numpy()
    assert np.isfinite(val).all()
    _, nb = NearestNeighbors(n_neighbors=SMOOTH + 1).fit(xy).kneighbors(xy)
    n_ep = int((s.obs["Subclass"].astype(str) == "Ependymal_Subclass").sum())
    head = (f"{sid} · {s.obs['Overall AD neuropathological Change'].astype(str).iloc[0]} ADNC"
            + (f" ({note})" if note else "") + f" · {n_ep:,} ependymal cells")
    sections.append((s, xy, val[nb].mean(1), head))

with plt.rc_context(ns["FIG_RC"]):
    n = len(sections)
    fig = plt.figure(figsize=(14, 14.5 * (1.8 + 1.0 * n) / 3.25), layout="constrained")
    sf_top, sf_rows = fig.subfigures(2, 1, height_ratios=[1.8, 1.0 * n])
    top = sf_top.add_gridspec(1, 2, width_ratios=[0.26, 0.74], wspace=0.08)
    ur = top[1].subgridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.05, wspace=0.02)
    axes = {"a": sf_top.add_subplot(top[0]), "b": sf_top.add_subplot(ur[0, 0])}
    ns["plot_factor1_multiomic_weights"](axes["a"], ns["mofa"], factor=FACTOR)
    ad_ = ns["mofa_adata"]
    ns["subclass_color_map"](ad_, "Subclass")
    ax_sub = sf_top.add_subplot(ur[0, 1])
    sc.pl.umap(ad_, color=FACTOR, ax=axes["b"], show=False, title=FACTOR, colorbar_loc="right")
    sc.pl.umap(ad_, color="Subclass", ax=ax_sub, show=False, title="Subclass", legend_loc="on data")
    for ax in (axes["b"], ax_sub):
        for col in ax.collections:
            col.set_rasterized(True)
    for t in list(ax_sub.texts):
        if not any(l in t.get_text() for l in LABELS):
            t.remove()
    axes["c"] = sf_top.add_subplot(ur[1, :])
    ns["plot_factor1_subclass_violin"](axes["c"], ad_, factor=FACTOR)
    for key, ax in axes.items():
        ax.text(-0.02, 1.02, key, transform=ax.transAxes, fontsize=22, fontweight="bold", ha="right", va="bottom")
    row_figs = np.atleast_1d(sf_rows.subfigures(n, 1))
    for r, ((s, xy, a_sm, head), rf) in enumerate(zip(sections, row_figs)):
        rf.suptitle(head, x=0.035, ha="left", fontsize=12)
        rf.text(0.005, 0.995, "def"[r], fontsize=22, fontweight="bold", ha="left", va="top")
        ax_d, ax_e, ax_f = rf.subplots(1, 3)
        ns["plot_spatial_continuous"](ax_d, s, "dens_epend", "local ependymal density",
                                      cmap="magma", diverging=False, smooth=0, size=10.0)
        ns["plot_spatial_continuous"](ax_e, s, "mofa_factor1", f"{FACTOR} ({SMOOTH}-NN mean)", smooth=SMOOTH, size=8.0)
        centre = float(np.median(a_sm)); span = float(np.percentile(np.abs(a_sm - centre), 98))
        h = ax_f.scatter(xy[:, 0], xy[:, 1], c=a_sm, s=8.0, linewidths=0, rasterized=True,
                         cmap="RdBu_r", vmin=centre - span, vmax=centre + span)
        ax_f.set_aspect("equal"); ax_f.set_xticks([]); ax_f.set_yticks([])
        for sp_ in ax_f.spines.values():
            sp_.set_visible(False)
        ax_f.set_title(f"astrocyte C2 ({SMOOTH}-NN mean)", fontsize=14)
        rf.colorbar(h, ax=ax_f, fraction=0.046, pad=0.02).ax.tick_params(labelsize=10)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"supp_pls_c2.{ext}", dpi=300, bbox_inches="tight")
print("wrote", OUT / "supp_pls_c2.png")

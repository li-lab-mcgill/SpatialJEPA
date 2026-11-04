"""Fig. 7 with the astrocyte panels (seed0), laid out on a fixed inch grid.

    python plot_cah_summary.py

a  top C1 RNA and OT-recovered MSI weights (narrower than before; drawn at its size, not squeezed)
b  RNA UMAPs coloured by C1 and by subclass
c  C1 by subclass, ordered by mean
d  donor-mean astrocyte C1 by ADNC (snRNA, 42 donors; OR from astrocyte_c1_adnc.py); spans the b and c rows
e  local oligodendrocyte density        f  transferred dopamine (30-NN mean)
g  projected C1 (30-NN mean)           h  astrocyte C1 (30-NN astrocyte mean; ridge on the 464 Xenium
                                          genes, count-adjusted per section: astro_c1_checks2.py)
i  bivariate Moran's I, smoothed (unsmoothed): the three pairs of the current figure, plus
   astrocyte C1 vs projected C1, computed over the section's astrocytes (W = symmetric binary 6-NN
   among astrocytes; smoothed = both fields as plotted, taken at the astrocytes).
Data path as cah_components.stage_figure (the MOFA file is opened only for its feature and nucleus
index, from fix_s0, as fig7_early_figure.py). One font scheme for every panel (FONT below); panel
letters sit at the top-left corner of each panel's slot, so letters in one row share a baseline
and letters in one column share an x position.
Writes sea_ad_cah_pls_astro.{png,svg} and cah_xenium_bivariate_moran_astro.csv into OUT.
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
HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/figures_pls_msi_ot_reg0.001_sec"
RUN, FACTOR, SMOOTH = "seed0", "PLS C1", 30
SECTION = "H21.33.002.CX18.CaH.02.01"
FIX_MOFA = f"{os.environ['SPATIALJEPA_HOME']}/outputs/human_cah_transfer/fix_s0/cah/snrna_mofa.h5mu"
A = "Overall AD neuropathological Change"
ADNC = ["Not AD", "Low", "Intermediate", "High"]
RAMP = dict(zip(ADNC, ["#c9bfe6", "#9d8ad1", "#6a4fb0", "#3b1f7d"]))       # ordinal: one hue, light -> dark
FONT = dict(title=14, label=13, tick=11.5, legend=11.5, annot=12, letter=22)

# ------------------------------------------------------------------ data (as cah_components.stage_figure)
p = F.paths(RUN, "msi", "ot", 0.001, "section")
OUT = p["figures"]
R = pickle.load(open(p["pls"] / "results.pkl", "rb"))
assert R["c_da"] == FACTOR
ns = {"__name__": "fig7_astro", "__file__": str(D.DOWNSTREAM_SCRIPT)}
D.run_cells(D.DOWNSTREAM_SCRIPT, ("load data", "load data and MOFA model"), ns, patches={"load data and MOFA model": [
    ('CAH_OUTPUT_ROOT = Path(\n    __import__("os").environ["SPATIALJEPA_HOME"], "outputs/sea_ad_lipid_gps_minimal/cah"\n)',
     f'CAH_OUTPUT_ROOT = Path({str(p["cah"])!r})'),
    ('mofa = mfx.mofa_model(CAH_OUTPUT_ROOT / "snrna_mofa.h5mu")', f"mofa = mfx.mofa_model({FIX_MOFA!r})")]})
real, rna = ns["mofa"], ns["scrna_msi_mudata"].mod["rna"]
weights = {}
for view, block in (("rna", "target RNA"), ("msi", "target MSI (OT)")):
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
ns["PRIMARY"] = A
cells = ("Reusable Factor 1 multi-panel figure", "Transfer-characterisation figure",
         "spatial cross-correlation between the three panels")
D.run_cells(D.DOWNSTREAM_SCRIPT, cells, ns, defs_only=cells)

w_rna = R["W"]["target RNA"][FACTOR]; w_rna = w_rna[~w_rna.index.duplicated()]
w_rna = w_rna.groupby(w_rna.index.str.upper()).first()


def project_spatial_onto_factor(spatial, *_, key="mofa_factor1", **__):
    """Fig. 7 recipe (cah_components.stage_figure): snRNA C1 gene weights on per-section z-scored expression."""
    X = spatial.X.toarray() if sp.issparse(spatial.X) else np.asarray(spatial.X, dtype=float)
    Z = (X - X.mean(0)) / X.std(0).clip(1e-6)
    genes = pd.Index(spatial.var_names.astype(str)).str.upper()
    shared = genes.isin(w_rna.index)
    wv = w_rna.reindex(genes[shared]).to_numpy()
    spatial.obs[key] = Z[:, shared] @ wv / np.abs(wv).sum()
    return spatial


ns["project_spatial_onto_factor"] = project_spatial_onto_factor
section = ns["load_xenium_factor_section"](section_id=SECTION)
section = ns["attach_xenium_imputed_msi"](section)
X = section.X.toarray() if sp.issparse(section.X) else np.asarray(section.X, dtype=float)
Z = (X - X.mean(0)) / X.std(0).clip(1e-6)
genes = pd.Index(section.var_names.astype(str)).str.upper()
beta = R["xenium_dopamine_map"]
shared = genes.isin(beta.index)
section.obs["imputed_dopamine"] = Z[:, shared] @ beta.reindex(genes[shared]).to_numpy()
section.obs["dens_oligo"] = ns["local_category_density"](section, "Oligodendrocyte_Subclass")
moran, _ = ns["cah_xenium_spatial_crosscorrelation"](section)

# astrocyte C1 on the section (count-adjusted ridge carry) and its Moran's I against projected C1
astro = section.obs["Subclass"].astype(str).str.startswith("Astrocyte").to_numpy()
xy_all = np.asarray(section.obsm["X_spatial"])[:, :2]
xy = xy_all[astro]
a_raw = pickle.load(open(HERE.parent / "astro_c1_adnc" / "astro_c1_checks2.pkl", "rb"))["c1_adj"].reindex(section.obs_names[astro]).to_numpy()
assert np.isfinite(a_raw).all()
_, nb = NearestNeighbors(n_neighbors=SMOOTH + 1).fit(xy).kneighbors(xy)
a_plot = a_raw[nb].mean(1)                                             # 30-NN astrocyte mean, as panel h
c1_plot = ns["smooth_over_neighbours"](section, "mofa_factor1", n_neighbors=SMOOTH)   # panel g field
fields = {"as plotted": {"astro_c1": a_plot, "projected_factor1": c1_plot[astro],
                         "imputed_dopamine": ns["smooth_over_neighbours"](section, "imputed_dopamine", n_neighbors=SMOOTH)[astro],
                         "oligo_density": section.obs["dens_oligo"].to_numpy(float)[astro]},
          "unsmoothed": {"astro_c1": a_raw, "projected_factor1": section.obs["mofa_factor1"].to_numpy(float)[astro],
                         "imputed_dopamine": section.obs["imputed_dopamine"].to_numpy(float)[astro],
                         "oligo_density": section.obs["dens_oligo"].to_numpy(float)[astro]}}
W = ns["spatial_weights"](xy, n_neighbors=6)
rows = []
for variant, f in fields.items():
    for other in ("projected_factor1", "imputed_dopamine", "oligo_density"):
        I, pval, _ = ns["bivariate_moran_permutation"](f["astro_c1"], f[other], W, 999, 0)
        rows.append(dict(variant=variant, field_a=other, field_b="astro_c1", bivariate_I=I, p_permutation=pval,
                         pearson_r=float(np.corrcoef(f["astro_c1"], f[other])[0, 1]), n_cells=int(astro.sum())))
astro_moran = pd.DataFrame(rows)
print(astro_moran.round(3).to_string(index=False))
moran_all = pd.concat([moran.assign(n_cells=section.n_obs), astro_moran], ignore_index=True)
moran_all.to_csv(OUT / "cah_xenium_bivariate_moran_astro.csv", index=False)

donors = pickle.load(open(HERE.parent / "astro_c1_adnc" / "astro_c1_cache.pkl", "rb"))
don, odds = donors["snrna_donors"], donors["snrna_or"].loc["seed0 all astrocytes"]

# ------------------------------------------------------------------ layout (inches)
FW, M = 14.0, 0.12                                  # figure width, outer margin
COL = (FW - 2 * M) / 4                              # map-row column width; d and h share column 4
X_A, X_R, X_D = M, M + 2.75, M + 3 * COL            # slot left edges: a (narrow); b and c; d and h
LET = 0.40                                          # letter band above each slot
H_B, H_CD, H_MAP, H_I = 2.7, 2.0, 3.0, 1.4         # axes heights (Fig. 7 + caption must fit one page)
GAP_BC, LAB_C, GAP_ROW, GAP_I = 0.35, 1.5, 0.15, 0.05
TTL = 0.32                                          # one-line map titles below the letter band
y = M                                               # running top-down coordinate (inches from the top)
Y_B = y; y += LET + H_B + GAP_BC
Y_C = y; y += LET + H_CD + LAB_C + GAP_ROW
Y_MAP = y; y += LET + TTL + H_MAP + GAP_I
Y_I = y; y += LET + H_I
FH = y + 0.1


def rect(x, ytop, w, h):
    """Inches from the top-left -> figure fraction [left, bottom, width, height]."""
    return [x / FW, 1 - (ytop + h) / FH, w / FW, h / FH]


def letter(fig, key, x, ytop):
    fig.text(x / FW, 1 - ytop / FH, key, fontsize=FONT["letter"], fontweight="bold", ha="left", va="top")


def style(ax, title=None):
    if title is not None:
        ax.set_title(title, fontsize=FONT["title"])
    else:
        ax.title.set_fontsize(FONT["title"])
    ax.xaxis.label.set_fontsize(FONT["label"]); ax.yaxis.label.set_fontsize(FONT["label"])
    ax.tick_params(labelsize=FONT["tick"])
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(FONT["legend"])
        if leg.get_title().get_text():
            leg.get_title().set_fontsize(FONT["legend"])


with plt.rc_context(ns["FIG_RC"]):
    fig = plt.figure(figsize=(FW, FH))
    # a: spans the b and c rows; its x axis ends level with c's
    a_left, a_w = X_A + 1.05, 2.75 - 1.05 - 0.2
    ax_a = fig.add_axes(rect(a_left, Y_B + LET + 0.35, a_w, (Y_C + LET + H_CD) - (Y_B + LET + 0.35)))
    ns["plot_factor1_multiomic_weights"](ax_a, ns["mofa"], factor=FACTOR)
    ax_a.set_xlabel(f"{FACTOR} weight")
    ax_a.set_xticks([-0.5, 0, 0.5])
    ax_a.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.42, 1.0), frameon=False, handletextpad=0.1,
                columnspacing=0.9, borderaxespad=0.15)                # one row, level with b's titles
    style(ax_a)
    letter(fig, "a", X_A, Y_B)

    # b: two UMAPs
    w_umap = (X_D - 0.45 - X_R - 0.8) / 2                # b ends where c ends; d takes the right column
    ax_b1 = fig.add_axes(rect(X_R + 0.15, Y_B + LET + 0.35, w_umap, H_B - 0.35))
    ax_b2 = fig.add_axes(rect(X_R + 0.15 + w_umap + 0.8, Y_B + LET + 0.35, w_umap - 0.15, H_B - 0.35))
    ns["plot_factor1_umaps"](ns["mofa_adata"], ax_b1, ax_b2, factor=FACTOR)
    for ax in (ax_b1, ax_b2):
        style(ax)
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
        for t in ax.texts:
            t.set_fontsize(FONT["annot"])
    letter(fig, "b", X_R, Y_B)

    # c: violin (y label inside the slot)
    ax_c = fig.add_axes(rect(X_R + 0.85, Y_C + LET, X_D - X_R - 0.85 - 0.45, H_CD))
    ns["plot_factor1_subclass_violin"](ax_c, ns["mofa_adata"], factor=FACTOR, label_rotation=60)
    style(ax_c)
    letter(fig, "c", X_R, Y_C)

    # d: astrocyte C1 by ADNC
    d_top = Y_B + LET + 0.35
    ax_d = fig.add_axes(rect(X_D + 0.75, d_top, FW - M - X_D - 0.75 - 0.1, (Y_C + LET + H_CD) - d_top))
    rng = np.random.default_rng(1)
    ticks = []
    for i, lvl in enumerate(ADNC):
        v = don.loc[don[A] == lvl, "C1_seed0"].dropna()
        ax_d.boxplot(v, positions=[i], widths=0.62, patch_artist=True, showfliers=False,
                     boxprops=dict(facecolor=RAMP[lvl], edgecolor="#555555", linewidth=1.0),
                     medianprops=dict(color="#1f1f1f", linewidth=1.6), whiskerprops=dict(color="#555555", linewidth=1.0),
                     capprops=dict(color="#555555", linewidth=1.0))
        ax_d.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=22, color="#1f1f1f", edgecolor="white",
                     linewidth=0.5, zorder=3)
        ticks.append({"Not AD": "Not AD", "Low": "Low", "Intermediate": "Inter.", "High": "High"}[lvl] + f"\n({len(v)})")
    ax_d.set_xticks(range(4), ticks)
    ax_d.set_xlabel("ADNC (donors)")
    ax_d.set_title("donor-mean astrocyte C1")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.text(0.98, 0.99, f"OR {odds.OR:.2f} [{odds.lo:.2f}, {odds.hi:.2f}]\np = {odds.p:.3f}",
              transform=ax_d.transAxes, ha="right", va="top", fontsize=FONT["annot"])
    lo_, hi_ = ax_d.get_ylim()
    ax_d.set_ylim(lo_, hi_ + 0.16 * (hi_ - lo_))
    ax_d.set_yticks([-2.0, -1.5, -1.0, -0.5])
    style(ax_d)
    letter(fig, "d", X_D, Y_B)                          # top row, level with a and b

    # e-h: Xenium maps, identical extent and top-anchored
    map_axes = {}
    pad = 0.04 * np.ptp(xy_all, axis=0)
    lims = (xy_all.min(0) - pad, xy_all.max(0) + pad)
    for k, (key, column, title, kwargs) in enumerate((
            ("e", "dens_oligo", "oligodendrocyte density", dict(cmap="magma", diverging=False, smooth=0, size=10.0)),
            ("f", "imputed_dopamine", "transferred dopamine", dict(smooth=SMOOTH, size=8.0)),   # 30-NN: caption
            ("g", "mofa_factor1", FACTOR, dict(smooth=SMOOTH, size=8.0)),
            ("h", None, "astrocyte C1", None))):
        x0 = M + k * COL
        ax = fig.add_axes(rect(x0 + 0.05, Y_MAP + LET + TTL, COL - 0.6, H_MAP))   # 0.55 in for colourbar labels
        ax.patch.set_visible(False)
        if column is not None:
            ns["plot_spatial_continuous"](ax, section, column, title, title_fontsize=FONT["title"],
                                          cbar_labelsize=FONT["tick"], **kwargs)
        else:
            centre = float(np.median(a_plot)); span = float(np.percentile(np.abs(a_plot - centre), 98))
            hdl = ax.scatter(xy[:, 0], xy[:, 1], c=a_plot, s=8.0, linewidths=0, rasterized=True,
                             cmap="RdBu_r", vmin=centre - span, vmax=centre + span)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_title(title, fontsize=FONT["title"])
            fig.colorbar(hdl, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=FONT["tick"])
        ax.set_xlim(lims[0][0], lims[1][0]); ax.set_ylim(lims[0][1], lims[1][1])
        ax.set_anchor("N")
        map_axes[key] = ax
        letter(fig, key, x0, Y_MAP)

    # i: Moran's I dumbbells under the map centres
    ax_i = fig.add_axes(rect(M, Y_I + LET, FW - 2 * M, H_I))
    fig.canvas.draw()
    centre_x = {k: (lambda b: (b.x0 + b.x1) / 2 * FW)(map_axes[k].get_position()) for k in "efgh"}
    ax_i.set_xlim(M, FW - M)
    pairs = [("oligo_density", "imputed_dopamine", "e", "f"), ("oligo_density", "projected_factor1", "e", "g"),
             ("imputed_dopamine", "projected_factor1", "f", "g"), ("projected_factor1", "astro_c1", "g", "h")]
    value = {(r.variant, r.field_a, r.field_b): r.bivariate_I for r in moran_all.itertuples()}
    text_x = FW - M - 1.0
    for row, (fa, fb, ka, kb) in enumerate(pairs):
        yy = -row
        ax_i.plot([centre_x[ka], centre_x[kb]], [yy, yy], color="k", lw=1.5, zorder=1)
        ax_i.scatter([centre_x[ka], centre_x[kb]], [yy, yy], s=160, color="#AAAAAA", edgecolor="#333333", lw=1.5, zorder=2)
        ax_i.text(text_x, yy, f"{value['as plotted', fa, fb]:+.2f} ({value['unsmoothed', fa, fb]:+.2f})",
                  va="center", ha="center", fontsize=FONT["annot"])
    ax_i.text(text_x, 1.25, "Moran's $I$", ha="center", va="bottom", fontsize=FONT["title"])
    ax_i.text(text_x, 0.55, "smoothed (unsmoothed)", ha="center", va="bottom", fontsize=FONT["tick"])
    ax_i.set_ylim(-len(pairs) + 0.5, 2.0)
    ax_i.set_axis_off()
    letter(fig, "i", M, Y_I)

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"sea_ad_cah_pls_astro.{ext}", dpi=300)
    plt.close(fig)
print("wrote", OUT / "sea_ad_cah_pls_astro.png", f"({FW:.1f} x {FH:.2f} in)")

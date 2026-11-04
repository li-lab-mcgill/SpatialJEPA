"""Astrocyte C1 vs ADNC with the early-stopped teacher (seed0): data for the boxplot and the
Xenium spatial contrast map. Writes astro_c1_cache.pkl next to this script.

snRNA  donor-mean astrocyte C1 (>= 50 nuclei) for seed0 and the fixed seeds; ordinal-logit ORs
       per SD (age, sex, PMI) for all astrocytes, the Astro_2 supertype alone, and all astrocytes
       + donor share of the low-C1 supertypes; check that the 241-gene projection used on Xenium
       (Fig. 7f) tracks the latent C1 within astrocytes.
Xenium the snRNA C1 gene weights applied to z-scored expression (Fig. 7f recipe, but z-scored over
       all 25 sections so donors stay comparable; per-section z-scoring kept as a sensitivity),
       local oligodendrocyte density per section (Fig. 7d KDE), astrocyte kNN-smoothed C1.

Env: $PY_ANALYSIS
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
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm
from scipy.stats import gaussian_kde, pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/astro_c1_adnc"
BASE = Path(os.environ['SPATIALJEPA_HOME'])
RUN = "seed0"
CAH = BASE / f"outputs/human_cah_transfer/{RUN}/cah"
XENIUM_PATH = BASE / "data/SEA_AD/sea_ad_cah/CaH_Xenium_final.2026-01-07.h5ad"
ADNC = ["Not AD", "Low", "Intermediate", "High"]
A = "Overall AD neuropathological Change"
SEEDS = ["fix_s0", "fix_s1", "fix_s2", "seed0"]
LOW_C1_SUPERTYPES = ["Astro_3", "Astro_4"]          # lowest astrocyte C1 in seed0 (-2.7 vs -1.2 to -1.7)

R = pickle.load(open(CAH / "pls_msi_ot_reg0.001_sec/results.pkl", "rb"))
w = R["W"]["target RNA"]
w = w[~w.index.duplicated()][R["c_da"]]
w = w.groupby(w.index.str.upper()).first()
out = {"run": RUN, "c_da": R["c_da"]}

# ---------------------------------------------------------------- snRNA donor level
cells = pd.read_parquet(HERE / "cells.parquet", columns=["Donor ID", "Subclass", "Supertype", A]
                        + [f"C1_{s}" for s in SEEDS])
cells["st"] = cells["Supertype"].astype(str).str.replace("Supertype", "", regex=False)
astro = cells[cells["Subclass"].astype(str) == "Astrocyte"]
donors = pd.read_csv(HERE / "donors.csv", index_col=0)
g = astro.groupby("Donor ID", observed=True)
tab = g[[f"C1_{s}" for s in SEEDS]].mean().where(g.size().ge(50), axis=0)
tab["n_astro"] = g.size()
a2 = astro[astro["st"] == "Astro_2"].groupby("Donor ID", observed=True)
tab["C1_seed0_astro2"] = a2["C1_seed0"].mean().where(a2.size().ge(50))
tab["frac_lowC1_st"] = g["st"].apply(lambda s: s.isin(LOW_C1_SUPERTYPES).mean())
tab = tab.join(donors[["Age at Death", "Sex", "PMI", A]])
tab["ADNC"] = pd.Categorical(tab[A], categories=ADNC, ordered=True)
out["snrna_donors"] = tab
out["snrna_supertype_c1"] = astro.groupby("st")[[f"C1_{s}" for s in SEEDS]].agg(["mean", "size"])
out["snrna_supertype_share"] = (pd.crosstab(astro["Donor ID"], astro["st"], normalize="index")
                                .join(tab["ADNC"]).groupby("ADNC", observed=True).mean())

z = lambda s: (s - s.mean()) / s.std()


def ordinal_or(frame, x, extra=()):
    f = frame.dropna(subset=[x, "Age at Death", "PMI", *extra]).copy()
    X = pd.DataFrame({"x": z(f[x]), "age": z(f["Age at Death"]), "sex_m": (f["Sex"] == "Male").astype(float),
                      "pmi": z(f["PMI"]), **{e: z(f[e]) for e in extra}}, index=f.index)
    y = pd.Series(pd.Categorical(f[A], categories=[l for l in ADNC if l in set(f[A])], ordered=True), index=f.index)
    res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    b, se = res.params["x"], res.bse["x"]
    rho, rho_p = spearmanr(f[x], f[A].map({l: i for i, l in enumerate(ADNC)}))
    return dict(n=len(f), OR=np.exp(b), lo=np.exp(b - 1.96 * se), hi=np.exp(b + 1.96 * se), p=res.pvalues["x"],
                rho=rho, rho_p=rho_p)


ors = {f"{s} all astrocytes": ordinal_or(tab, f"C1_{s}") for s in SEEDS}
ors["seed0 Astro_2 only"] = ordinal_or(tab, "C1_seed0_astro2")
ors["seed0 + low-C1 supertype share"] = ordinal_or(tab, "C1_seed0", extra=("frac_lowC1_st",))
out["snrna_or"] = pd.DataFrame(ors).T
print(out["snrna_or"].round(3).to_string())
print("low-C1 supertype share vs ADNC:", ordinal_or(tab.assign(f=tab["frac_lowC1_st"]), "f"))

# within-astrocyte check of the 241-gene Xenium projection, on the snRNA nuclei
with h5py.File(CAH / "snrna_spatialmeta_aligned.h5mu", "r") as h:
    r = h["mod/rna"]
    genes = pd.Index(r["var/_index"].asstr()[:]).str.replace("rna:", "", regex=False).str.upper()
    obs = pd.Index(r["obs/_index"].asstr()[:])
    Xs = sp.csr_matrix((r["X/data"][:], r["X/indices"][:], r["X/indptr"][:]), shape=(len(obs), len(genes)))
assert obs.equals(pd.Index(cells.index.astype(str))), "aligned h5mu and cells.parquet nuclei differ"
mu = np.asarray(Xs.mean(0), dtype=np.float64).ravel()
sd = np.sqrt(np.maximum(np.asarray(Xs.multiply(Xs).mean(0), dtype=np.float64).ravel() - mu ** 2, 0)).clip(1e-6)
ia = np.flatnonzero((cells["Subclass"].astype(str) == "Astrocyte").to_numpy())
Za = (Xs[ia].toarray() - mu) / sd
panel = pd.Index(ad.read_h5ad(XENIUM_PATH, backed="r").var_names.astype(str)).str.upper()
wv = w.reindex(genes).to_numpy()
on = np.isfinite(wv) & genes.isin(panel)
assert on.sum() > 200, on.sum()
proj_241 = Za[:, on] @ wv[on] / np.abs(wv[on]).sum()
proj_all = Za[:, np.isfinite(wv)] @ wv[np.isfinite(wv)] / np.abs(wv[np.isfinite(wv)]).sum()
c1 = cells["C1_seed0"].to_numpy()[ia]
dm = pd.DataFrame({"c1": c1, "p241": proj_241, "pall": proj_all,
                   "d": cells["Donor ID"].astype(str).to_numpy()[ia]}).groupby("d").mean()
dm = dm[g.size().set_axis(g.size().index.astype(str)).reindex(dm.index).ge(50)]
out["proxy"] = dict(n_shared=int(on.sum()), cell_r_241=pearsonr(c1, proj_241)[0], cell_r_all=pearsonr(c1, proj_all)[0],
                    donor_r_241=pearsonr(dm.c1, dm.p241)[0], donor_r_all=pearsonr(dm.c1, dm.pall)[0])
out["snrna_donors"]["proj241"] = dm["p241"].reindex(tab.index.astype(str)).to_numpy()
out["proj241_or"] = ordinal_or(out["snrna_donors"], "proj241")
assert np.isfinite(proj_241).all() and np.isfinite(proj_all).all()
print("proxy:", {k: round(v, 3) for k, v in out["proxy"].items()}, "\nproj241 OR:", out["proj241_or"])
del Xs, Za

# ---------------------------------------------------------------- Xenium
x = ad.read_h5ad(XENIUM_PATH)
obs = x.obs[["Donor ID", "Specimen_ID", "Subclass", "Supertype", A]].copy()
for c in ("Subclass", "Supertype"):
    obs[c] = obs[c].astype(str).str.replace(f"_{c}", "", regex=False)
xy = np.asarray(x.obsm["spatial"])[:, :2]
X = x.X.tocsr().astype(np.float64)
tot = np.asarray(X.sum(1)).ravel()
X = sp.diags(1e4 / np.where(tot > 0, tot, 1)) @ X
X.data = np.log1p(X.data)
xg = pd.Index(x.var_names.astype(str)).str.upper()
on = xg.isin(w.index)
wv = w.reindex(xg[on]).to_numpy()
L = X[:, np.flatnonzero(on)].toarray()
del X, x


def project(L, idx_groups):
    s = np.empty(len(L))
    for idx in idx_groups:
        m, d = L[idx].mean(0), L[idx].std(0).clip(1e-6)
        s[idx] = np.clip((L[idx] - m) / d, -10, 10) @ wv / np.abs(wv).sum()
    return s


obs["C1_pooled"] = project(L, [np.arange(len(L))])
obs["C1_section"] = project(L, [np.flatnonzero(obs["Specimen_ID"].to_numpy() == s) for s in obs["Specimen_ID"].unique()])
del L
print(f"[xenium] projection over {int(on.sum())} shared genes")

obs["dens_oligo"] = np.nan
obs["C1_astro_smooth"] = np.nan
rng = np.random.default_rng(0)
for s in obs["Specimen_ID"].unique():
    idx = np.flatnonzero(obs["Specimen_ID"].to_numpy() == s)
    sub = obs.iloc[idx]
    train = xy[idx][(sub["Subclass"] == "Oligodendrocyte").to_numpy()]
    if len(train) > 5000:
        train = train[rng.choice(len(train), 5000, replace=False)]
    kde = gaussian_kde(train.T, bw_method=lambda k: 0.25 * k.scotts_factor())
    v = kde(xy[idx].T)
    obs.iloc[idx, obs.columns.get_loc("dens_oligo")] = (v - v.min()) / (np.ptp(v) or 1.0)
    ai = idx[(sub["Subclass"] == "Astrocyte").to_numpy()]
    _, nb = NearestNeighbors(n_neighbors=16).fit(xy[ai]).kneighbors(xy[ai])
    obs.iloc[ai, obs.columns.get_loc("C1_astro_smooth")] = obs["C1_pooled"].to_numpy()[ai][nb].mean(1)
obs["x"], obs["y"] = xy[:, 0], xy[:, 1]
out["xenium"] = obs
xa = obs[obs["Subclass"] == "Astrocyte"]
print("[xenium] astrocyte C1 vs oligodendrocyte density r =",
      round(pearsonr(xa["C1_pooled"], xa["dens_oligo"])[0], 3))
print(xa.groupby("Supertype")[["C1_pooled", "dens_oligo"]].mean().round(3))
print(xa.groupby(["Donor ID", A], observed=True)[["C1_pooled", "C1_section", "dens_oligo"]].mean().round(3))
pickle.dump(out, open(HERE / "astro_c1_cache.pkl", "wb"))
print("wrote", HERE / "astro_c1_cache.pkl")

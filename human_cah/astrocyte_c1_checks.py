"""Checks of the alternative explanations for low astrocyte C1 (seed0) raised in review.

A  snRNA astrocytes: which marker genes move with C1, per nucleus and per donor (reactive, fibrous,
   protoplasmic markers; ambient/contaminating oligodendrocyte, OPC, neuronal and ependymal RNA).
B  Donor level: does the ADNC association survive the donor's tissue composition (oligodendrocyte,
   MSN, ependymal fractions) and ambient oligodendrocyte RNA in astrocytes, and does astrocyte C1
   track caudate histology (GFAP+, AT8+, 6E10+ area)? All three early-stopped seeds.
C  Xenium astrocytes (ridge-carried C1): location (distance to the tissue edge, to the nearest
   ependymal cell = ventricle wall, local oligodendrocyte density) and transcript counts; donor
   means with edge / periventricular astrocytes removed, against snRNA.
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
from scipy import ndimage
from scipy.stats import pearsonr, spearmanr
from sklearn.neighbors import NearestNeighbors
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")
HERE = Path(os.environ["SPATIALJEPA_HOME"]) / "outputs/human_cah_transfer/seed0/cah/astro_c1_adnc"
BASE = Path(os.environ['SPATIALJEPA_HOME'])
ROOT = BASE / "outputs/human_cah_transfer"
A = "Overall AD neuropathological Change"
ADNC = ["Not AD", "Low", "Intermediate", "High"]
z = lambda s: (s - s.mean()) / s.std()

cells = pd.read_parquet(HERE / "cells.parquet", columns=["Donor ID", "Subclass", "C1_seed0"])
sub = cells["Subclass"].astype(str).to_numpy()
donor = cells["Donor ID"].astype(str).to_numpy()
astro = sub == "Astrocyte"
C1 = {r: pickle.load(open(ROOT / f"{r}/cah/pls_msi_ot_reg0.001_sec/results.pkl", "rb"))["scores"]["PLS C1"].to_numpy()
      for r in ("seed0", "seed1", "seed2")}
D = pickle.load(open(HERE / "astro_c1_cache.pkl", "rb"))
tab = D["snrna_donors"].copy(); tab.index = tab.index.astype(str)
don = pd.read_csv(HERE / "donors.csv", index_col=0); don.index = don.index.astype(str)

# ------------------------------------------------------------------ A
GROUPS = {"reactive": ["GFAP", "VIM", "CHI3L1", "SERPINA3", "C3", "CD44", "TNC", "CRYAB", "HSPB1"],
          "protoplasmic": ["SLC1A2", "GJA1", "ALDH1L1", "SPARCL1", "AQP4", "S100B"],
          "C1 ridge genes": ["SLC14A1", "VCAN", "CLU", "CXCL14", "MT1E", "GABRA2"],
          "oligodendrocyte": ["MBP", "PLP1", "MOBP", "ST18"], "OPC": ["PDGFRA", "OLIG2"],
          "neuron/MSN": ["PDE10A"], "ependymal": ["DNAH11"]}
with h5py.File(ROOT / "seed0/cah/snrna_spatialmeta_aligned.h5mu", "r") as h:
    r = h["mod/rna"]
    genes = pd.Index(r["var/_index"].asstr()[:]).str.replace("rna:", "", regex=False)
    Xs = sp.csr_matrix((r["X/data"][:], r["X/indices"][:], r["X/indptr"][:]), shape=(len(cells), len(genes)))
ia = np.flatnonzero(astro)
want = [g for grp in GROUPS.values() for g in grp if g in genes]
E = pd.DataFrame(Xs[ia][:, genes.get_indexer(want)].toarray(), columns=want)
del Xs
c1a = C1["seed0"][ia]
E["_d"] = donor[ia]
dmE = E.groupby("_d").mean()
dmC = pd.Series(c1a).groupby(donor[ia]).mean()
keep = pd.Series(donor[ia]).value_counts().ge(50)
dmE, dmC = dmE[keep.reindex(dmE.index).to_numpy()], dmC[keep.reindex(dmC.index).to_numpy()]
rowsA = []
for grp, gl in GROUPS.items():
    for g in gl:
        if g in E:
            rowsA.append(dict(group=grp, gene=g, mean_expr=E[g].mean(), detect=(E[g] > 0).mean(),
                              r_cell=pearsonr(c1a, E[g])[0], rho_donor=spearmanr(dmC, dmE[g])[0],
                              rho_donor_adnc=spearmanr(dmE[g], tab.loc[dmE.index, A].map({l: i for i, l in enumerate(ADNC)}))[0]))
A_tab = pd.DataFrame(rowsA)
print("A. snRNA astrocytes: marker vs C1 (seed0); r_cell per nucleus, rho_donor over donor means\n",
      A_tab.round(3).to_string(index=False))

# ------------------------------------------------------------------ B
frac = pd.crosstab(donor, sub, normalize="index")
amb = E.groupby("_d")[["MBP", "PLP1"]].mean().mean(1)
cov = pd.DataFrame({"frac_oligo": frac["Oligodendrocyte"], "frac_msn": frac[["STR D1 MSN", "STR D2 MSN", "STR Hybrid MSN"]].sum(1),
                    "frac_epend": frac["Ependymal"], "frac_astro": frac["Astrocyte"], "ambient_oligo_in_astro": amb})


def ordinal(x, extra=()):
    f = tab.join(cov).assign(x=x).dropna(subset=["x", "Age at Death", "PMI", *extra])
    X = pd.DataFrame({"x": z(f["x"]), "age": z(f["Age at Death"]), "sex": (f["Sex"] == "Male").astype(float),
                      "pmi": z(f["PMI"]), **{e: z(f[e]) for e in extra}})
    res = OrderedModel(pd.Series(pd.Categorical(f[A], categories=ADNC, ordered=True), index=f.index), X,
                       distr="logit").fit(method="bfgs", disp=False)
    b, se = res.params["x"], res.bse["x"]
    return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se), res.pvalues["x"]


rowsB = []
for run, c in C1.items():
    s = pd.Series(c[astro]).groupby(donor[astro])
    m = s.mean()[s.size() >= 50]
    for extra in ((), ("frac_oligo",), ("frac_msn",), ("frac_epend",), ("frac_astro",), ("ambient_oligo_in_astro",),
                  ("frac_oligo", "frac_epend", "ambient_oligo_in_astro")):
        o = ordinal(m, extra)
        rowsB.append(dict(run=run, adjust="+".join(extra) or "age, sex, PMI", OR=o[0], lo=o[1], hi=o[2], p=o[3]))
    for col in ("percent GFAP positive area", "percent AT8 positive area", "percent 6e10 positive area",
                "percent NeuN positive area", "CPS"):
        f = pd.concat([m.rename("c1"), don[col]], axis=1).dropna()
        rho, p = spearmanr(f["c1"], f[col])
        rowsB.append(dict(run=run, adjust=f"rho with {col}", OR=rho, lo=np.nan, hi=np.nan, p=p, n=len(f)))
B_tab = pd.DataFrame(rowsB)
print("\nB. donor level (OR per SD, ordinal ADNC; or Spearman rho with histology)\n", B_tab.round(3).to_string(index=False))
print("composition covariates vs ADNC (rho):", {k: round(spearmanr(cov[k].reindex(tab.index), tab[A].map({l: i for i, l in enumerate(ADNC)}), nan_policy="omit")[0], 2) for k in cov})

# ------------------------------------------------------------------ C
RG = pickle.load(open(HERE / "astro_c1_ridge.pkl", "rb"))
ridge = RG["xenium_astro"][next(k for k in RG if k.startswith("Xenium panel"))]
X = D["xenium"]
xo = ad.read_h5ad(BASE / "data/SEA_AD/sea_ad_cah/CaH_Xenium_final.2026-01-07.h5ad", backed="r").obs
X["total_counts"] = pd.to_numeric(xo["total_counts"], errors="coerce").reindex(X.index).to_numpy()
X["d_edge"], X["d_epend"] = np.nan, np.nan
GRID = 20.0                                                  # um per pixel of the tissue mask
for sid, idx in X.groupby("Specimen_ID", observed=True).indices.items():
    xy = X[["x", "y"]].to_numpy()[idx]
    ij = np.floor((xy - xy.min(0)) / GRID).astype(int) + 3
    mask = np.zeros(ij.max(0) + 4, bool); mask[ij[:, 0], ij[:, 1]] = True
    mask = ndimage.binary_fill_holes(ndimage.binary_closing(mask, iterations=3))
    dist = ndimage.distance_transform_edt(mask) * GRID
    X.iloc[idx, X.columns.get_loc("d_edge")] = dist[ij[:, 0], ij[:, 1]]
    ep = (X["Subclass"].to_numpy()[idx] == "Ependymal")
    if ep.sum() >= 20:
        d, _ = NearestNeighbors(n_neighbors=1).fit(xy[ep]).kneighbors(xy)
        X.iloc[idx, X.columns.get_loc("d_epend")] = d[:, 0]
xa = X[X["Subclass"] == "Astrocyte"].copy()
xa["c1"] = ridge.reindex(xa.index).to_numpy()
print(f"\nC. Xenium astrocytes {len(xa):,}; coordinate span per section ~{np.median(X.groupby('Specimen_ID', observed=True)['x'].agg(np.ptp)):.0f} units")
for col in ("d_edge", "d_epend", "dens_oligo", "total_counts"):
    f = xa[["c1", col]].dropna()
    print(f"  rho(C1, {col}) = {spearmanr(f['c1'], f[col])[0]:+.3f}  (n {len(f):,})")
xa["edge"] = xa["d_edge"] < 200
xa["periv"] = xa["d_epend"] < 500
for lab, m in (("< 200 um from the tissue edge", xa["edge"]), ("< 500 um from an ependymal cell", xa["periv"].fillna(False)),
               ("interior (neither)", ~xa["edge"] & ~xa["periv"].fillna(False))):
    print(f"  {lab}: {m.mean():.1%} of astrocytes, mean C1 {xa.loc[m, 'c1'].mean():+.3f}")
q = pd.qcut(xa["c1"], 5, labels=["Q1 low", "Q2", "Q3", "Q4", "Q5 high"])
print("  by C1 quintile:\n", xa.groupby(q, observed=True).agg(edge=("edge", "mean"), periv=("periv", "mean"),
                                                            dens_oligo=("dens_oligo", "mean"), counts=("total_counts", "median")).round(3).to_string())
import statsmodels.formula.api as smf
f = xa.dropna(subset=["c1", "d_edge", "dens_oligo", "total_counts"]).copy()
f["log_edge"], f["log_counts"] = np.log1p(f["d_edge"]), np.log(f["total_counts"])
f["periv"] = f["periv"].fillna(False).astype(float)
for form in ("c1 ~ log_edge", "c1 ~ periv", "c1 ~ dens_oligo", "c1 ~ log_counts",
             "c1 ~ log_edge + periv + dens_oligo + log_counts", "c1 ~ C(Specimen_ID)",
             "c1 ~ log_edge + periv + dens_oligo + log_counts + C(Specimen_ID)"):
    print(f"  R2 {smf.ols(form, f).fit().rsquared:.3f}  {form}")
snr = D["snrna_donors"]["C1_seed0"]; snr.index = snr.index.astype(str)
for lab, m in (("all astrocytes", np.ones(len(xa), bool)), ("interior only", (~xa["edge"] & ~xa["periv"].fillna(False)).to_numpy())):
    dm = xa[m].groupby("Donor ID", observed=True)["c1"].mean(); dm.index = dm.index.astype(str)
    adnc = xa.groupby("Donor ID", observed=True)[A].first(); adnc.index = adnc.index.astype(str)
    print(f"  Xenium donor means, {lab}: rho vs snRNA {spearmanr(dm, snr.reindex(dm.index))[0]:+.2f}, "
          f"vs ADNC {spearmanr(dm, adnc.reindex(dm.index).map({l: i for i, l in enumerate(ADNC)}))[0]:+.2f}")
sec = "H21.33.002.CX18.CaH.02.01"
s = xa[xa["Specimen_ID"] == sec]
print(f"  Fig. 7 section: rho(C1, dens_oligo) {spearmanr(s['c1'], s['dens_oligo'])[0]:+.2f}, rho(C1, d_edge) {spearmanr(s['c1'], s['d_edge'])[0]:+.2f}, "
      f"ependymal cells {int((X['Specimen_ID'].eq(sec) & X['Subclass'].eq('Ependymal')).sum())}")
pickle.dump(dict(A=A_tab, B=B_tab, xa=xa[["Donor ID", "Specimen_ID", A, "c1", "d_edge", "d_epend", "dens_oligo", "total_counts"]]),
            open(HERE / "astro_c1_checks.pkl", "wb"))
A_tab.to_csv(BASE / "outputs/human_cah_transfer/seed0/cah/astro_c1_adnc/checks_A_markers.csv", index=False)
B_tab.to_csv(BASE / "outputs/human_cah_transfer/seed0/cah/astro_c1_adnc/checks_B_donor.csv", index=False)

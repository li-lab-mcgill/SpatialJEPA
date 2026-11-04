try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% load data
from pathlib import Path

import muon as mu
import scanpy as sc
import numpy as np
import pandas as pd
from anndata import AnnData
import mofax as mfx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import scanpy._utils as sc_utils
import scipy.sparse as sp

def get_sparse_from_igraph_compat(graph, weight_attr=None):
    edges = graph.get_edgelist()
    shape = (graph.vcount(), graph.vcount())
    if len(edges) == 0:
        return csr_matrix(shape)

    weights = np.ones(len(edges)) if weight_attr is None else graph.es[weight_attr]
    rows, cols = zip(*edges)
    return csr_matrix((weights, (np.asarray(rows), np.asarray(cols))), shape=shape)

sc_utils.get_sparse_from_igraph = get_sparse_from_igraph_compat

def aggregate(adata, by, func):
    if func != "mean":
        raise NotImplementedError("This workaround supports func='mean' only")
    labels = adata.obs[by]
    groups = labels.dropna().unique()
    means = np.vstack([
        np.asarray(adata.X[labels.to_numpy() == group].mean(axis=0)).ravel()
        for group in groups
    ])
    stds = np.vstack([
        np.asarray(
            (adata.X[labels.to_numpy() == group].toarray()
             if sp.issparse(adata.X)
             else np.asarray(adata.X[labels.to_numpy() == group])
             ).std(axis=0)
        ).ravel()
        for group in groups
    ])
    ns = np.array([(labels.to_numpy() == group).sum() for group in groups], dtype=float)
    sems = stds / np.sqrt(ns)[:, None]
    result = AnnData(
        X=means.copy(),
        obs=pd.DataFrame({by: groups, "n": ns}, index=groups.astype(str)),
        var=adata.var.copy(),
    )
    result.layers["mean"] = means
    result.layers["std"] = stds
    result.layers["sem"] = sems
    return result

def plot_subclass_dpt_trends(
    adata,
    subclass,
    genes,
    *,
    subclass_key="Subclass",
    dpt_key="dpt_pseudotime",
    q=5,
    figsize=(6, 10),
    composites=None,
    bin_key=None,
    plot_umap=False,
    leiden_key="leiden",
    leiden_groups=None,
    umap_genes=None,
    umap_size=80,
    umap_ncols=3,
):
    """Bin subclass cells by DPT, aggregate gene means, and plot SEM errorbars.

    Parameters
    ----------
    genes : list[str] | list[list[str]]
        One gene panel, or multiple panels (each becomes its own figure).
    composites : list[tuple[list[str], str]] | None
        Optional averaged gene panels as (gene_list, ylabel).
    plot_umap : bool
        If True, plot gene UMAPs on subclass cells (optionally filtered by leiden).
    leiden_groups : sequence | None
        If given, further restrict the UMAP to these leiden clusters.
    umap_genes : list[str] | None
        Genes for the UMAP; defaults to flattening `genes`.
    """
    if genes and isinstance(genes[0], str):
        gene_panels = [genes]
    else:
        gene_panels = list(genes)

    mask = adata.obs[subclass_key].eq(subclass)
    if bin_key is None:
        safe = "".join(ch if ch.isalnum() else "_" for ch in subclass).lower()
        bin_key = f"{safe}_dpt_bins"

    bins = pd.qcut(adata.obs.loc[mask, dpt_key], q=q)
    adata.obs[bin_key] = bins.reindex(adata.obs_names)

    sub = adata[mask].copy()
    agg = aggregate(sub, by=bin_key, func="mean")
    mean_df = agg.to_df().sort_index()
    sem_df = pd.DataFrame(
        agg.layers["sem"],
        index=agg.obs_names,
        columns=agg.var_names,
    ).loc[mean_df.index]

    x = range(len(mean_df))
    for panel in gene_panels:
        fig, axs = plt.subplots(len(panel), 1, figsize=figsize, sharex=True)
        axs = np.atleast_1d(axs)
        for ax, gene in zip(axs, panel):
            ax.errorbar(
                x,
                mean_df[gene],
                yerr=sem_df[gene],
                marker="o",
                capsize=3,
            )
            ax.set_ylabel(gene)
            ax.set_xticks(list(x))
            ax.set_xticklabels(mean_df.index, rotation=45, ha="right")
        axs[-1].set_xlabel(bin_key)
        fig.suptitle(f"{subclass} ({bin_key})", y=1.01)
        plt.tight_layout()

    if composites:
        for comp_genes, ylabel in composites:
            comp_mean = mean_df[comp_genes].mean(axis=1)
            comp_sem = np.sqrt((sem_df[comp_genes] ** 2).mean(axis=1))
            fig, ax = plt.subplots()
            ax.errorbar(x, comp_mean, yerr=comp_sem, marker="o", capsize=3)
            ax.set_xticks(list(x))
            ax.set_xticklabels(mean_df.index, rotation=45, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_xlabel(bin_key)
            ax.set_title(subclass)
            plt.tight_layout()

    if plot_umap:
        umap_mask = mask.copy()
        if leiden_groups is not None:
            umap_mask = umap_mask & adata.obs[leiden_key].isin(leiden_groups)
        if umap_genes is None:
            umap_genes = [g for panel in gene_panels for g in panel]
        fig = sc.pl.umap(
            adata[umap_mask],
            color=umap_genes,
            size=umap_size,
            ncols=umap_ncols,
            return_fig=True,
            show=False,
        )
        n_cells = int(umap_mask.sum())
        if leiden_groups is not None:
            leiden_str = ",".join(map(str, leiden_groups))
            title = f"{subclass} gene UMAP (leiden ∈ MSN-oligo axis)"
        else:
            title = f"{subclass} gene UMAP (n={n_cells})"
        fig.suptitle(title, y=1.02)
        plt.tight_layout()
        plt.show()

    return mean_df, sem_df, agg


def ordered_violin(adata, keys, groupby):
    try:
        aggregated = sc.get.aggregate(adata, by=groupby, func="mean")
    except:
        sc.get.aggregate = aggregate
        aggregated = sc.get.aggregate(adata, by=groupby, func="mean")

    dopamine_ix = list(aggregated.var_names).index(keys)
    rank = aggregated.layers["mean"][:, dopamine_ix].ravel().argsort()[::-1]
    ordered_groups = aggregated.obs[groupby].iloc[rank].tolist()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 4))
    sc.pl.violin(
        adata,
        keys=keys,
        groupby=groupby,
        order=ordered_groups,
        log=True,
        stripplot=False,
        rotation=90,
        ax=ax,
    )
    return ordered_groups

def graph_mean(adata, values, connectivity_key="connectivities"):
    """Compute a weighted mean of values over each observation's neighbors."""
    W = adata.obsp[connectivity_key].tocsr(copy=True)

    # Exclude the focal cell
    W.setdiag(0)
    W.eliminate_zeros()

    row_sums = np.asarray(W.sum(axis=1)).ravel()

    inv_row_sums = np.divide(
        1.0,
        row_sums,
        out=np.zeros_like(row_sums, dtype=float),
        where=row_sums != 0,
    )

    W_normalized = sp.diags(inv_row_sums) @ W
    return np.asarray(W_normalized @ values).ravel()

#%% load data and MOFA model

CAH_OUTPUT_ROOT = Path(
    __import__("os").environ["SPATIALJEPA_HOME"], "outputs/sea_ad_lipid_gps_minimal/cah"
)
SPATIALMETA_SNRNA_DIR = CAH_OUTPUT_ROOT / "spatialmeta_handoff" / "outputs" / "snrna"

scrna_msi_mudata = mu.read_h5mu(CAH_OUTPUT_ROOT / "snrna_spatialmeta_aligned.h5mu")
mofa = mfx.mofa_model(CAH_OUTPUT_ROOT / "snrna_mofa.h5mu")

## load spatialMETA embeddings
rna = scrna_msi_mudata.mod["rna"]
rna.obs['mofa_factor1'] = mofa.get_factors(factors=[0])
assert rna.obs_names.equals(mofa.get_cells().set_index('cell').index)

donor_ids = [str(d) for d in scrna_msi_mudata.uns["donor_ids"]]
obs_names_all = []
teacher_parts = []
student_parts = []

for donor_id in donor_ids:
    npz_path = SPATIALMETA_SNRNA_DIR / donor_id / "embeddings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing spatialMETA embeddings for {donor_id}: {npz_path}")
    payload = np.load(npz_path, allow_pickle=False)
    obs_names_all.append(payload["obs_names"].astype(str))
    teacher_parts.append(np.asarray(payload["teacher"], dtype=np.float32))
    student_parts.append(np.asarray(payload["student"], dtype=np.float32))

emb_index = pd.Index(np.concatenate(obs_names_all))
order = emb_index.get_indexer(rna.obs_names.astype(str))
if (order < 0).any():
    raise RuntimeError(
        f"spatialMETA embeddings missing for {(order < 0).sum():,}/{rna.n_obs:,} RNA cells."
    )

rna.obsm["X_spatialmeta_teacher"] = np.vstack(teacher_parts)[order]
rna.obsm["X_spatialmeta_student"] = np.vstack(student_parts)[order]
print(
    "Attached spatialMETA embeddings to rna.obsm: "
    f"teacher={rna.obsm['X_spatialmeta_teacher'].shape}, "
    f"student={rna.obsm['X_spatialmeta_student'].shape}"
)

## clean-up Subclass labels by dropping the "Subclass" suffix
scrna_msi_mudata.mod["msi"].obs["Subclass"] = scrna_msi_mudata.mod["msi"].obs["Subclass"].str.replace("Subclass", "")
scrna_msi_mudata.mod["rna"].obs["Subclass"] = scrna_msi_mudata.mod["rna"].obs["Subclass"].str.replace("Subclass", "")

# %% MOFA weights for Factor1

assert 'msi:Dopamine' in mofa.get_top_features(factors=[0])

mofa_msi_weights_df = mofa.get_weights(df=True, views=['msi'])
mofa_rna_weights_df = mofa.get_weights(df=True, views=['rna'])

mofa_msi_weights_df = mofa_msi_weights_df.sort_values(by='Factor1', ascending=False)
mofa_rna_weights_df = mofa_rna_weights_df.sort_values(by='Factor1', ascending=False)

mofa_top_msi_weights_df = pd.concat([mofa_msi_weights_df.head(5), mofa_rna_weights_df.tail(5)])
mofa_top_rna_weights_df = pd.concat([mofa_rna_weights_df.head(5), mofa_msi_weights_df.tail(5)])
mofa_top_weights_df = pd.concat([
    mofa_top_msi_weights_df.assign(View='msi'),
    mofa_top_rna_weights_df.assign(View='rna')
]).sort_values(by='Factor1', ascending=False)

plot_df = mofa_top_weights_df.reset_index()
plot_df['index'] = plot_df['index'].str.replace('msi:', '').str.replace('rna:', '')
plot_df['View'] = plot_df['View'].str.upper()

# Map views to specific colors
colors = {'MSI': '#1f77b4', 'RNA': '#ff7f0e'}
plot_df['color'] = plot_df['View'].map(colors)

fig, ax = plt.subplots(figsize=(4, 6))

# 1. Draw the horizontal lines from 0 to the weight value
ax.hlines(
    y=plot_df['index'], 
    xmin=0, 
    xmax=plot_df['Factor1'], 
    color='lightgray', 
    linewidth=2,
    zorder=1
)

# 2. Draw the colored marker at the end of the line
for view, group in plot_df.groupby('View'):
    ax.scatter(
        group['Factor1'], 
        group['index'], 
        color=colors[view], 
        s=100,             
        label=view, 
        edgecolors='white', 
        linewidths=1,
        zorder=2
    )

# --- THE FIX ---
# Invert the y-axis so the top of the dataframe appears at the top of the plot
ax.invert_yaxis()
# ---------------

# Clean-up details
ax.axvline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False) 

ax.legend(title='View', frameon=True, facecolor='white')
plt.xlabel('Factor 1 Weight')
plt.ylabel('')
plt.tight_layout()
plt.show()

# %% MOFA factors and Subclass labels, UMAP

mofa_adata = sc.AnnData(
    mofa.get_factors(),
    obs=scrna_msi_mudata.mod["msi"].obs,
    obsm = scrna_msi_mudata.mod["rna"].obsm,
    var=mofa.get_weights(df=True).columns.to_frame()
)

# Plot Factor1 and every Subclass, but retain only MSN and Oligodendrocyte labels on the data
ax = sc.pl.umap(
    mofa_adata,
    color=['Factor1', 'Subclass'],
    legend_loc='on data',
    show=False,
)
for label in list(ax[-1].texts):
    if 'MSN' not in label.get_text() and 'Oligodendrocyte' not in label.get_text():
        label.remove()
plt.show()

#%% ordered violin plots for dopamine and MOFA Factor1

ordered_violin(scrna_msi_mudata.mod["msi"], keys="msi:Dopamine", groupby="Subclass")
ordered_violin(mofa_adata, keys="Factor1", groupby="Subclass")

#%% Import continuous pseudo-progression scores (CPS) from Kana 2026

kana2026_cps_scores = pd.read_excel(
    f"{__import__('os').environ['SPATIALJEPA_HOME']}/data/SEA_AD/6a457afa3e9ab7e81cabccb1_sea-ad_global_and_local_cps_050126.xlsx",
    sheet_name="Kana 2026"
)

CPS_OUTCOMES = ["CPS", "CPS_AT8", "CPS_6E10"]
kana2026_cps_scores["Donor ID"] = kana2026_cps_scores["Donor ID"].astype(str)
missing_cps_cols = sorted(set(["Donor ID", *CPS_OUTCOMES]) - set(kana2026_cps_scores.columns))
if missing_cps_cols:
    raise KeyError(f"Kana 2026 CPS sheet is missing columns: {missing_cps_cols}")
if kana2026_cps_scores["Donor ID"].duplicated().any():
    duplicated = kana2026_cps_scores.loc[
        kana2026_cps_scores["Donor ID"].duplicated(), "Donor ID"
    ].tolist()
    raise ValueError(f"Kana 2026 CPS sheet has duplicated Donor ID values: {duplicated}")
for _outcome in CPS_OUTCOMES:
    kana2026_cps_scores[_outcome] = pd.to_numeric(
        kana2026_cps_scores[_outcome],
        errors="coerce",
    )
if kana2026_cps_scores[CPS_OUTCOMES].isna().any().any():
    raise ValueError("Kana 2026 CPS sheet has missing/non-numeric CPS values.")
kana2026_cps_by_donor = kana2026_cps_scores.set_index("Donor ID")[CPS_OUTCOMES]

scrna_msi_mudata.obs["CPS"] = scrna_msi_mudata.obs["rna:Donor ID"].map(kana2026_cps_by_donor["CPS"])
scrna_msi_mudata.obs["CPS_AT8"] = scrna_msi_mudata.obs["rna:Donor ID"].map(kana2026_cps_by_donor["CPS_AT8"])
scrna_msi_mudata.obs["CPS_6E10"] = scrna_msi_mudata.obs["rna:Donor ID"].map(kana2026_cps_by_donor["CPS_6E10"])

mu.pl.embedding(scrna_msi_mudata, basis='rna:X_umap', color=['CPS', 'CPS_AT8', 'CPS_6E10'])

# %% KEY ANALYSIS — dopamine and the dopamine-enriched Factor1 are altered in the CaH in AD
# ----------------------------------------------------------------------------------------
# Message: within the caudate head (CaH), the dopamine signal — and MOFA Factor1, the
# multi-omic factor that msi:Dopamine loads onto positively (+0.61; see assertion above) —
# decline with AD neuropathology, specifically in the Factor-1-positive medium spiny neurons
# (MSNs), and not in the Factor-1-negative cell types.
#
# Design (avoids pseudoreplication — every neuropathology label is one value per donor):
# aggregate the per-cell predictor to the donor level *within each Subclass*, then fit one
# ordinal (cumulative-logit) model per Subclass across the ~42 donors. Predictor + continuous
# covariates are z-scored, so the slope is a per-SD log-odds effect and OR < 1 means
# "more dopamine / Factor1 -> lower AD stage". We adjust for age/sex/PMI and corroborate each
# OR with an assumption-light donor-level Spearman correlation.

from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr

MOFA_FACTOR_DF = mofa.get_factors(df=True)
MOFA_FACTOR_NAMES = list(MOFA_FACTOR_DF.columns)

# Two per-cell readouts of the caudate dopamine axis (both aligned to msi.obs).
PREDICTORS = {
    "Dopamine": np.asarray(scrna_msi_mudata.mod["msi"][:, "msi:Dopamine"].X).ravel(),
    "Factor1":  MOFA_FACTOR_DF["Factor1"].values,
}
# Donor-level ordinal AD-staging axes (amyloid/tau/global). CERAD (neuritic-plaque density)
# was also tested and carried no association, so it is dropped from the focused analysis.
OUTCOMES = {
    "Overall AD neuropathological Change": ["Not AD", "Low", "Intermediate", "High"],
    "Braak": ["Braak 0", "Braak I", "Braak II", "Braak III",
              "Braak IV", "Braak V", "Braak VI"],
    "Thal":  ["Thal 0", "Thal 1", "Thal 2", "Thal 3", "Thal 4", "Thal 5"],
}
PRIMARY = "Overall AD neuropathological Change"

# Caudate cell types (Subclass suffix already stripped above): the Factor-1-positive MSNs
# vs Factor-1-negative cell types (low/negative Factor1 loading).
CELLTYPE_GROUPS = {
    "Factor-1 positive": ["STR D1 MSN", "STR D2 MSN", "STR Hybrid MSN"],
    "Factor-1 negative": ["Ependymal", "Microglia-PVM", "Oligodendrocyte"],
}
MIN_DONORS = 20  # for the unbiased scan; the target/control cell types all clear this

# Per-cell table: both predictors + donor-level metadata/outcomes.
msi_obs = scrna_msi_mudata.mod["msi"].obs
cell_df = pd.DataFrame({
    "donor": msi_obs["Donor ID"].astype(str).values,
    "subclass": msi_obs["Subclass"].astype(str).values,
    "age": pd.to_numeric(msi_obs["Age at Death"], errors="coerce").values,
    "sex": msi_obs["Sex"].astype(str).values,
    "pmi": pd.to_numeric(msi_obs["PMI"].astype(str), errors="coerce").values,
})
for _name, _vals in PREDICTORS.items():
    cell_df[_name] = np.asarray(_vals).ravel()
for _factor in MOFA_FACTOR_NAMES:
    if _factor not in cell_df:
        cell_df[_factor] = MOFA_FACTOR_DF[_factor].values
for _outcome in OUTCOMES:
    cell_df[_outcome] = msi_obs[_outcome].astype(str).values
for _outcome in CPS_OUTCOMES:
    cell_df[_outcome] = cell_df["donor"].map(kana2026_cps_by_donor[_outcome])

cps_overlap_donors = sorted(set(cell_df["donor"]) & set(kana2026_cps_by_donor.index))
missing_cps_donors = sorted(set(cell_df["donor"]) - set(kana2026_cps_by_donor.index))
print(
    f"\nKana 2026 CPS donor overlap: {len(cps_overlap_donors)} / "
    f"{cell_df['donor'].nunique()} CaH donors."
)
if missing_cps_donors:
    print(f"CaH donors without Kana CPS: {', '.join(missing_cps_donors)}")


def donor_level(subclass_name, outcome):
    """Donor-level table for one Subclass: donor-mean of each predictor + first-of
    covariates/outcome, keeping donors with the outcome observed and covariates present."""
    g = cell_df[cell_df["subclass"] == subclass_name]
    agg = g.groupby("donor").agg({
        **{p: "mean" for p in PREDICTORS}, outcome: "first",
        "age": "first", "sex": "first", "pmi": "first",
    }).reset_index()
    return agg[agg[outcome].isin(OUTCOMES[outcome])].dropna(subset=["age", "pmi"])


def fit_ordinal(agg, predictor, outcome, covariates=("age", "sex", "pmi")):
    """Cumulative-logit <outcome> ~ z(predictor) (+ z-covariates) across donors.
    Returns (coef, pval, n, ci_low, ci_high, se) for the predictor slope (log-odds),
    or None if it can't fit."""
    exog = pd.DataFrame(index=agg["donor"].values)
    exog["pred"] = (agg[predictor].values - agg[predictor].mean()) / agg[predictor].std()
    for cov in covariates:
        col = agg[cov]
        if cov == "sex":
            exog["sex_Male"] = (col.values == "Male").astype(float)
        else:
            if col.std() == 0:
                return None
            exog[cov] = (col.values - col.mean()) / col.std()
    if exog["pred"].std() == 0 or exog.isna().any().any():
        return None
    # Drop empty ordinal levels (an unobserved level makes its threshold unidentifiable).
    present = [c for c in OUTCOMES[outcome] if c in set(agg[outcome].values)]
    y = pd.Series(pd.Categorical(agg[outcome].values, categories=present, ordered=True),
                  index=agg["donor"].values)
    try:
        res = OrderedModel(y, exog, distr="logit").fit(method="bfgs", disp=False)
    except Exception:
        return None
    ci = res.conf_int()
    lo, hi = (ci.loc["pred"] if hasattr(ci, "loc")
              else ci[list(res.params.index).index("pred")])
    return (res.params["pred"], res.pvalues["pred"], len(agg),
            float(lo), float(hi), float(res.bse["pred"]))


def spearman_trend(agg, predictor, outcome):
    """Assumption-light donor-level rank correlation of predictor vs ordinal stage."""
    present = [c for c in OUTCOMES[outcome] if c in set(agg[outcome].values)]
    codes = pd.Categorical(agg[outcome].values, categories=present, ordered=True).codes
    return spearmanr(agg[predictor].values, codes)


pd.set_option("display.width", 240, "display.max_columns", 30)

# --- HEADLINE: Factor-1-positive vs Factor-1-negative cell types, per predictor (OR_adj + rho) ---
for predictor in PREDICTORS:
    focus_rows = []
    for group, subclasses in CELLTYPE_GROUPS.items():
        for sc_name in subclasses:
            row = {"group": group, "subclass": sc_name,
                   "n": len(donor_level(sc_name, PRIMARY))}
            for outcome in OUTCOMES:
                agg = donor_level(sc_name, outcome)
                fit = fit_ordinal(agg, predictor, outcome)
                rho, _ = spearman_trend(agg, predictor, outcome)
                row[f"{outcome[:5]}_OR"] = round(np.exp(fit[0]) if fit else np.nan, 3)
                row[f"{outcome[:5]}_rho"] = round(rho, 3)
            focus_rows.append(row)
    print(f"\n=== {predictor} in CaH: Factor-1-positive vs Factor-1-negative cell types "
          f"(OR_adj + donor-level Spearman rho) ===")
    print(pd.DataFrame(focus_rows).to_string(index=False))

# --- Raw direction/magnitude: predictor vs ADNC stage in the D1 MSN (direct pathway) ---
d1 = donor_level("STR D1 MSN", PRIMARY)
present = [c for c in OUTCOMES[PRIMARY] if c in set(d1[PRIMARY].values)]
d1["stage"] = pd.Categorical(d1[PRIMARY], categories=present, ordered=True)
print("\n=== STR D1 MSN: donor-mean predictor by ADNC stage ===")
for predictor in PREDICTORS:
    trend = d1.groupby("stage", observed=True)[predictor].agg(["mean", "sem", "size"])
    rho, p = spearmanr(d1[predictor].values, d1["stage"].cat.codes.values)
    print(f"\n-- {predictor} (Spearman rho={rho:.3f}, p={p:.3f}) --")
    print(trend.round(4).to_string())

# %% Supplementary: unbiased per-Subclass scan on the primary outcome (ADNC), both predictors.
# Confirms the MSN signal is not an artifact of pre-selecting cell types; BH-corrected across
# Subclasses. (Braak/Thal for the cells of interest are in the headline above.)
for predictor in PREDICTORS:
    scan = []
    for sc_name, g in cell_df.groupby("subclass"):
        agg = donor_level(sc_name, PRIMARY)
        if agg["donor"].nunique() < MIN_DONORS:
            continue
        fit = fit_ordinal(agg, predictor, PRIMARY)
        if fit is None:
            continue
        scan.append(dict(subclass=sc_name, n=fit[2],
                         OR_adj=np.exp(fit[0]), p_adj=fit[1]))
    scan = pd.DataFrame(scan)
    scan["p_adj_BH"] = multipletests(scan["p_adj"], method="fdr_bh")[1]
    scan = scan.sort_values("p_adj").reset_index(drop=True)
    print(f"\n=== Unbiased scan — {predictor} vs ADNC across all Subclasses (adj.) ===")
    print(scan.round(4).to_string(index=False))

# %% CPS analysis — continuous Kana 2026 donor-level pathology scores
# -------------------------------------------------------------------
# CPS is continuous and donor-level, so the analysis mirrors the ordinal ADNC design above:
# aggregate MOFA factors to one value per donor within each Subclass, then fit donor-level
# models. The primary family is Factor1 in the three Factor-1-positive MSN subclasses
# against CPS/CPS_AT8/CPS_6E10.

def _zscore(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    sd = values.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    return (values - values.mean()) / sd


def donor_level_cps(subclass_name, predictors=None, outcomes=None):
    """Donor-level table for continuous CPS analysis within one Subclass."""
    predictors = list(dict.fromkeys(predictors or MOFA_FACTOR_NAMES))
    outcomes = list(dict.fromkeys(outcomes or CPS_OUTCOMES))
    g = cell_df[cell_df["subclass"] == subclass_name]
    agg = g.groupby("donor").agg({
        **{p: "mean" for p in predictors},
        **{o: "first" for o in outcomes},
        "age": "first",
        "sex": "first",
        "pmi": "first",
    }).reset_index()
    return agg.dropna(subset=["age", "pmi"])


def fit_cps_ols(agg, predictor, outcome):
    """OLS <continuous CPS> ~ z(predictor) + z(age) + sex + z(PMI), HC3 robust SE."""
    data = agg[["donor", predictor, outcome, "age", "sex", "pmi"]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna(subset=[predictor, outcome, "age", "sex", "pmi"])
    if data["donor"].nunique() < MIN_DONORS:
        return None

    pred_z = _zscore(data[predictor])
    age_z = _zscore(data["age"])
    pmi_z = _zscore(data["pmi"])
    if pred_z is None or age_z is None or pmi_z is None:
        return None

    exog = pd.DataFrame({
        "pred": pred_z.to_numpy(),
        "age": age_z.to_numpy(),
        "sex_Male": data["sex"].astype(str).eq("Male").astype(float).to_numpy(),
        "pmi": pmi_z.to_numpy(),
    }, index=data["donor"].values)
    if exog.isna().any().any():
        return None
    exog = sm.add_constant(exog, has_constant="add")
    y = data[outcome].to_numpy(dtype=float)
    try:
        res = sm.OLS(y, exog).fit(cov_type="HC3")
    except Exception:
        return None
    ci_low, ci_high = res.conf_int().loc["pred"]
    return {
        "n": int(data["donor"].nunique()),
        "beta": float(res.params["pred"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(res.pvalues["pred"]),
        "adj_r2": float(res.rsquared_adj),
    }


def spearman_cps(agg, predictor, outcome):
    data = agg[[predictor, outcome]].copy()
    data[outcome] = pd.to_numeric(data[outcome], errors="coerce")
    data = data.dropna()
    if len(data) < MIN_DONORS or data[predictor].nunique() <= 1 or data[outcome].nunique() <= 1:
        return np.nan, np.nan
    rho, p_value = spearmanr(data[predictor].values, data[outcome].values)
    return float(rho), float(p_value)


def add_bh_qvalues(results):
    results = results.copy()
    if results.empty:
        results["q_value"] = []
        return results
    results["q_value"] = multipletests(results["p_value"], method="fdr_bh")[1]
    return results.sort_values(["q_value", "p_value", "subclass", "factor", "outcome"]).reset_index(drop=True)


msi_dopamine_loadings = mofa.get_weights(df=True, views=["msi"]).reindex(columns=MOFA_FACTOR_NAMES)
factor_dopamine_loading = {
    factor: (
        float(msi_dopamine_loadings.loc["msi:Dopamine", factor])
        if "msi:Dopamine" in msi_dopamine_loadings.index else np.nan
    )
    for factor in MOFA_FACTOR_NAMES
}


def cps_association_table(subclasses, factors, *, family, group=None):
    rows = []
    for sc_name in subclasses:
        agg = donor_level_cps(sc_name, predictors=factors, outcomes=CPS_OUTCOMES)
        for factor in factors:
            for outcome in CPS_OUTCOMES:
                fit = fit_cps_ols(agg, factor, outcome)
                if fit is None:
                    continue
                rho, rho_p = spearman_cps(agg, factor, outcome)
                rows.append({
                    "family": family,
                    "group": group,
                    "subclass": sc_name,
                    "factor": factor,
                    "outcome": outcome,
                    "dopamine_loading": factor_dopamine_loading.get(factor, np.nan),
                    **fit,
                    "spearman_rho": rho,
                    "spearman_p": rho_p,
                })
    return pd.DataFrame(rows)


primary_cps = add_bh_qvalues(cps_association_table(
    CELLTYPE_GROUPS["Factor-1 positive"],
    ["Factor1"],
    family="factor1_positive",
    group="Factor-1 positive",
))
specificity_cps = add_bh_qvalues(cps_association_table(
    CELLTYPE_GROUPS["Factor-1 negative"],
    ["Factor1"],
    family="factor1_negative",
    group="Factor-1 negative",
))
headline_cps = pd.concat([primary_cps, specificity_cps], ignore_index=True)
headline_cols = [
    "family", "group", "subclass", "factor", "outcome", "n",
    "beta", "ci_low", "ci_high", "p_value", "q_value",
    "spearman_rho", "spearman_p",
]
print("\n=== Kana 2026 CPS vs Factor1 in CaH: Factor-1-positive and -negative cell types (OLS HC3 + Spearman) ===")
print(headline_cps[headline_cols].round(4).to_string(index=False))

# Raw donor-level direction for the direct-pathway D1 MSN.
d1_cps = donor_level_cps("STR D1 MSN", predictors=["Factor1"], outcomes=CPS_OUTCOMES)
print("\n=== STR D1 MSN: donor-mean Factor1 by Kana CPS quartile ===")
for outcome in CPS_OUTCOMES:
    d1_tmp = d1_cps[["Factor1", outcome]].dropna().copy()
    d1_tmp["quartile"] = pd.qcut(d1_tmp[outcome], q=4, duplicates="drop")
    quartile_summary = d1_tmp.groupby("quartile", observed=True)["Factor1"].agg(
        ["mean", "sem", "size"]
    )
    rho, p = spearman_cps(d1_cps, "Factor1", outcome)
    print(f"\n-- {outcome} (Spearman rho={rho:.3f}, p={p:.3g}) --")
    print(quartile_summary.round(4).to_string())

# Exploratory all-factor/all-subclass screen. This is interpreted separately from the primary
# Factor1/MSN family and BH-corrected across the full exploratory test set.
exploratory_cps = add_bh_qvalues(cps_association_table(
    sorted(cell_df["subclass"].dropna().unique()),
    MOFA_FACTOR_NAMES,
    family="exploratory_all_factors",
))
exploratory_cps = exploratory_cps.sort_values(
    ["q_value", "p_value", "subclass", "factor", "outcome"]
).reset_index(drop=True)
print("\n=== Exploratory screen — all MOFA factors vs Kana CPS across all Subclasses (adj.) ===")
exploratory_cols = [
    "subclass", "factor", "outcome", "n", "beta", "ci_low", "ci_high",
    "p_value", "q_value", "spearman_rho", "dopamine_loading", "adj_r2",
]
print(exploratory_cps[exploratory_cols].head(50).round(4).to_string(index=False))

# %% FIGURE — continuous CPS score association for Factor1 in STR D1 MSN
CPS_LABELS = {
    "CPS": "global CPS",
    "CPS_AT8": "AT8/pTau CPS",
    "CPS_6E10": "6E10/Abeta CPS",
}
primary_lookup = primary_cps.set_index(["subclass", "factor", "outcome"])
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, outcome in zip(axes, CPS_OUTCOMES):
    plot_df = d1_cps[["Factor1", outcome]].dropna().copy()
    plot_df["Factor1_z"] = _zscore(plot_df["Factor1"]).to_numpy()
    ax.scatter(
        plot_df[outcome],
        plot_df["Factor1_z"],
        s=44,
        color="#A24316",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    if len(plot_df) >= 2:
        slope, intercept = np.polyfit(plot_df[outcome], plot_df["Factor1_z"], deg=1)
        xs = np.linspace(plot_df[outcome].min(), plot_df[outcome].max(), 100)
        ax.plot(xs, intercept + slope * xs, color="#E08A54", lw=2.0, zorder=2)
    rho, rho_p = spearman_cps(d1_cps, "Factor1", outcome)
    fit_row = primary_lookup.loc[("STR D1 MSN", "Factor1", outcome)]
    ax.text(
        0.04,
        0.06,
        f"adj. beta={fit_row['beta']:.3f}, q={fit_row['q_value']:.3g}\n"
        f"Spearman rho={rho:.2f}, p={rho_p:.3g}",
        transform=ax.transAxes,
        fontsize=8.5,
        color="0.25",
    )
    ax.set_title(CPS_LABELS[outcome], fontsize=10, weight="bold")
    ax.set_xlabel("Kana 2026 CPS")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("STR D1 MSN donor-mean Factor1 (z-scored)")
fig.tight_layout()
plt.show()

# %% FIGURE — the key result in one panel-pair (Factor1, the dopamine-enriched MOFA factor)
# (a) Forest plot: ordinal effect of Factor1 on AD stage (OR per SD, adjusted) per cell type,
#     drawn as a NESTED interval — thick bar = +/-1 SE (~68%), thin line = 95% CI. The
#     Factor-1-positive MSNs fall left of OR=1 (Factor1 DECREASES with AD); the
#     Factor-1-negative cell types straddle 1,
#     so the effect is cell-type specific.
# (b) Dose-response in the direct-pathway D1 MSN: donor-mean Factor1 (z-scored, +/-1 SEM band)
#     drops at the most advanced AD stage.
from matplotlib.lines import Line2D

PREDICTOR = "Factor1"
order = CELLTYPE_GROUPS["Factor-1 positive"] + CELLTYPE_GROUPS["Factor-1 negative"]
n_target = len(CELLTYPE_GROUPS["Factor-1 positive"])
C_MARK, C_BAND = "#A24316", "#E08A54"   # dark marker/95% line, lighter 68% bar/SEM band

# OR with nested 68% (+/-1 SE) and 95% (CI) intervals on the primary outcome, per cell type.
eff = {}
for sc_name in order:
    coef, _, _, lo95, hi95, se = fit_ordinal(donor_level(sc_name, PRIMARY), PREDICTOR, PRIMARY)
    eff[sc_name] = dict(OR=np.exp(coef), lo68=np.exp(coef - se), hi68=np.exp(coef + se),
                        lo95=np.exp(lo95), hi95=np.exp(hi95))

fig, (axf, axd) = plt.subplots(1, 2, figsize=(12, 4.8),
                               gridspec_kw={"width_ratios": [1.3, 1]})

# --- (a) forest plot with nested dual interval ---
yy = np.arange(len(order))[::-1].astype(float)       # top row = first cell type
ypos = dict(zip(order, yy))
axf.axhspan(yy[n_target - 1] - 0.5, yy[0] + 0.5, color="#C0392B", alpha=0.07, zorder=0)
axf.axhspan(yy[-1] - 1.5, yy[n_target - 1] - 0.5, color="#7F8C8D", alpha=0.07, zorder=0)
for sc_name in order:
    e, y0 = eff[sc_name], ypos[sc_name]
    axf.plot([e["lo95"], e["hi95"]], [y0, y0], color=C_MARK, lw=1.3, zorder=3)          # 95% CI
    axf.plot([e["lo68"], e["hi68"]], [y0, y0], color=C_BAND, lw=6,
             solid_capstyle="round", zorder=4)                                          # +/-1 SE
    axf.scatter([e["OR"]], [y0], color=C_MARK, s=42, zorder=5,
                edgecolor="white", linewidth=0.7)
axf.axvline(1, color="0.35", ls="--", lw=1)
axf.set_xscale("log")
axf.set_xticks([0.3, 0.5, 1, 2])
axf.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
axf.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
axf.set_yticks(yy)
axf.set_yticklabels(order)
axf.set_ylim(yy[-1] - 0.6, yy[0] + 0.6)
axf.set_xlabel("Factor1 odds ratio per SD on AD neuropathological change (adj. age/sex/PMI)\n\n"
               f"{' '*20}$\\leftarrow$ lower with AD          higher with AD $\\rightarrow$")
          
#axf.set_title("(a)  Effect on AD stage is specific to MSNs", loc="left", weight="bold")
axf.text(0.015, 0.985, "Factor-1 positive", transform=axf.transAxes, va="top",
         color="#8B2E2E", weight="bold", fontsize=9)
axf.text(0.015, 0.47, "Factor-1 negative", transform=axf.transAxes, va="top",
         color="#4A5658", weight="bold", fontsize=9)
axf.legend(handles=[Line2D([0], [0], color=C_BAND, lw=6, label="$\\pm$1 SE (68%)"),
                    Line2D([0], [0], color=C_MARK, lw=1.3, label="95% CI")],
           loc="lower right", frameon=True, fontsize=8)

# --- (b) D1 MSN dose-response, +/-1 SEM shaded band ---
d1 = donor_level("STR D1 MSN", PRIMARY)
stages = [c for c in OUTCOMES[PRIMARY] if c in set(d1[PRIMARY].values)]
codes = pd.Categorical(d1[PRIMARY], categories=stages, ordered=True).codes
xs = np.arange(len(stages))
z = (d1[PREDICTOR].values - d1[PREDICTOR].mean()) / d1[PREDICTOR].std()
grp = pd.DataFrame({"c": codes, "z": z}).groupby("c")["z"]
mean, sem = grp.mean().reindex(xs).values, grp.sem().reindex(xs).values
rho, p = spearmanr(codes, d1[PREDICTOR].values)
axd.fill_between(xs, mean - sem, mean + sem, color=C_BAND, alpha=0.10, lw=0)
axd.plot(xs, mean, marker="o", ms=7, lw=2.2, color=C_MARK)
#axd.axhline(0, color="0.7", lw=0.8, ls=":")
axd.axhline(mean[0], color="0.7", lw=1.5, ls=":")
axd.set_xticks(xs)
axd.set_xticklabels(stages, rotation=0)
axd.set_xlim(-0.2, len(stages) - 0.8)
axd.set_xlabel("\nAD neuropathological change")
axd.set_ylabel("donor-mean Factor1 (z-scored, $\\pm$1 SEM)")
#axd.set_title("(b)  D1 MSN: Factor1 drops at advanced AD", loc="left", weight="bold")
#axd.text(0.03, 0.05, f"Spearman $\\rho$={rho:.2f}, p={p:.2f}",
#         transform=axd.transAxes, fontsize=9, color="0.25")

#fig.suptitle("Factor1 (dopamine-enriched multi-omic factor) declines with AD neuropathology, "
#             "specifically in striatal MSNs (caudate head)", weight="bold", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# %% Reusable Factor 1 multi-panel figure
#: Typography for the manuscript figures. Applied via plt.rc_context inside the
#: FIG_* helpers so it never leaks into an interactive session's other plots.
FIG_RC = {
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "figure.titlesize": 16,
}


def _finish_svg(fig, output_file):
    """Show a figure or save it as SVG, regardless of the supplied suffix."""
    if output_file is None:
        plt.show()
    else:
        output_path = Path(output_file).with_suffix(".svg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # dpi sets the resolution of the rasterized scatter layers only; everything
        # else in the SVG stays vector, so this is the print quality of the panels.
        fig.savefig(output_path, format="svg", bbox_inches="tight", dpi=400)
        plt.close(fig)
        print(f"Wrote {output_path}")
    return fig


def _factor_loading_plot_data(mofa_model, factor="Factor1", n_each=5):
    """Top positive/negative RNA and MSI loadings with correct view labels."""
    frames = []
    for view in ("msi", "rna"):
        weights = mofa_model.get_weights(df=True, views=[view])[factor].dropna()
        selected = pd.concat([
            weights.nlargest(n_each),
            weights.nsmallest(n_each),
        ]).loc[lambda x: ~x.index.duplicated()]
        frames.append(pd.DataFrame({
            "feature": selected.index.str.replace(r"^(msi|rna):", "", regex=True),
            "weight": selected.to_numpy(),
            "view": view.upper(),
        }))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )


def plot_factor1_multiomic_weights(
    ax,
    mofa_model,
    *,
    factor="Factor1",
    n_each=5,
    colors=None,
):
    """Lollipop panel of the factor's strongest RNA and MSI loadings."""
    colors = {"MSI": "#1f77b4", "RNA": "#ff7f0e"} if colors is None else colors
    plot_df = _factor_loading_plot_data(mofa_model, factor=factor, n_each=n_each)
    ypos = np.arange(len(plot_df))
    ax.hlines(ypos, 0, plot_df["weight"], color="lightgray", linewidth=2, zorder=1)
    for view, group in plot_df.groupby("view", sort=False):
        idx = group.index.to_numpy()
        ax.scatter(
            group["weight"], idx, color=colors[view], s=110, label=view,
            edgecolors="white", linewidths=0.9, zorder=2,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(plot_df["feature"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel(f"{factor.replace('Factor', 'Factor ')} Weight")
    ax.set_ylabel("")
    ax.legend(title="View", frameon=True, facecolor="white", loc="upper left")
    return plot_df


def subclass_color_map(factor_adata, subclass_key="Subclass"):
    """Category -> colour mapping shared by the UMAP and violin subclass panels.

    Reuses ``uns["<key>_colors"]`` so the two panels cannot drift apart; if scanpy
    has not assigned colours yet (the violin may be drawn first), assign them here
    with the same default scanpy would have used.
    """
    if not isinstance(factor_adata.obs[subclass_key].dtype, pd.CategoricalDtype):
        factor_adata.obs[subclass_key] = factor_adata.obs[subclass_key].astype("category")
    categories = list(factor_adata.obs[subclass_key].cat.categories)
    colors = factor_adata.uns.get(f"{subclass_key}_colors")
    if colors is None or len(colors) != len(categories):
        from scanpy.plotting._utils import _set_default_colors_for_categorical_obs

        _set_default_colors_for_categorical_obs(factor_adata, subclass_key)
        colors = factor_adata.uns[f"{subclass_key}_colors"]
    return dict(zip(categories, colors))


def plot_factor1_umaps(
    factor_adata,
    ax_factor,
    ax_subclass,
    *,
    factor="Factor1",
    subclass_key="Subclass",
):
    """UMAP panels for the factor score and cell subclass."""
    subclass_color_map(factor_adata, subclass_key)
    sc.pl.umap(
        factor_adata, color=factor, ax=ax_factor, show=False, title=factor,
        colorbar_loc="right",
    )
    sc.pl.umap(
        factor_adata, color=subclass_key, ax=ax_subclass, show=False,
        title=subclass_key, legend_loc="on data",
    )
    # scanpy draws one path per nucleus; left as vector these two panels alone put
    # ~2 x 10^5 paths into the SVG. Axes, text and legends stay vector.
    for ax in (ax_factor, ax_subclass):
        for collection in ax.collections:
            collection.set_rasterized(True)
    for label in list(ax_subclass.texts):
        text = label.get_text()
        if "MSN" not in text and "Oligodendrocyte" not in text:
            label.remove()
    return ax_factor, ax_subclass


def plot_factor1_subclass_violin(
    ax,
    factor_adata,
    *,
    factor="Factor1",
    subclass_key="Subclass",
    label_rotation=75,
):
    """Factor-score violin panel ordered by descending subclass mean."""
    frame = pd.DataFrame({
        factor: np.asarray(factor_adata[:, factor].X).ravel(),
        subclass_key: factor_adata.obs[subclass_key].astype(str).to_numpy(),
    })
    order = (
        frame.groupby(subclass_key, observed=True)[factor]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    colors = subclass_color_map(factor_adata, subclass_key)
    sns.violinplot(
        data=frame, x=subclass_key, y=factor, order=order, ax=ax,
        palette=[colors[name] for name in order], cut=0, inner=None,
        linewidth=0.8, saturation=1.0,
    )
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1.2, zorder=0)
    ax.tick_params(axis="x", labelrotation=label_rotation)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    ax.set_xlabel("")
    ax.set_ylabel(factor)
    return order


def _factor1_ordinal_effects(
    *,
    predictor="Factor1",
    outcome=PRIMARY,
    celltype_groups=CELLTYPE_GROUPS,
):
    """Adjusted ordinal-effect intervals used in the Factor 1 forest panel."""
    order = celltype_groups["Factor-1 positive"] + celltype_groups["Factor-1 negative"]
    effects = {}
    for subclass in order:
        fit = fit_ordinal(donor_level(subclass, outcome), predictor, outcome)
        if fit is None:
            raise RuntimeError(f"Ordinal model failed for {subclass!r}.")
        coef, _, _, lo95, hi95, se = fit
        effects[subclass] = {
            "OR": np.exp(coef),
            "lo68": np.exp(coef - se),
            "hi68": np.exp(coef + se),
            "lo95": np.exp(lo95),
            "hi95": np.exp(hi95),
        }
    return order, effects


def plot_factor1_ad_forest(
    ax,
    *,
    predictor="Factor1",
    outcome=PRIMARY,
    celltype_groups=CELLTYPE_GROUPS,
    marker_color="#A24316",
    interval_color="#E08A54",
):
    """Adjusted ordinal odds ratios with nested 68% and 95% intervals."""
    order, effects = _factor1_ordinal_effects(
        predictor=predictor, outcome=outcome, celltype_groups=celltype_groups
    )
    n_target = len(celltype_groups["Factor-1 positive"])
    yy = np.arange(len(order))[::-1].astype(float)
    ypos = dict(zip(order, yy))
    ax.axhspan(yy[n_target - 1] - 0.5, yy[0] + 0.5,
               color="#C0392B", alpha=0.07, zorder=0)
    ax.axhspan(yy[-1] - 1.5, yy[n_target - 1] - 0.5,
               color="#7F8C8D", alpha=0.07, zorder=0)
    for subclass in order:
        effect, y0 = effects[subclass], ypos[subclass]
        ax.plot([effect["lo95"], effect["hi95"]], [y0, y0],
                color=marker_color, lw=1.3, zorder=3)
        ax.plot([effect["lo68"], effect["hi68"]], [y0, y0],
                color=interval_color, lw=6, solid_capstyle="round", zorder=4)
        ax.scatter(effect["OR"], y0, color=marker_color, s=42, zorder=5,
                   edgecolor="white", linewidth=0.7)
    ax.axvline(1, color="0.35", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2])
    ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.matplotlib.ticker.NullFormatter())
    ax.set_yticks(yy)
    ax.set_yticklabels(order)
    ax.set_ylim(yy[-1] - 0.6, yy[0] + 0.6)
    ax.set_xlabel(
        f"{predictor} odds ratio per SD on AD neuropathological change "
        "(adj. age/sex/PMI)\n\n"
        r"$\leftarrow$ lower with AD          higher with AD $\rightarrow$"
    )
    ax.text(0.015, 0.985, "Factor-1 positive", transform=ax.transAxes, va="top",
            color="#8B2E2E", weight="bold", fontsize=9)
    ax.text(0.015, 0.47, "Factor-1 negative", transform=ax.transAxes, va="top",
            color="#4A5658", weight="bold", fontsize=9)
    ax.legend(
        handles=[
            Line2D([0], [0], color=interval_color, lw=6, label=r"$\pm$1 SE (68%)"),
            Line2D([0], [0], color=marker_color, lw=1.3, label="95% CI"),
        ],
        loc="lower right", frameon=True, fontsize=8,
    )
    return effects


def plot_factor1_ad_dose_response(
    ax,
    *,
    subclass="STR D1 MSN",
    predictor="Factor1",
    outcome=PRIMARY,
    marker_color="#A24316",
    band_color="#E08A54",
):
    """Donor-mean standardized Factor 1 by ordinal AD stage."""
    donor_df = donor_level(subclass, outcome)
    stages = [level for level in OUTCOMES[outcome] if level in set(donor_df[outcome])]
    codes = pd.Categorical(
        donor_df[outcome], categories=stages, ordered=True
    ).codes
    xs = np.arange(len(stages))
    values = donor_df[predictor].to_numpy()
    z = (values - values.mean()) / values.std()
    grouped = pd.DataFrame({"stage": codes, "z": z}).groupby("stage")["z"]
    mean = grouped.mean().reindex(xs).to_numpy()
    sem = grouped.sem().reindex(xs).to_numpy()
    ax.fill_between(xs, mean - sem, mean + sem, color=band_color, alpha=0.10, lw=0)
    ax.plot(xs, mean, marker="o", ms=7, lw=2.2, color=marker_color)
    ax.axhline(mean[0], color="0.7", lw=1.5, ls=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(stages)
    ax.set_xlim(-0.2, len(stages) - 0.8)
    ax.set_xlabel("AD neuropathological change")
    ax.set_ylabel(f"donor-mean {predictor} (z-scored, $\\pm$1 SEM)")
    return pd.DataFrame({"stage": stages, "mean": mean, "sem": sem})


def FIG_factor1_cah_multipanel(
    mofa_model=mofa,
    factor_adata=mofa_adata,
    *,
    factor="Factor1",
    output_file=None,
):
    """Assemble loadings, UMAPs, subclass violin, and AD association panels."""
    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0])
    top = outer[0].subgridspec(1, 2, width_ratios=[0.34, 0.66], wspace=0.12)
    upper_right = top[1].subgridspec(
        2, 2, height_ratios=[1.0, 0.9], hspace=0.28, wspace=0.18
    )
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.25, 0.95], wspace=0.18)

    ax_weights = fig.add_subplot(top[0])
    ax_factor = fig.add_subplot(upper_right[0, 0])
    ax_subclass = fig.add_subplot(upper_right[0, 1])
    ax_violin = fig.add_subplot(upper_right[1, :])
    ax_forest = fig.add_subplot(bottom[0])
    ax_dose = fig.add_subplot(bottom[1])

    plot_factor1_multiomic_weights(ax_weights, mofa_model, factor=factor)
    plot_factor1_umaps(
        factor_adata, ax_factor, ax_subclass, factor=factor
    )
    plot_factor1_subclass_violin(ax_violin, factor_adata, factor=factor)
    plot_factor1_ad_forest(ax_forest, predictor=factor)
    plot_factor1_ad_dose_response(ax_dose, predictor=factor)
    return _finish_svg(fig, output_file)


FIG_factor1_cah_multipanel(
    output_file=CAH_OUTPUT_ROOT / "figures" / "factor1_cah_multipanel.svg"
)


# %% Transfer-characterisation figure (supersedes the AD-association panels)
# The ADNC association did not survive adjustment for donor cell-type composition, and
# the imputed-MSI view proved rank-limited by the 10-dimensional student embedding and
# the 4-component PLS decoder. These panels characterise what the transfer does
# deliver: a low-dimensional metabolic axis that separates MSN from oligodendrocyte
# identity in snRNA-seq where no metabolomics was measured.

CAH_MOFA_PATH = Path(
    getattr(mofa, "filepath", None) or (CAH_OUTPUT_ROOT / "snrna_mofa.h5mu")
)


def load_imputed_msi_sample(mofa_path=None, n_cells=20000, seed=0):
    """Sample the imputed-MSI matrix MOFA was actually trained on, from the model file.

    The aligned MuData on disk may have been rebuilt with a different SpatialJEPA
    bundle than the one behind these factors, so the model file is the only
    self-consistent source for these panels.
    """
    import h5py

    mofa_path = Path(mofa_path or CAH_MOFA_PATH)
    rng = np.random.default_rng(seed)
    with h5py.File(mofa_path, "r") as handle:
        dset = handle["data/msi/group1"]
        idx = np.sort(
            rng.choice(dset.shape[0], size=min(n_cells, dset.shape[0]), replace=False)
        )
        matrix = dset[idx, :].astype(np.float64)
        features = [x.decode() for x in handle["features/msi"][:]]
    return idx, matrix, features


def msi_rank_summary(matrix):
    """Variance spectrum and participation-ratio effective rank of the imputed MSI."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    eigenvalues = np.linalg.svd(centred, full_matrices=False, compute_uv=False) ** 2
    fractions = eigenvalues / eigenvalues.sum()
    return {
        "fractions": fractions,
        "cumulative": np.cumsum(fractions),
        "effective_rank": float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()),
        "n_for_95": int(np.searchsorted(np.cumsum(fractions), 0.95) + 1),
        "n_features": matrix.shape[1],
    }


def plot_msi_effective_rank(ax, matrix, n_show=12, bar_color="#4A6FA5"):
    """Scree bars with the cumulative-variance trace and the effective rank annotated."""
    summary = msi_rank_summary(matrix)
    xs = np.arange(1, n_show + 1)
    ax.bar(xs, summary["fractions"][:n_show], color=bar_color, width=0.7)
    ax.set_xlabel("imputed-MSI principal component")
    ax.set_ylabel("fraction of variance")
    ax.set_xticks(xs[::2])
    twin = ax.twinx()
    twin.plot(xs, summary["cumulative"][:n_show], color="#A24316", marker="o",
              ms=3.5, lw=1.4)
    twin.axhline(0.95, color="0.55", ls=":", lw=1)
    twin.set_ylim(0, 1.05)
    twin.set_ylabel("cumulative", color="#A24316")
    twin.tick_params(axis="y", colors="#A24316")
    ax.text(
        0.97, 0.55,
        f"effective rank {summary['effective_rank']:.2f}\nof {summary['n_features']} "
        f"features\n{summary['n_for_95']} PCs for 95%",
        transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
    )
    return summary


def plot_hexbin_agreement(ax, x, y, xlabel, ylabel, gridsize=55, cmap="viridis"):
    """Per-nucleus agreement between two continuous readouts, annotated with Pearson r."""
    finite = np.isfinite(x) & np.isfinite(y)
    ax.hexbin(x[finite], y[finite], gridsize=gridsize, bins="log", cmap=cmap,
              linewidths=0, rasterized=True)
    r = float(np.corrcoef(x[finite], y[finite])[0, 1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.04, 0.95, f"$r$ = {r:+.3f}", transform=ax.transAxes, va="top",
            fontsize=9.5, bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=2.5))
    return r


def plot_factor_rna_loadings(ax, mofa_model, factor="Factor1", n_each=8,
                             pos_color="#A24316", neg_color="#4A6FA5"):
    """Top positive and negative RNA-view loadings; the high-dimensional half of the model."""
    weights = mofa_model.get_weights(df=True, views=["rna"])[factor].sort_values()
    selected = pd.concat([weights.head(n_each), weights.tail(n_each)])
    labels = [name.split(":", 1)[-1] for name in selected.index]
    colors = [neg_color if v < 0 else pos_color for v in selected.values]
    ax.barh(np.arange(len(selected)), selected.values, color=colors, height=0.72)
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(labels, fontsize=8, style="italic")
    ax.axvline(0, color="0.35", lw=1)
    ax.set_xlabel(f"{factor} RNA loading")
    return selected


def donor_composition(cell_df_, msn_subclasses=("STR D1 MSN", "STR D2 MSN",
                                                "STR Hybrid MSN")):
    """Per-donor cell-type fractions, computed from the per-cell table."""
    counts = cell_df_.groupby(["donor", "subclass"]).size().unstack(fill_value=0)
    fractions = counts.div(counts.sum(axis=1), axis=0)
    present = [s for s in msn_subclasses if s in fractions.columns]
    out = pd.DataFrame(index=counts.index)
    out["frac_oligo"] = (fractions["Oligodendrocyte"]
                         if "Oligodendrocyte" in fractions else np.nan)
    out["frac_msn"] = fractions.reindex(columns=present, fill_value=0.0).sum(axis=1)
    out.index.name = "donor"
    return out


def plot_composition_sensitivity(ax, cell_df_, subclass="Oligodendrocyte",
                                 factor="Factor1", point_color="#4A6FA5"):
    """Donor-mean factor against donor MSN fraction: the compositional caveat."""
    composition = donor_composition(cell_df_)
    means = (cell_df_[cell_df_["subclass"] == subclass]
             .groupby("donor")[factor].mean())
    merged = pd.concat([means.rename("factor"), composition["frac_msn"]],
                       axis=1).dropna()
    ax.scatter(merged["frac_msn"], merged["factor"], s=26, color=point_color,
               edgecolor="white", linewidth=0.6)
    slope, intercept = np.polyfit(merged["frac_msn"], merged["factor"], deg=1)
    grid = np.linspace(merged["frac_msn"].min(), merged["frac_msn"].max(), 50)
    ax.plot(grid, slope * grid + intercept, color="#A24316", lw=1.4)
    r, p = pearsonr(merged["frac_msn"], merged["factor"])
    ax.set_xlabel("donor MSN fraction")
    ax.set_ylabel(f"donor-mean {factor}\nin {subclass}")
    ax.text(0.04, 0.95, f"$r$ = {r:+.2f}\n$p$ = {p:.1g}\n$n$ = {len(merged)}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=2.5))
    return merged


def annotate_gm_wm(section, neighborhood_key="Neighborhood", subclass_key="Subclass",
                   mode="neighborhood"):
    """Grey/white-matter labels for a spatial section.

    ``mode="neighborhood"`` (default) defines both compartments from the spatial
    neighbourhood annotation, so the downstream separation test asks whether the
    factor tracks tissue context. ``mode="mixed"`` reproduces
    3__sea_ad_cah_spatial_snrna_dopamine.py, which takes grey matter from the
    neighbourhood label and white matter from the oligodendrocyte subclass; that
    version is partly circular here, because the factor is already known to
    separate MSN from oligodendrocyte identity.
    """
    if mode == "neighborhood":
        section.obs["GM_WM"] = section.obs[neighborhood_key].map(
            {"LGE_Neighborhood": "GM", "OPC-Oligo_Neighborhood": "WM"}
        )
    elif mode == "mixed":
        section.obs["GM_WM"] = (
            section.obs[neighborhood_key].map({"LGE_Neighborhood": "GM"})
            .combine_first(
                section.obs[subclass_key].map({"Oligodendrocyte_Subclass": "WM"})
            )
        )
    else:
        raise ValueError("mode must be 'neighborhood' or 'mixed'")
    return section


def prepare_spatial_for_projection(spatial, target_sum=1e4, max_value=10):
    """Put a spatial dataset on the same scale as the model's RNA view.

    That view is centred and standardised, so the input must be depth-normalised,
    log-transformed and z-scored per gene before projection. Raw counts are kept in
    ``layers['counts']`` and the pre-normalisation library size in
    ``obs['total_counts_raw']``, both of which are needed for QC and for checking
    that the projected score is not simply tracking sequencing depth.
    """
    from scipy.sparse import issparse

    spatial.layers["counts"] = spatial.X.copy()
    totals = spatial.X.sum(axis=1)
    spatial.obs["total_counts_raw"] = np.asarray(
        totals.A1 if hasattr(totals, "A1") else totals
    ).ravel()
    sc.pp.normalize_total(spatial, target_sum=target_sum)
    sc.pp.log1p(spatial)
    sc.pp.scale(spatial, max_value=max_value)
    return spatial


def project_spatial_onto_factor(spatial, mofa_model=None, factor_index=0,
                                key="mofa_factor1", view="rna", backend="direct"):
    """Project spatial cells onto a factor using the RNA-view loadings only.

    The imputed-MSI view is rank-limited, but the RNA view is not, so this tests
    whether the transcriptional half of the transferred axis is spatially coherent
    in data the model never saw.

    ``backend="direct"`` reproduces mofax's projection -- the pseudo-inverse of the
    full weights matrix, restricted to the features shared with the input -- but
    keeps the expression matrix sparse. mofax's own ``project_data`` densifies the
    matrix and rebuilds it as a labelled DataFrame first, which dominates the
    runtime for a Xenium section; the arithmetic itself is a matrix-vector product.
    Pass ``backend="mofax"`` to use the library implementation instead (requires
    ``mofax_gpu_patch.patch_mofax_gpu()``, since upstream mofax mishandles both the
    pseudo-inverse shape and the view prefix when intersecting features).
    """
    from scipy.sparse import issparse

    # The model's RNA view is centred and standardised (mean ~0, sd ~1, ~80% of
    # entries negative). Projecting raw counts through weights fitted on that scale
    # yields a score dominated by library size rather than expression pattern, so
    # refuse to do it silently.
    probe = spatial.X[:1000] if spatial.n_obs > 1000 else spatial.X
    probe = np.asarray(probe.todense()) if issparse(probe) else np.asarray(probe)
    if probe.size and probe.min() >= 0 and np.allclose(probe, np.round(probe)):
        raise ValueError(
            "Input X looks like raw integer counts, but the model's RNA view is "
            "z-scored; the projection would mostly measure total counts. Run "
            "prepare_spatial_for_projection(section) first, or pass an already "
            "normalised, log-transformed and scaled matrix."
        )

    mofa_model = mofa if mofa_model is None else mofa_model
    if backend == "mofax":
        projected = mofa_model.project_data(
            spatial, factors=[factor_index], view=view, feature_intersection=True
        )
        spatial.obs[key] = np.asarray(projected).ravel()
        return spatial

    prefix = f"{view}:"
    view_features = np.array([
        f[len(prefix):] if f.startswith(prefix) else f
        for f in mofa_model.features[view]
    ])
    # Pseudo-inverse over the full feature set, then restricted to the shared
    # features -- the order mofax uses, and not the same as inverting the subset.
    winv = np.linalg.pinv(mofa_model.get_weights(views=view, factors=[factor_index]))
    shared = np.intersect1d(np.asarray(spatial.var_names), view_features)
    if shared.size == 0:
        raise ValueError(
            f"No features shared between the model's '{view}' view and the input."
        )
    order = np.argsort(view_features)
    indices = order[np.searchsorted(view_features[order], shared)]

    matrix = spatial[:, shared].X
    weights = np.asarray(winv[:, indices], dtype=np.float64)
    scores = (matrix @ weights.T) if issparse(matrix) else np.asarray(matrix) @ weights.T

    spatial.obs[key] = np.asarray(scores).ravel()
    print(f"[project] {shared.size} of {view_features.size} '{view}' features present "
          f"in the input ({spatial.n_obs:,} cells)")
    return spatial


def plot_spatial_factor(ax, section, factor_key="mofa_factor1", spatial_key="spatial",
                        size=2.0, cmap="RdBu_r"):
    """One tissue section coloured by the projected factor."""
    coords = np.asarray(section.obsm[spatial_key])
    values = np.asarray(section.obs[factor_key], dtype=float)
    limit = np.nanpercentile(np.abs(values), 99)
    handle = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, cmap=cmap,
                        vmin=-limit, vmax=limit, linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = ax.figure.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
    bar.set_label(f"projected {factor_key}", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    return handle


def plot_gm_wm_separation(ax, section, factor_key="mofa_factor1", group_key="GM_WM",
                          colors=("#A24316", "#4A6FA5")):
    """Projected factor in grey- versus white-matter cells, with AUROC."""
    from sklearn.metrics import roc_auc_score

    table = section.obs[[factor_key, group_key]].dropna()
    groups = ["GM", "WM"]
    values = [table.loc[table[group_key] == g, factor_key].to_numpy(dtype=float)
              for g in groups]
    parts = ax.violinplot(values, showextrema=False, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    for position, series in enumerate(values, start=1):
        ax.plot([position - 0.18, position + 0.18], [np.median(series)] * 2,
                color="0.15", lw=1.6)
    auroc = roc_auc_score(
        (table[group_key] == "GM").astype(int), table[factor_key].astype(float)
    )
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"{g}\n($n$={len(v):,})" for g, v in zip(groups, values)])
    ax.set_ylabel(f"projected {factor_key}")
    ax.text(0.04, 0.95, f"AUROC = {auroc:.3f}", transform=ax.transAxes, va="top",
            fontsize=9, bbox=dict(fc="white", ec="0.7", alpha=0.85, pad=2.5))
    return auroc


def FIG_cah_transfer_main(
    mofa_model=mofa,
    factor_adata=mofa_adata,
    xenium_section=None,
    *,
    factor="Factor1",
    smooth=30,
    output_file=None,
):
    """Main manuscript figure: the transferred axis, then the same signal in tissue.

    Top row  (a) multi-omic Factor 1 loadings, metabolites alongside genes
             (b) RNA UMAPs coloured by Factor 1 and by subclass
             (c) Factor 1 across subclasses
    Bottom   (d) local oligodendrocyte density  (e) imputed dopamine
             (f) projected Factor 1

    The bottom row is illustrative: it places the transferred signal in tissue and
    makes no spatial-association claim (see FIG_cah_spatial_row and the negative
    controls recorded in CAH_SECTION_REWRITE.md).
    """
    has_spatial = xenium_section is not None
    height = 11.5 if has_spatial else 7.0
    with plt.rc_context(FIG_RC):
        fig = plt.figure(figsize=(14, height), constrained_layout=True)
        # The near-vertical subclass tick labels claim a lot of the top row, so it
        # gets a larger share; the lollipop panel spans that row and grows with it.
        outer = fig.add_gridspec(2 if has_spatial else 1, 1,
                                 height_ratios=[1.8, 1.0] if has_spatial else [1.0])

        # The wspace/hspace here are what keep the UMAPs from collapsing: under
        # constrained_layout they are charged against the axes, not the figure, so
        # small increases cost a lot of panel area (wspace 0.18 -> 0.02 roughly
        # doubles each UMAP's width and height).
        top = outer[0].subgridspec(1, 2, width_ratios=[0.26, 0.74], wspace=0.08)
        upper_right = top[1].subgridspec(2, 2, height_ratios=[1.3, 1.0],
                                         hspace=0.05, wspace=0.02)
        plot_factor1_multiomic_weights(fig.add_subplot(top[0]), mofa_model,
                                       factor=factor)
        plot_factor1_umaps(factor_adata, fig.add_subplot(upper_right[0, 0]),
                           fig.add_subplot(upper_right[0, 1]), factor=factor)
        plot_factor1_subclass_violin(fig.add_subplot(upper_right[1, :]), factor_adata,
                                     factor=factor)

        if has_spatial:
            bottom = outer[1].subgridspec(1, 3, wspace=0.08)
            if "X_spatial" not in xenium_section.obsm:
                xenium_section.obsm["X_spatial"] = xenium_section.obsm["spatial"]
            if "dens_oligo" not in xenium_section.obs:
                xenium_section.obs["dens_oligo"] = local_category_density(
                    xenium_section, "Oligodendrocyte_Subclass"
                )
            plot_spatial_continuous(fig.add_subplot(bottom[0]), xenium_section,
                                    "dens_oligo", "local oligodendrocyte density",
                                    cmap="magma", diverging=False, smooth=0, size=10.0)
            plot_spatial_continuous(fig.add_subplot(bottom[1]), xenium_section,
                                    "imputed_dopamine",
                                    f"imputed dopamine ({smooth}-NN mean)",
                                    smooth=smooth, size=8.0)
            plot_spatial_continuous(fig.add_subplot(bottom[2]), xenium_section,
                                    "mofa_factor1",
                                    f"projected Factor 1 ({smooth}-NN mean)",
                                    smooth=smooth, size=8.0)
        return _finish_svg(fig, output_file)


def FIG_cah_transfer_supplementary(
    mofa_model=mofa,
    cell_table=None,
    *,
    factor="Factor1",
    dopamine_feature="msi:Dopamine",
    n_cells=20000,
    n_pcs_shown=8,
    output_file=None,
):
    """Supplementary: what the transferred metabolic channel can and cannot resolve.

    (a) rank spectrum of the imputed-MSI view  (b) factor vs imputed dopamine
    (c) donor-composition sensitivity  (d) factor vs total imputed-MSI intensity
    """
    cell_table = cell_df if cell_table is None else cell_table
    idx, matrix, features = load_imputed_msi_sample(n_cells=n_cells)
    factor_values = mofa_model.get_factors(df=True)[factor].values[idx]
    dopamine = matrix[:, features.index(dopamine_feature)]
    intensity = matrix.mean(axis=1)

    with plt.rc_context(FIG_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.0), constrained_layout=True)
        summary = plot_msi_effective_rank(axes[0, 0], matrix, n_show=n_pcs_shown)
        # Shared abscissa scaling so the eye compares the relationships, not units.
        r_dopamine = plot_hexbin_agreement(
            axes[0, 1], (dopamine - dopamine.mean()) / dopamine.std(), factor_values,
            f"imputed {dopamine_feature.split(':', 1)[-1].lower()} (z)", factor,
        )
        plot_composition_sensitivity(axes[1, 0], cell_table, factor=factor)
        r_intensity = plot_hexbin_agreement(
            axes[1, 1], (intensity - intensity.mean()) / intensity.std(), factor_values,
            "mean imputed-MSI intensity (z)", factor,
        )
        axes[1, 1].set_xlim(axes[0, 1].get_xlim())

        print(f"[SUPP] effective rank {summary['effective_rank']:.2f}; "
              f"r({factor}, dopamine) = {r_dopamine:+.3f}; "
              f"r({factor}, intensity) = {r_intensity:+.3f}")
        if output_file is not None:
            fig.savefig(str(output_file).replace('.svg', '.png'), dpi=200)
            print(f"Wrote {output_file}")
        fig.show()
        plt.close(fig)
        return None   


def FIG_cah_transfer_multipanel(
    mofa_model=mofa,
    factor_adata=mofa_adata,
    cell_table=None,
    xenium_section=None,
    *,
    factor="Factor1",
    dopamine_feature="msi:Dopamine",
    n_cells=20000,
    n_pcs_shown=8,
    output_file=None,
):
    """Characterise what the SpatialMETA transfer delivers in SEA-AD CaH snRNA-seq.

    (a) rank spectrum of the imputed-MSI view   (b) factor vs imputed dopamine
    (c) factor vs total imputed-MSI intensity   (d) factor RNA loadings
    (e) factor across subclasses                (f) donor-composition sensitivity

    Passing ``xenium_section`` (a single Xenium section carrying ``obsm['spatial']``,
    ``obs['GM_WM']`` and the projected factor) adds two spatial-validation panels:
    (g) the section coloured by the projected factor, (h) grey- versus white-matter
    separation with AUROC.
    """
    cell_table = cell_df if cell_table is None else cell_table
    idx, matrix, features = load_imputed_msi_sample(n_cells=n_cells)
    factor_values = mofa_model.get_factors(df=True)[factor].values[idx]
    dopamine = matrix[:, features.index(dopamine_feature)]
    intensity = matrix.mean(axis=1)

    n_cols = 4 if xenium_section is not None else 3
    fig = plt.figure(figsize=(4.5 * n_cols, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, n_cols, height_ratios=[1.0, 1.0])
    ax_rank = fig.add_subplot(grid[0, 0])
    ax_dopa = fig.add_subplot(grid[0, 1])
    ax_intensity = fig.add_subplot(grid[0, 2])
    ax_loadings = fig.add_subplot(grid[1, 0])
    ax_violin = fig.add_subplot(grid[1, 1])
    ax_composition = fig.add_subplot(grid[1, 2])

    summary = plot_msi_effective_rank(ax_rank, matrix, n_show=n_pcs_shown)
    # Both agreement panels share a colour map and z-scored abscissa so the eye
    # compares the two relationships rather than their units.
    r_dopamine = plot_hexbin_agreement(
        ax_dopa, (dopamine - dopamine.mean()) / dopamine.std(), factor_values,
        f"imputed {dopamine_feature.split(':', 1)[-1].lower()} (z)", factor,
    )
    r_intensity = plot_hexbin_agreement(
        ax_intensity, (intensity - intensity.mean()) / intensity.std(), factor_values,
        "mean imputed-MSI intensity (z)", factor,
    )
    ax_intensity.set_xlim(ax_dopa.get_xlim())
    plot_factor_rna_loadings(ax_loadings, mofa_model, factor=factor)
    plot_factor1_subclass_violin(ax_violin, factor_adata, factor=factor)
    plot_composition_sensitivity(ax_composition, cell_table, factor=factor)

    auroc = None
    if xenium_section is not None:
        plot_spatial_factor(fig.add_subplot(grid[0, 3]), xenium_section)
        auroc = plot_gm_wm_separation(fig.add_subplot(grid[1, 3]), xenium_section)

    print(f"[FIG] effective rank {summary['effective_rank']:.2f}; "
          f"r({factor}, dopamine) = {r_dopamine:+.3f}; "
          f"r({factor}, intensity) = {r_intensity:+.3f}"
          + (f"; GM/WM AUROC = {auroc:.3f}" if auroc is not None else ""))
    return _finish_svg(fig, output_file)


CAH_XENIUM_PATH = Path(
    f"{__import__('os').environ['SPATIALJEPA_HOME']}/data/SEA_AD/sea_ad_cah/"
    "CaH_Xenium_final.2026-01-07.h5ad"
)
CAH_XENIUM_SECTION_KEY = "Specimen_ID"
# The most GM/WM-balanced of the 25 sections (24.4% LGE neighbourhood, 26.2%
# oligodendrocyte subclass), and the donor Kana et al. use for their spatial
# compartment figures, so this panel sits alongside published maps of the same
# tissue. The script's own criterion (highest LGE fraction) picks
# H20.33.026.CX18.CaH.03.06, which carries only 12.6% oligodendrocytes.
CAH_XENIUM_SECTION_ID = "H21.33.002.CX18.CaH.02.01"


def load_xenium_factor_section(section_id=CAH_XENIUM_SECTION_ID, xenium=None,
                               mode="neighborhood"):
    """One Xenium section, projected onto Factor 1 and labelled grey/white matter.

    Subsets before projecting on purpose: mofax marshals the whole expression
    matrix, and the 25 sections together hold ~963k cells for a panel that needs
    one. Reuses ``cah_xenium`` from the session if it is already loaded.
    """
    if xenium is None:
        xenium = globals().get("cah_xenium")
    if xenium is None:
        xenium = sc.read_h5ad(CAH_XENIUM_PATH)
    mask = xenium.obs[CAH_XENIUM_SECTION_KEY].eq(section_id).to_numpy()
    if not mask.any():
        raise ValueError(
            f"No cells match {CAH_XENIUM_SECTION_KEY}={section_id!r}; "
            f"available sections: {sorted(xenium.obs[CAH_XENIUM_SECTION_KEY].unique())[:3]} ..."
        )
    section = xenium[mask].copy()
    if "reads" in section.layers:
        del section.layers["reads"]
    section = prepare_spatial_for_projection(section)
    section = project_spatial_onto_factor(section)
    print(f"[Xenium] {section_id}: {section.n_obs:,} cells")
    return annotate_gm_wm(section, mode=mode)


SPATIALMETA_XENIUM_KIND = "merfish"   # "merfish" (raw panel) | "tangram" (imputed)


def attach_xenium_imputed_msi(section, section_id=None, kind=SPATIALMETA_XENIUM_KIND,
                              source="student_msi", features=("msi:Dopamine",)):
    """Attach SpatialMETA-imputed MSI features to a Xenium section, by barcode.

    The handoff .npz keeps both decoder outputs; ``student_msi`` is the PLS path
    used for the snRNA MOFA, so it is the consistent choice here even though the
    companion .h5ad was written from ``teacher_msi``.
    """
    section_id = section_id or CAH_XENIUM_SECTION_ID
    donor = ".".join(section_id.split(".")[:3])
    path = (CAH_OUTPUT_ROOT / "spatialmeta_handoff" / "outputs" / kind / donor
            / section_id / "embeddings.npz")
    bundle = np.load(path, allow_pickle=True)
    names = np.asarray(bundle["msi_feature_names"], dtype=str)
    barcodes = pd.Index(np.asarray(bundle["obs_names"], dtype=str))
    matrix = bundle[source]

    positions = barcodes.get_indexer(pd.Index(np.asarray(section.obs_names, dtype=str)))
    if (positions < 0).all():
        raise ValueError(
            f"No barcode overlap between {path.name} and the section "
            f"(npz e.g. {barcodes[0]!r}, section e.g. {section.obs_names[0]!r})."
        )
    for feature in features:
        column = np.full(section.n_obs, np.nan)
        matched = positions >= 0
        column[matched] = matrix[positions[matched], list(names).index(feature)]
        section.obs[f"imputed_{feature.split(':', 1)[-1].lower()}"] = column
    print(f"[MSI] attached {len(features)} feature(s) from {kind}/{source}; "
          f"{int((positions >= 0).sum()):,}/{section.n_obs:,} cells matched")
    return section


def local_category_density(section, category, groupby="Subclass",
                           spatial_key="X_spatial", bw_scale=0.25, max_train=5000,
                           seed=0):
    """KDE of one cell category, evaluated at EVERY cell rather than its own members.

    scanpy's embedding_density scores each category only where that category sits;
    here the field is needed everywhere, so that any cell can be described by the
    tissue around it. Training points are subsampled because gaussian_kde costs
    O(n_train x n_eval).
    """
    from scipy.stats import gaussian_kde

    if spatial_key not in section.obsm:
        section.obsm[spatial_key] = section.obsm["spatial"]
    coords = np.asarray(section.obsm[spatial_key])[:, :2]
    mask = (section.obs[groupby].astype(str) == category).to_numpy()
    if not mask.any():
        raise ValueError(f"No cells with {groupby} == {category!r}.")
    train = coords[mask]
    rng = np.random.default_rng(seed)
    if len(train) > max_train:
        train = train[rng.choice(len(train), max_train, replace=False)]
    kde = gaussian_kde(train.T, bw_method=lambda k: bw_scale * k.scotts_factor())
    values = kde(coords.T)
    return (values - values.min()) / (np.ptp(values) or 1.0)


def smooth_over_neighbours(section, key, spatial_key="X_spatial", n_neighbors=30):
    """kNN spatial mean of a per-cell value, for legible display maps."""
    from sklearn.neighbors import NearestNeighbors

    coords = np.asarray(section.obsm[spatial_key])[:, :2]
    _, neighbours = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords).kneighbors(coords)
    values = section.obs[key].to_numpy(dtype=float)
    return np.nanmean(values[neighbours], axis=1)


def plot_spatial_continuous(ax, section, key, title, spatial_key="X_spatial",
                            cmap="RdBu_r", diverging=True, smooth=30, size=6.0,
                            title_fontsize=14, cbar_labelsize=10):
    """One section coloured by a continuous per-cell value, optionally kNN-smoothed."""
    values = (smooth_over_neighbours(section, key, spatial_key, smooth) if smooth
              else section.obs[key].to_numpy(dtype=float))
    coords = np.asarray(section.obsm[spatial_key])[:, :2]
    if diverging:
        centre = float(np.nanmedian(values))
        span = float(np.nanpercentile(np.abs(values - centre), 98))
        kwargs = dict(cmap=cmap, vmin=centre - span, vmax=centre + span)
    else:
        kwargs = dict(cmap=cmap,
                      vmin=float(np.nanpercentile(values, 2)),
                      vmax=float(np.nanpercentile(values, 98)))
    handle = ax.scatter(coords[:, 0], coords[:, 1], c=values, s=size, linewidths=0,
                        rasterized=True, **kwargs)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=title_fontsize)
    bar = ax.figure.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
    bar.ax.tick_params(labelsize=cbar_labelsize)
    return values


# --------------------------------------------------------------------------
# Student / teacher latent space
#
# The two assays are embedded SEPARATELY on purpose. A joint UMAP of snRNA and
# Xenium student embeddings splits by platform rather than by biology, so a
# combined plot would show a batch effect and nothing else. Read each panel
# within its own assay; do not compare positions across the two.
# --------------------------------------------------------------------------

ADNC_ORDER = ["Not AD", "Low", "Intermediate", "High"]
#: Ordinal, so the palette is a progression rather than arbitrary hues.
ADNC_COLORS = ["#3B7EA1", "#8FBF9F", "#E0B040", "#B2182B"]

_XENIUM_OBS = None


def load_xenium_obs():
    """Xenium `obs` alone, without touching the 2.2 GB expression matrix."""
    global _XENIUM_OBS
    if _XENIUM_OBS is None:
        import h5py
        from anndata.io import read_elem

        with h5py.File(CAH_XENIUM_PATH, "r") as handle:
            _XENIUM_OBS = read_elem(handle["obs"])
    return _XENIUM_OBS


def assign_shared_colors(adatas, key, order=None, palette=None):
    """Give every AnnData the same category -> colour mapping for `key`.

    The snRNA and Xenium panels carry overlapping but not identical Subclass sets;
    colours assigned per-object would not correspond between them. Categories are
    unioned so an absent subclass keeps its slot in the legend.
    """
    from scanpy.plotting import palettes

    seen = []
    for adata in adatas:
        seen.extend(pd.Categorical(adata.obs[key]).categories.tolist())
    order = order or sorted(dict.fromkeys(seen))
    if palette is None:
        base = palettes.vega_20_scanpy if len(order) <= 20 else palettes.default_102
        palette = [base[i % len(base)] for i in range(len(order))]
    mapping = dict(zip(order, palette))
    for adata in adatas:
        adata.obs[key] = pd.Categorical(adata.obs[key], categories=order)
        adata.uns[f"{key}_colors"] = [mapping[c] for c in order]
    return mapping


def _subsample(n_obs, n_cells, seed):
    if not n_cells or n_cells >= n_obs:
        return np.arange(n_obs)
    return np.sort(np.random.default_rng(seed).choice(n_obs, n_cells, replace=False))


def load_student_latent_snrna(embedding="student", n_cells=150_000, seed=0):
    """snRNA student (or teacher) latent space, labelled with Subclass and ADNC."""
    key = f"X_spatialmeta_{embedding}"
    labels = scrna_msi_mudata.mod["msi"].obs
    idx = _subsample(rna.n_obs, n_cells, seed)
    latent = AnnData(
        X=np.asarray(rna.obsm[key], dtype=np.float32)[idx],
        obs=pd.DataFrame(
            {
                "Subclass": labels["Subclass"].astype(str).to_numpy()[idx],
                "ADNC": labels["Overall AD neuropathological Change"]
                        .astype(str).to_numpy()[idx],
                "donor": labels["Donor ID"].astype(str).to_numpy()[idx],
            },
            index=pd.Index(rna.obs_names.astype(str)[idx], name=None),
        ),
    )
    print(f"[latent] snRNA {embedding}: {latent.shape}, "
          f"{latent.obs['donor'].nunique()} donors")
    return latent


def load_student_latent_xenium(kind=SPATIALMETA_XENIUM_KIND, embedding="student",
                               sections=None, n_cells=150_000, seed=0):
    """Xenium student (or teacher) latent space, pooled over the handoff sections.

    Subclass and ADNC come from the Xenium `obs`, matched by barcode; the embedding
    itself comes from the SpatialMETA handoff .npz files.
    """
    root = CAH_OUTPUT_ROOT / "spatialmeta_handoff" / "outputs" / kind
    paths = sorted(root.rglob("embeddings.npz"))
    if sections is not None:
        wanted = set(sections)
        paths = [p for p in paths if p.parent.name in wanted]
    if not paths:
        raise FileNotFoundError(f"No embeddings.npz under {root}")

    parts, names, section_ids = [], [], []
    for path in paths:
        bundle = np.load(path, allow_pickle=True)
        parts.append(np.asarray(bundle[embedding], dtype=np.float32))
        names.append(np.asarray(bundle["obs_names"], dtype=str))
        section_ids.append(np.repeat(path.parent.name, len(names[-1])))

    matrix = np.vstack(parts)
    barcodes = pd.Index(np.concatenate(names))
    xen_obs = load_xenium_obs()
    positions = pd.Index(np.asarray(xen_obs.index, dtype=str)).get_indexer(barcodes)
    matched = positions >= 0
    if not matched.any():
        raise ValueError(
            f"No barcode overlap between {kind} handoff and the Xenium obs "
            f"(npz e.g. {barcodes[0]!r}, obs e.g. {xen_obs.index[0]!r})."
        )

    matrix = matrix[matched]
    barcodes = barcodes[matched]
    rows = positions[matched]
    frame = pd.DataFrame(
        {
            "Subclass": xen_obs["Subclass"].astype(str).to_numpy()[rows],
            "ADNC": xen_obs["Overall AD neuropathological Change"]
                    .astype(str).to_numpy()[rows],
            "donor": xen_obs["Donor ID"].astype(str).to_numpy()[rows],
            "section": np.concatenate(section_ids)[matched],
        },
        index=barcodes,
    )
    # The Xenium labels keep the "_Subclass" suffix the snRNA side already dropped.
    frame["Subclass"] = frame["Subclass"].str.replace(r"_?Subclass$", "", regex=True)

    idx = _subsample(len(frame), n_cells, seed)
    latent = AnnData(X=matrix[idx], obs=frame.iloc[idx].copy())
    print(f"[latent] Xenium {kind}/{embedding}: {latent.shape}, "
          f"{latent.obs['section'].nunique()} sections, "
          f"{latent.obs['donor'].nunique()} donors "
          f"({int(matched.sum()):,}/{len(matched):,} barcodes matched)")
    return latent


def umap_latent(latent, n_neighbors=15, min_dist=0.3, seed=0):
    """Neighbour graph + UMAP directly on the latent coordinates in `X`."""
    sc.pp.neighbors(latent, n_neighbors=n_neighbors, use_rep="X", random_state=seed)
    sc.tl.umap(latent, min_dist=min_dist, random_state=seed)
    return latent


def plot_latent_umap(ax, latent, color, title, legend_loc="right margin",
                     size=None, palette=None):
    """One latent UMAP panel, rasterized (these carry 10^5 points)."""
    sc.pl.umap(latent, color=color, ax=ax, show=False, title=title,
               legend_loc=legend_loc, palette=palette,
               size=size if size is not None else 120_000 / max(latent.n_obs, 1),
               frameon=True)
    for collection in ax.collections:
        collection.set_rasterized(True)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    return ax


def FIG_student_latent_umaps(snrna_latent, xenium_latent, *, embedding="student",
                             output_file=None):
    """2x2: snRNA and Xenium student latent spaces, by Subclass and by ADNC.

    Rows are assays, columns are labellings. The two assays are embedded
    independently, so only within-row comparisons are meaningful.
    """
    assign_shared_colors([snrna_latent, xenium_latent], "Subclass")
    for latent in (snrna_latent, xenium_latent):
        present = [lvl for lvl in ADNC_ORDER if lvl in set(latent.obs["ADNC"])]
        latent.obs["ADNC"] = pd.Categorical(latent.obs["ADNC"], categories=present,
                                            ordered=True)
        latent.uns["ADNC_colors"] = [
            ADNC_COLORS[ADNC_ORDER.index(lvl)] for lvl in present
        ]

    with plt.rc_context(FIG_RC):
        fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.0), constrained_layout=True)
        panels = [
            (axes[0, 0], snrna_latent, "Subclass", f"snRNA-seq {embedding} — Subclass"),
            (axes[0, 1], snrna_latent, "ADNC", f"snRNA-seq {embedding} — ADNC"),
            (axes[1, 0], xenium_latent, "Subclass", f"Xenium {embedding} — Subclass"),
            (axes[1, 1], xenium_latent, "ADNC", f"Xenium {embedding} — ADNC"),
        ]
        for ax, latent, color, title in panels:
            plot_latent_umap(ax, latent, color, title)
        return _finish_svg(fig, output_file)


def add_latent_dim_columns(latent, prefix="dim"):
    """Expose each latent coordinate as an obs column so sc.pl.umap can colour by it."""
    matrix = np.asarray(latent.X)
    names = [f"{prefix}{i}" for i in range(matrix.shape[1])]
    for i, name in enumerate(names):
        latent.obs[name] = matrix[:, i]
    return names


def FIG_latent_dimension_umaps(latent, title, *, ncols=5, cmap="RdBu_r",
                               clip_percentile=99.0, size=None, output_file=None):
    """One UMAP per student latent dimension, coloured by that dimension's value.

    Each dimension gets its own symmetric colour scale centred at zero, so the sign
    is readable; scales are NOT shared between panels because the dimensions have
    no common unit.
    """
    names = add_latent_dim_columns(latent)
    nrows = int(np.ceil(len(names) / ncols))
    with plt.rc_context(FIG_RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(3.2 * ncols, 3.4 * nrows),
                                 constrained_layout=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, name in zip(axes, names):
            values = latent.obs[name].to_numpy()
            limit = float(np.percentile(np.abs(values), clip_percentile)) or 1.0
            sc.pl.umap(
                latent, color=name, ax=ax, show=False, title=name,
                cmap=cmap, vmin=-limit, vmax=limit, colorbar_loc="right",
                size=size if size is not None else 120_000 / max(latent.n_obs, 1),
                frameon=True,
            )
            for collection in ax.collections:
                collection.set_rasterized(True)
            ax.set_xlabel("")
            ax.set_ylabel("")
        for ax in axes[len(names):]:
            ax.set_visible(False)
        fig.suptitle(title)
        return _finish_svg(fig, output_file)


# --------------------------------------------------------------------------
# Ordinal decoder: student latent -> ADNC
#
# ADNC is a DONOR-level label, so the latent is aggregated to the donor before
# modelling. Fitting across cells would pseudo-replicate the outcome ~17,000x
# per donor and manufacture significance.
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# psupertime on genes: cell-level proportional-odds fit, then GSEA of the gene
# coefficients against the Factor 1 RNA loadings.
#
# statsmodels' OrderedModel uses numerical derivatives and will not fit ~1000
# predictors at cell scale, so the cumulative logit is implemented directly with
# an analytic gradient. Validated against OrderedModel to ~1e-5 on the
# coefficients; a 60k x 1057 fit takes about 1.5 s, which is what makes the
# label-permutation null affordable.
# --------------------------------------------------------------------------

def _polr_unpack(params, n_features, n_levels):
    beta = params[:n_features]
    raw = params[n_features:]
    alphas = np.concatenate(([raw[0]], raw[0] + np.cumsum(np.exp(raw[1:]))))
    return beta, alphas, raw


def _polr_nll_grad(params, X, y, n_levels, l2):
    from scipy.special import expit

    beta, alphas, raw = _polr_unpack(params, X.shape[1], n_levels)
    eta = X @ beta
    edges = np.concatenate(([-np.inf], alphas, [np.inf]))
    c_hi, c_lo = expit(edges[y + 1] - eta), expit(edges[y] - eta)
    prob = np.clip(c_hi - c_lo, 1e-12, None)
    d_hi, d_lo = c_hi * (1 - c_hi), c_lo * (1 - c_lo)

    nll = -np.sum(np.log(prob)) + l2 * (beta @ beta)
    g_beta = X.T @ ((d_hi - d_lo) / prob) + 2 * l2 * beta

    g_alpha = np.zeros(n_levels - 1)
    upper = y < n_levels - 1
    lower = y > 0
    np.add.at(g_alpha, y[upper], (-d_hi / prob)[upper])
    np.add.at(g_alpha, (y - 1)[lower], (d_lo / prob)[lower])

    jac = np.zeros((n_levels - 1, n_levels - 1))
    jac[:, 0] = 1.0
    for k in range(1, n_levels - 1):
        jac[k:, k] = np.exp(raw[k])
    return nll, np.concatenate([g_beta, jac.T @ g_alpha])


def fit_polr(X, y, n_levels, l2=1.0, maxiter=500):
    """Proportional-odds cumulative logit, P(y <= k) = sigmoid(alpha_k - x'beta)."""
    from scipy.optimize import minimize

    counts = np.bincount(y, minlength=n_levels).astype(float)
    cumulative = np.clip(np.cumsum(counts) / counts.sum(), 1e-6, 1 - 1e-6)[:-1]
    alphas = np.log(cumulative / (1 - cumulative))
    start = np.concatenate([
        np.zeros(X.shape[1]), alphas[:1],
        np.log(np.clip(np.diff(alphas), 1e-3, None)),
    ])
    res = minimize(_polr_nll_grad, start, args=(X, y, n_levels, l2), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    beta, thresholds, _ = _polr_unpack(res.x, X.shape[1], n_levels)
    return {"beta": beta, "thresholds": thresholds, "nll": float(res.fun),
            "success": bool(res.success), "n": len(y), "l2": l2}


def load_mofa_rna_cells(n_cells=60_000, subclass=None, seed=0, chunk=50_000,
                        levels=None):
    """Cell subsample of the MOFA RNA training matrix, with donor/Subclass/ADNC.

    Reads `data/rna/group1` from snrna_mofa.h5mu, which is the matrix the Factor 1
    loadings were fitted on. Do NOT substitute scrna_msi_mudata.mod["rna"]: that is
    the 1681-gene bundle and its columns do not correspond to these loadings.
    """
    import h5py

    levels = levels or ADNC_ORDER
    with h5py.File(CAH_MOFA_PATH, "r") as h5:
        genes = np.array([g.decode().replace("rna:", "") for g in h5["features/rna"][:]])
        n_total = h5["data/rna/group1"].shape[0]
        meta = {k: h5[f"samples_metadata/group1/rna:{k}"][:] for k in
                ("Donor ID", "Subclass", "Age at Death", "Sex", "PMI",
                 "Overall AD neuropathological Change")}

    decode = lambda a: np.array([x.decode() if isinstance(x, bytes) else x for x in a])
    frame = pd.DataFrame({
        "donor": decode(meta["Donor ID"]),
        "subclass": pd.Series(decode(meta["Subclass"]))
                      .str.replace(r"Subclass$", "", regex=True).to_numpy(),
        "age": meta["Age at Death"].astype(float),
        "sex_male": (decode(meta["Sex"]) == "Male").astype(float),
        "pmi": pd.to_numeric(pd.Series(decode(meta["PMI"])), errors="coerce").to_numpy(),
        "adnc": decode(meta["Overall AD neuropathological Change"]),
    })

    eligible = frame["adnc"].isin(levels).to_numpy() & np.isfinite(frame["pmi"].to_numpy())
    if subclass is not None:
        eligible &= (frame["subclass"] == subclass).to_numpy()
    candidates = np.flatnonzero(eligible)
    rng = np.random.default_rng(seed)
    if n_cells and n_cells < len(candidates):
        candidates = np.sort(rng.choice(candidates, n_cells, replace=False))

    matrix = np.empty((len(candidates), len(genes)), dtype=np.float32)
    with h5py.File(CAH_MOFA_PATH, "r") as h5:
        data = h5["data/rna/group1"]
        cursor = 0
        for start in range(0, n_total, chunk):
            end = min(start + chunk, n_total)
            take = candidates[(candidates >= start) & (candidates < end)]
            if not len(take):
                continue
            matrix[cursor:cursor + len(take)] = data[start:end][take - start]
            cursor += len(take)

    cells = AnnData(X=matrix,
                    obs=frame.iloc[candidates].reset_index(drop=True),
                    var=pd.DataFrame(index=pd.Index(genes, name=None)))
    cells.obs.index = cells.obs.index.astype(str)
    print(f"[genes] {cells.shape} cells x genes, "
          f"{cells.obs['donor'].nunique()} donors"
          + (f", Subclass={subclass!r}" if subclass else ""))
    return cells


def fit_gene_ordinal_adnc(cells, levels=None, l2=10.0, covariates=("age", "sex_male", "pmi"),
                          verbose=True):
    """psupertime-style cumulative logit of ADNC on gene expression, across cells."""
    levels = levels or [l for l in ADNC_ORDER if l in set(cells.obs["adnc"])]
    genes = list(cells.var_names)
    design = np.asarray(cells.X, dtype=np.float64)
    if covariates:
        extra = cells.obs[list(covariates)].to_numpy(dtype=float)
        extra = (extra - extra.mean(axis=0)) / np.where(extra.std(axis=0) == 0, 1.0,
                                                        extra.std(axis=0))
        design = np.hstack([design, extra])
    y = pd.Categorical(cells.obs["adnc"], categories=levels, ordered=True).codes

    fit = fit_polr(design, y.astype(int), len(levels), l2=l2)
    coefficients = pd.Series(fit["beta"][:len(genes)], index=genes, name="coef")
    if verbose:
        print(f"[psupertime-genes] n = {fit['n']:,} cells, {len(genes)} genes, "
              f"l2 = {l2}, converged = {fit['success']}")
        top = coefficients.sort_values()
        print("  most negative:", ", ".join(f"{g} ({v:+.3f})"
                                            for g, v in top.head(5).items()))
        print("  most positive:", ", ".join(f"{g} ({v:+.3f})"
                                            for g, v in top.tail(5)[::-1].items()))
    return coefficients, fit


def gsea_enrichment_score(ranking, members, weight=1.0):
    """Classic weighted-KS running enrichment score for one gene set."""
    order = np.argsort(-np.asarray(ranking, dtype=float))
    hits = np.asarray(members, dtype=bool)[order]
    magnitude = np.abs(np.asarray(ranking, dtype=float))[order] ** weight
    hit_total = magnitude[hits].sum()
    miss_total = max(int((~hits).sum()), 1)
    if hit_total == 0:
        return 0.0
    steps = np.where(hits, magnitude / hit_total, -1.0 / miss_total)
    running = np.cumsum(steps)
    return float(running[np.argmax(np.abs(running))])


def gsea_factor1_test(cells, mofa_model=mofa, factor="Factor1", top_n=100,
                      l2=10.0, n_perm=200, seed=0, covariates=("age", "sex_male", "pmi")):
    """Are the ADNC gene coefficients enriched for the Factor 1 loading extremes?

    The null permutes the DONOR -> ADNC mapping and refits, because donors are the
    exchangeable unit; permuting gene labels instead would ignore both the
    donor structure and gene-gene correlation.
    """
    weights = (mofa_model.get_weights(df=True, views=["rna"])[factor]
               .dropna())
    weights.index = weights.index.str.replace(r"^rna:", "", regex=True)
    weights = weights.reindex(list(cells.var_names)).dropna()
    genes = list(weights.index)

    observed, _ = fit_gene_ordinal_adnc(cells, l2=l2, covariates=covariates)
    observed = observed.reindex(genes)

    sets = {
        f"Factor1 top +{top_n}": weights.nlargest(top_n).index,
        f"Factor1 top -{top_n}": weights.nsmallest(top_n).index,
    }
    membership = {name: np.isin(genes, list(index)) for name, index in sets.items()}
    stats = {name: gsea_enrichment_score(observed.to_numpy(), flags)
             for name, flags in membership.items()}
    rho = spearmanr(observed.to_numpy(), weights.to_numpy()).statistic

    levels = [l for l in ADNC_ORDER if l in set(cells.obs["adnc"])]
    donor_label = cells.obs.groupby("donor")["adnc"].first()
    rng = np.random.default_rng(seed)
    null_es = {name: np.empty(n_perm) for name in sets}
    null_rho = np.empty(n_perm)

    shuffled_cells = cells.copy()
    for b in range(n_perm):
        mapping = dict(zip(donor_label.index, rng.permutation(donor_label.to_numpy())))
        shuffled_cells.obs["adnc"] = cells.obs["donor"].map(mapping).to_numpy()
        permuted, _ = fit_gene_ordinal_adnc(shuffled_cells, levels=levels, l2=l2,
                                            covariates=covariates, verbose=False)
        permuted = permuted.reindex(genes).to_numpy()
        for name, flags in membership.items():
            null_es[name][b] = gsea_enrichment_score(permuted, flags)
        null_rho[b] = spearmanr(permuted, weights.to_numpy()).statistic
        if (b + 1) % 25 == 0:
            print(f"  permutation {b + 1}/{n_perm}")

    rows = []
    for name, value in stats.items():
        null = null_es[name]
        rows.append({"gene_set": name, "size": int(membership[name].sum()),
                     "ES": value,
                     "p_perm": (np.sum(np.abs(null) >= abs(value)) + 1) / (n_perm + 1),
                     "null_sd": float(null.std())})
    rows.append({"gene_set": "rho(coef, loading)", "size": len(genes), "ES": rho,
                 "p_perm": (np.sum(np.abs(null_rho) >= abs(rho)) + 1) / (n_perm + 1),
                 "null_sd": float(null_rho.std())})
    return pd.DataFrame(rows), observed, weights


def fit_cell_ordinal_adnc(latent, dims=None, levels=None, n_fit=200_000, seed=0,
                          verbose=True):
    """psupertime-style cumulative logit: ADNC ~ student latent, fitted across CELLS.

    Following Macnair et al., the donor-level ordinal label is broadcast onto every
    cell and the model is fitted at cell resolution. The deliverable is the per-cell
    linear predictor (a supervised pseudotime), not the coefficient table.

    What this buys: a continuous per-cell ADNC axis that can be mapped back onto
    tissue. What it costs: the standard errors, p-values and pseudo-R2 of the fit
    are meaningless, because ~17,000 cells share each donor's label. Judge the model
    by donor-held-out prediction (crossval_cell_ordinal), never by its summary().
    """
    matrix = np.asarray(latent.X, dtype=np.float64)
    dims = dims or [f"dim{i}" for i in range(matrix.shape[1])]
    levels = levels or [l for l in ADNC_ORDER if l in set(latent.obs["ADNC"])]

    keep = pd.Series(latent.obs["ADNC"].astype(str).values).isin(levels).to_numpy()
    idx = np.flatnonzero(keep)
    rng = np.random.default_rng(seed)
    if n_fit and n_fit < len(idx):
        idx = np.sort(rng.choice(idx, n_fit, replace=False))

    train = matrix[idx]
    center, scale = train.mean(axis=0), train.std(axis=0)
    scale[scale == 0] = 1.0
    exog = pd.DataFrame((train - center) / scale, columns=dims)
    endog = pd.Series(
        pd.Categorical(latent.obs["ADNC"].astype(str).values[idx],
                       categories=levels, ordered=True)
    )
    model = OrderedModel(endog, exog, distr="logit")

    # OrderedModel's default start values make the exp()-parameterised thresholds
    # overflow on some folds, which silently returns all-zero coefficients rather
    # than raising. Starting from the marginal cumulative logits (betas at zero)
    # is well inside the feasible region.
    counts = endog.value_counts().reindex(levels).to_numpy(dtype=float)
    cumulative = np.cumsum(counts) / counts.sum()
    alphas = np.log(cumulative[:-1] / (1.0 - cumulative[:-1]))
    start = np.concatenate([np.zeros(len(dims)), alphas[:1], np.log(np.diff(alphas))])

    res, converged = None, False
    for method in ("bfgs", "lbfgs", "nm"):
        try:
            candidate = model.fit(method=method, start_params=start, disp=False,
                                  maxiter=2000)
        except Exception:  # noqa: BLE001 - try the next optimiser
            continue
        if bool(candidate.mle_retvals.get("converged")) and \
                np.isfinite(np.asarray(candidate.params, dtype=float)).all():
            res, converged = candidate, True
            break
        res = res or candidate
    if res is None:
        raise RuntimeError("Ordinal fit failed under every optimiser.")

    decoder = {"res": res, "dims": dims, "levels": levels,
               "center": center, "scale": scale,
               "beta": np.asarray(res.params[:len(dims)], dtype=float),
               "n_fit": len(idx), "converged": converged,
               "donors_fit": int(pd.Series(latent.obs["donor"].values[idx]).nunique())}
    if verbose:
        print(f"[psupertime] fitted on {len(idx):,} cells from "
              f"{decoder['donors_fit']} donors, {len(levels)} ADNC levels; "
              f"converged={converged}")
        print("  coefficients (higher score -> higher ADNC):")
        for name, value in zip(dims, decoder["beta"]):
            print(f"    {name:6s} {value:+.3f}")
    if not converged:
        print("  [warn] no optimiser converged; coefficients are not trustworthy.")
    return decoder


def apply_cell_ordinal(decoder, latent_or_matrix, prefix="adnc", write=True):
    """Per-cell score and class probabilities from a fitted cell-level decoder.

    Returns a DataFrame with `<prefix>_score` (the linear predictor / supervised
    pseudotime), `<prefix>_expected` (probability-weighted mean level) and one
    `<prefix>_p_<level>` column per ADNC level.
    """
    latent = latent_or_matrix if hasattr(latent_or_matrix, "obs") else None
    matrix = np.asarray(latent.X if latent is not None else latent_or_matrix,
                        dtype=np.float64)
    standardised = (matrix - decoder["center"]) / decoder["scale"]
    res, levels = decoder["res"], decoder["levels"]

    score = standardised @ decoder["beta"]
    probabilities = np.asarray(res.model.predict(res.params, exog=standardised))
    codes = np.arange(len(levels), dtype=float)

    frame = pd.DataFrame(
        {f"{prefix}_score": score,
         f"{prefix}_expected": probabilities @ codes},
        index=(latent.obs_names if latent is not None else None),
    )
    for j, level in enumerate(levels):
        frame[f"{prefix}_p_{level.replace(' ', '')}"] = probabilities[:, j]
    if write and latent is not None:
        for column in frame.columns:
            latent.obs[column] = frame[column].to_numpy()
    return frame


def crossval_cell_ordinal(latent, dims=None, levels=None, n_splits=6,
                          n_fit_per_fold=80_000, n_perm=10_000, seed=0):
    """Donor-held-out evaluation of the cell-level ADNC decoder.

    Cells of a donor never appear in the fold that scores them, so the donor-level
    out-of-fold means are honest. Significance is assessed by permuting the 42 donor
    labels against those means, which is the only exchangeable unit here.
    """
    from sklearn.model_selection import GroupKFold

    matrix = np.asarray(latent.X, dtype=np.float64)
    dims = dims or [f"dim{i}" for i in range(matrix.shape[1])]
    levels = levels or [l for l in ADNC_ORDER if l in set(latent.obs["ADNC"])]
    donors = latent.obs["donor"].astype(str).to_numpy()
    labels = latent.obs["ADNC"].astype(str).to_numpy()

    keep = np.isin(labels, levels)
    oof = np.full(latent.n_obs, np.nan)
    failed = []
    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(donors[keep]))))
    for fold, (train_idx, test_idx) in enumerate(
            splitter.split(matrix[keep], labels[keep], groups=donors[keep]), start=1):
        absolute_train = np.flatnonzero(keep)[train_idx]
        absolute_test = np.flatnonzero(keep)[test_idx]
        sub = AnnData(X=matrix[absolute_train],
                      obs=pd.DataFrame({"ADNC": labels[absolute_train],
                                        "donor": donors[absolute_train]}))
        decoder = fit_cell_ordinal_adnc(sub, dims=dims, levels=levels,
                                        n_fit=n_fit_per_fold, seed=seed + fold,
                                        verbose=False)
        if not decoder["converged"]:
            # Leaving these NaN drops the fold's donors from the evaluation. Scoring
            # them with a failed fit would enter ~1/n_splits of the donors at an
            # all-zero coefficient vector and bias the correlation toward zero.
            failed.append(fold)
            print(f"  fold {fold}: FAILED to converge - "
                  f"{len(absolute_test):,} cells left unscored")
            continue
        standardised = (matrix[absolute_test] - decoder["center"]) / decoder["scale"]
        oof[absolute_test] = standardised @ decoder["beta"]
        print(f"  fold {fold}: trained on {decoder['donors_fit']} donors, "
              f"scored {len(absolute_test):,} cells")

    latent.obs["adnc_score_oof"] = oof
    donor_frame = (pd.DataFrame({"donor": donors, "ADNC": labels, "score": oof})
                   .dropna(subset=["score"])
                   .groupby("donor")
                   .agg(score=("score", "mean"), ADNC=("ADNC", "first")))
    donor_frame["code"] = pd.Categorical(donor_frame["ADNC"], categories=levels,
                                         ordered=True).codes
    rho = spearmanr(donor_frame["score"], donor_frame["code"]).statistic

    rng = np.random.default_rng(seed)
    codes = donor_frame["code"].to_numpy()
    scores = donor_frame["score"].to_numpy()
    null = np.array([spearmanr(scores, rng.permutation(codes)).statistic
                     for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1)
    if failed:
        print(f"[psupertime CV] {len(failed)} of {splitter.get_n_splits()} folds "
              f"failed to converge (folds {failed}); their donors are excluded.")
    print(f"[psupertime CV] donor-level rho = {rho:+.3f}, permutation p = {p:.4f} "
          f"(n = {len(donor_frame)} donors, null sd = {null.std():.3f})")
    return donor_frame, rho, p


def attach_student_latent(section, section_id=None, kind=SPATIALMETA_XENIUM_KIND,
                          embedding="student", key=None):
    """Attach a SpatialMETA latent to a Xenium section by barcode, into `obsm`."""
    section_id = section_id or CAH_XENIUM_SECTION_ID
    donor = ".".join(section_id.split(".")[:3])
    path = (CAH_OUTPUT_ROOT / "spatialmeta_handoff" / "outputs" / kind / donor
            / section_id / "embeddings.npz")
    bundle = np.load(path, allow_pickle=True)
    barcodes = pd.Index(np.asarray(bundle["obs_names"], dtype=str))
    matrix = np.asarray(bundle[embedding], dtype=np.float32)

    positions = barcodes.get_indexer(pd.Index(np.asarray(section.obs_names, dtype=str)))
    matched = positions >= 0
    if not matched.any():
        raise ValueError(f"No barcode overlap between {path.name} and the section.")
    out = np.full((section.n_obs, matrix.shape[1]), np.nan, dtype=np.float32)
    out[matched] = matrix[positions[matched]]
    key = key or f"X_spatialmeta_{embedding}"
    section.obsm[key] = out
    print(f"[latent] attached {embedding} to section.obsm[{key!r}]; "
          f"{int(matched.sum()):,}/{section.n_obs:,} cells matched")
    return section


def attach_adnc_decoder_scores(section, decoder, key=None, prefix="adnc"):
    """Score a Xenium section's cells with a fitted cell-level ADNC decoder."""
    key = key or f"X_spatialmeta_student"
    if key not in section.obsm:
        attach_student_latent(section, key=key)
    matrix = np.asarray(section.obsm[key], dtype=np.float64)
    finite = np.isfinite(matrix).all(axis=1)
    frame = apply_cell_ordinal(decoder, matrix[finite], prefix=prefix, write=False)
    for column in frame.columns:
        values = np.full(section.n_obs, np.nan)
        values[finite] = frame[column].to_numpy()
        section.obs[column] = values

    # Cross-assay use is out-of-distribution: the student embeddings separate the
    # two platforms, so report the shift rather than assuming it away.
    print(f"[psupertime] scored {int(finite.sum()):,}/{section.n_obs:,} cells; "
          f"score range {np.nanmin(section.obs[f'{prefix}_score']):+.2f} to "
          f"{np.nanmax(section.obs[f'{prefix}_score']):+.2f}")
    return section


def donor_latent_features(latent, donor_meta=None, subclass=None, min_cells=50):
    """Donor x latent-dimension means, joined to donor covariates and ADNC.

    `subclass=None` averages over all cells of a donor, which makes the feature
    vector partly a composition summary; pass a subclass name to hold cell type
    fixed instead.
    """
    frame = latent.obs.copy()
    matrix = np.asarray(latent.X)
    dims = [f"dim{i}" for i in range(matrix.shape[1])]
    for i, name in enumerate(dims):
        frame[name] = matrix[:, i]
    if subclass is not None:
        frame = frame[frame["Subclass"].astype(str) == subclass]
        if frame.empty:
            raise ValueError(f"No cells with Subclass == {subclass!r}.")

    counts = frame.groupby("donor").size().rename("n_cells")
    table = frame.groupby("donor")[dims].mean().join(counts)
    table = table[table["n_cells"] >= min_cells]

    if donor_meta is None:
        donor_meta = (cell_df.groupby("donor")
                      .agg(age=("age", "first"), sex=("sex", "first"),
                           pmi=("pmi", "first"),
                           adnc=(PRIMARY, "first")))
        donor_meta["sex_male"] = (donor_meta["sex"] == "Male").astype(float)
    table = table.join(donor_meta, how="inner")

    composition = donor_composition(cell_df)
    table = table.join(composition, how="left")
    return table.dropna(subset=["adnc", "age", "pmi"]), dims


def ordinal_latent_decoder(table, dims, *, covariates=("age", "sex_male", "pmi"),
                           composition=(), levels=None, verbose=True):
    """Cumulative-logit ADNC ~ latent dims (+ covariates), fitted across donors."""
    levels = levels or [l for l in ADNC_ORDER if l in set(table["adnc"])]
    predictors = list(dims) + list(covariates) + list(composition)
    X = table[predictors].astype(float).copy()
    for col in X.columns:
        if col == "sex_male":
            continue
        spread = X[col].std()
        X[col] = (X[col] - X[col].mean()) / (spread if spread else 1.0)
    y = pd.Series(pd.Categorical(table["adnc"], categories=levels, ordered=True),
                  index=table.index)
    try:
        res = OrderedModel(y, X, distr="logit").fit(method="bfgs", disp=False)
    except Exception as error:  # noqa: BLE001 - report rather than abort a sweep
        if verbose:
            print(f"  [decoder] fit failed: {type(error).__name__}: {error}")
        return None
    if verbose:
        print(f"  n = {len(table)} donors, {len(predictors)} predictors, "
              f"{len(levels)} ADNC levels")
        print(f"  pseudo-R2 = {res.prsquared:.3f}, LLR p = {res.llr_pvalue:.4f}")
    return res


def loo_decode_adnc(table, feature_sets, *, levels=None, n_perm=500, seed=0):
    """Leave-one-donor-out ADNC prediction from each feature set, vs a permutation null.

    Held-out prediction is what makes this interpretable: a donor-level ordinal fit
    with ten predictors on ~40 donors will look good in-sample no matter what.
    Ridge on the ordinal code is used as the decoder so the comparison across
    feature sets is on equal, regularised footing.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    levels = levels or [l for l in ADNC_ORDER if l in set(table["adnc"])]
    y = pd.Categorical(table["adnc"], categories=levels, ordered=True).codes.astype(float)
    rng = np.random.default_rng(seed)
    alphas = np.logspace(-3, 3, 25)

    def _loo(X, target):
        pred = np.empty(len(target))
        for i in range(len(target)):
            train = np.ones(len(target), bool)
            train[i] = False
            scaler = StandardScaler().fit(X[train])
            model = RidgeCV(alphas=alphas).fit(scaler.transform(X[train]), target[train])
            pred[i] = model.predict(scaler.transform(X[i:i + 1]))[0]
        return pred

    rows = []
    for name, columns in feature_sets.items():
        X = table[list(columns)].astype(float).to_numpy()
        pred = _loo(X, y)
        rho = spearmanr(pred, y).statistic
        mae = float(np.mean(np.abs(np.clip(np.rint(pred), 0, len(levels) - 1) - y)))
        null = np.empty(n_perm)
        for b in range(n_perm):
            null[b] = spearmanr(_loo(X, rng.permutation(y)), y).statistic
        p = (np.sum(null >= rho) + 1) / (n_perm + 1)
        rows.append({"features": name, "n_features": len(columns),
                     "loo_rho": rho, "loo_mae_levels": mae,
                     "p_perm": p, "null_sd": float(null.std())})
        print(f"  {name:24s} rho={rho:+.3f}  MAE={mae:.2f}  p={p:.4f}")
    return pd.DataFrame(rows).sort_values("p_perm")


def FIG_cah_spatial_row(section, output_file=None, smooth=30):
    """Tissue -> imputed metabolite -> multi-omic axis, in one Xenium section.

    Complementary illustration that the transferred signal can be placed in tissue;
    no spatial association is claimed (see the session's negative controls).
    """
    if "X_spatial" not in section.obsm:
        section.obsm["X_spatial"] = section.obsm["spatial"]
    if "dens_oligo" not in section.obs:
        raise KeyError("Run the oligodendrocyte-density KDE before this figure.")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    # The density panel is unsmoothed, so it needs the largest markers to close the
    # gaps between nuclei; otherwise the background reads as white speckle.
    plot_spatial_continuous(axes[0], section, "dens_oligo",
                            "local oligodendrocyte density",
                            cmap="magma", diverging=False, smooth=0, size=10.0)
    plot_spatial_continuous(axes[1], section, "imputed_dopamine",
                            f"imputed dopamine ({smooth}-NN mean)", smooth=smooth,
                            size=8.0)
    plot_spatial_continuous(axes[2], section, "mofa_factor1",
                            f"projected Factor 1 ({smooth}-NN mean)", smooth=smooth,
                            size=8.0)
    return _finish_svg(fig, output_file)


try:
    cah_xenium_factor_section = load_xenium_factor_section()
    cah_xenium_factor_section = attach_xenium_imputed_msi(cah_xenium_factor_section)
    cah_xenium_factor_section.obs["dens_oligo"] = local_category_density(
        cah_xenium_factor_section, "Oligodendrocyte_Subclass"
    )
except Exception as _error:  # noqa: BLE001 - the top row is still useful without them
    print(f"[FIG] spatial panels skipped ({type(_error).__name__}: {_error})")
    cah_xenium_factor_section = None

FIG_cah_transfer_main(
    xenium_section=cah_xenium_factor_section,
    output_file=CAH_OUTPUT_ROOT / "figures" / "cah_transfer_main.svg",
)
FIG_cah_transfer_supplementary(
    output_file=CAH_OUTPUT_ROOT / "figures" / "cah_transfer_supplementary.png",
)


# %% spatial cross-correlation between the three panels of the Xenium row
def _moran_zscore(v):
    v = np.asarray(v, dtype=float).ravel()
    return (v - np.nanmean(v)) / np.nanstd(v)


def spatial_weights(coords, n_neighbors=6):
    """Symmetric binary kNN adjacency, matching the SMA Visium convention.

    Not row-standardised, so ``S0`` counts edges rather than cells and the leading
    ``n / S0`` factor of Moran's I does not reduce to one.
    """
    from sklearn.neighbors import kneighbors_graph

    W = kneighbors_graph(np.asarray(coords)[:, :2], n_neighbors=n_neighbors,
                         mode="connectivity", include_self=False)
    return W.maximum(W.T)


def bivariate_morans_i(x, y, W):
    """Bivariate Moran's I of x against the spatial lag of y.

    Same estimator as ``3__sea_ad_cah_spatial_snrna_dopamine.py`` and
    ``rna_msi/dopamine_nmf_signature.py``:
    I = (n / S0) * (z_x . W z_y) / (z_x . z_x). Both arguments are z-scored, so the
    denominator is n and the statistic reduces to (z_x . W z_y) / S0, which is
    symmetric in x and y whenever W is; only one orientation is reported.
    """
    zx, zy = _moran_zscore(x), _moran_zscore(y)
    n, S0 = len(zx), float(W.sum())
    return float((n / S0) * (zx @ (W @ zy)) / (zx @ zx))


def bivariate_moran_permutation(x, y, W, n_permutations=999, seed=0):
    """Permutation p-value for bivariate Moran's I, shuffling y only.

    Free permutation destroys the autocorrelation of y as well as its pairing with
    x, so this null is liberal: it tests "no spatial cross-correlation" against a
    spatially structureless alternative, and cannot separate genuine coupling from
    two independently smooth fields. Read it alongside the residualised variant.
    """
    rng = np.random.default_rng(seed)
    observed = bivariate_morans_i(x, y, W)
    y = np.asarray(y, dtype=float)
    null = np.array([bivariate_morans_i(x, rng.permutation(y), W)
                     for _ in range(n_permutations)])
    p = (1 + np.sum(np.abs(null) >= abs(observed))) / (1 + n_permutations)
    return observed, float(p), null


def _residualise(values, covariates):
    """Least-squares residuals of ``values`` on an intercept plus ``covariates``."""
    X = np.column_stack([np.ones(len(values))] + [np.asarray(c, float).ravel()
                                                  for c in covariates])
    y = np.asarray(values, dtype=float).ravel()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def cah_xenium_spatial_crosscorrelation(section, n_neighbors=6, smooth=30,
                                        n_permutations=999, seed=0,
                                        output_file=None):
    """Bivariate Moran's I among the three fields plotted in the Xenium row.

    Quantifies what Fig. 7d-f only shows qualitatively: how strongly local
    oligodendrocyte density (d) co-varies in space with SpatialMETA-imputed
    dopamine (e) and with the projected Factor 1 score (f). The fields are taken
    exactly as plotted -- density unsmoothed, the other two averaged over the same
    ``smooth``-nearest-neighbour sets used by ``plot_spatial_continuous`` -- so the
    numbers describe the published panels rather than a separate analysis.

    Each pair is reported with the univariate Moran's I of each field for context and
    again after residualising both fields on log library size. Section 6.3 of
    CAH_SECTION_REWRITE.md records that the apparent density/Factor 1 relationship
    did not survive that adjustment in the per-cell (non-spatial) analysis, so the
    residualised column is the one to quote if the raw and adjusted values disagree.

    Both the as-plotted fields and their unsmoothed counterparts are scored. The
    plotted fields are already averaged over 30 spatial neighbours, which drives
    their autocorrelation close to one and makes the bivariate statistic almost
    indistinguishable from Pearson r; the unsmoothed rows are what separate genuine
    spatial cross-correlation from per-cell correlation.
    """
    if "X_spatial" not in section.obsm:
        section.obsm["X_spatial"] = section.obsm["spatial"]
    raw = {
        "oligo_density": section.obs["dens_oligo"].to_numpy(dtype=float),
        "imputed_dopamine": section.obs["imputed_dopamine"].to_numpy(dtype=float),
        "projected_factor1": section.obs["mofa_factor1"].to_numpy(dtype=float),
    }
    # The density panel is drawn unsmoothed, so it is the same field in both variants.
    plotted = {
        "oligo_density": raw["oligo_density"],
        "imputed_dopamine": smooth_over_neighbours(section, "imputed_dopamine",
                                                   n_neighbors=smooth),
        "projected_factor1": smooth_over_neighbours(section, "mofa_factor1",
                                                    n_neighbors=smooth),
    }
    depth = np.log1p(section.obs["total_counts_raw"].to_numpy(dtype=float))

    finite = np.isfinite(depth)
    for variant in (raw, plotted):
        for values in variant.values():
            finite &= np.isfinite(values)
    print(f"[Moran] {int(finite.sum()):,}/{section.n_obs:,} cells with all three "
          f"fields finite")

    coords = np.asarray(section.obsm["X_spatial"])[:, :2][finite]
    W = spatial_weights(coords, n_neighbors=n_neighbors)

    pairs = [("oligo_density", "imputed_dopamine"),
             ("oligo_density", "projected_factor1"),
             ("imputed_dopamine", "projected_factor1")]
    rows, univariate_rows = [], []
    for variant_name, variant in (("as plotted", plotted), ("unsmoothed", raw)):
        variant = {name: values[finite] for name, values in variant.items()}
        residual = {name: _residualise(values, [depth[finite]])
                    for name, values in variant.items()}
        univariate_rows += [
            {"variant": variant_name, "field": name,
             "univariate_I": bivariate_morans_i(values, values, W)}
            for name, values in variant.items()
        ]
        for a, b in pairs:
            observed, p_value, _ = bivariate_moran_permutation(
                variant[a], variant[b], W, n_permutations, seed
            )
            rows.append({
                "variant": variant_name,
                "field_a": a,
                "field_b": b,
                "bivariate_I": observed,
                "p_permutation": p_value,
                "bivariate_I_depth_adjusted": bivariate_morans_i(
                    residual[a], residual[b], W
                ),
                "pearson_r": float(np.corrcoef(variant[a], variant[b])[0, 1]),
            })
    table = pd.DataFrame(rows)
    univariate = pd.DataFrame(univariate_rows)
    print(univariate.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_file, index=False)
        univariate.to_csv(output_file.with_name(output_file.stem + "_univariate.csv"),
                          index=False)
        print(f"[Moran] wrote {output_file}")
    return table, univariate


if cah_xenium_factor_section is not None:
    cah_moran_table, cah_moran_univariate = cah_xenium_spatial_crosscorrelation(
        cah_xenium_factor_section,
        output_file=CAH_OUTPUT_ROOT / "figures" / "cah_xenium_bivariate_moran.csv",
    )


# %%
rna_sub = rna[::2].copy()

sc.pp.neighbors(rna_sub, n_neighbors=10, use_rep="X_spatialmeta_student")
sc.tl.umap(rna_sub)

strd_mask = rna.obs["Supertype"].eq('STRd D2 Striosome MSNSupertype')
iroot_idx = rna_sub.obs_names.get_indexer(
    [rna.obs[strd_mask]["mofa_factor1"].idxmax()]
)[0]
rna_sub.uns["iroot"] = iroot_idx
sc.tl.diffmap(rna_sub)
sc.tl.dpt(rna_sub, n_branchings=0) # n_branchings=1, if want to learn splits to Astro-Epend and Oligo

iroot_idx = rna_sub.uns["iroot"]
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for i, col in enumerate(["dpt_pseudotime", "mofa_factor1", "Neighborhood"] + (["dpt_groups"] if "dpt_groups" in rna_sub.obsm.keys() else [])):
    sc.pl.umap(
        rna_sub, 
        color=col, 
        ax=axs[i], 
        show=False, 
        size=50
    )
    # Highlight iroot point
    x, y = rna_sub.obsm['X_umap'][iroot_idx]
    axs[i].scatter(
        [x], [y], 
        s=150, 
        c='none', 
        edgecolors='red', 
        linewidths=2, 
        marker='o', 
        label='iroot'
    )
    axs[i].legend(loc='best')
plt.tight_layout()
plt.show()

#sc.tl.leiden(rna_sub, resolution=0.4)
#mask = rna_sub.obs['leiden'].isin(['0','1','2','3','5','6','8'])
#sc.pl.umap(rna_sub[mask], color=['dpt_pseudotime', 'mofa_factor1', 'rna:GFAP', "Subclass"], ncols=2, size=80)

#%%

d2_msn_dpt_agg_df, d2_msn_dpt_agg_sem, d2_msn_dpt_agg = plot_subclass_dpt_trends(
    rna_sub,
    "STR D2 MSN",
    genes=[
        ["rna:ST18", "rna:FOS", "rna:JUN", "rna:JUNB", "rna:NR4A1", "rna:VGF"],
        ["rna:GRIA4", "rna:GRIN1", "rna:HIP1", "rna:BIN1", "rna:PICALM"],
    ],
    q=5,
    plot_umap=True,
    leiden_groups=["0", "1", "2", "3", "5", "6", "8", "9"],
    umap_size=60,
)
# %%
st18_gaba_dpt_agg_df, st18_gaba_dpt_agg_sem, st18_gaba_dpt_agg = plot_subclass_dpt_trends(
    rna_sub,
    "CN ST18 GABA",
    genes=[
        ["rna:ST18", "rna:FOS", "rna:JUN", "rna:JUNB", "rna:NR4A1", "rna:VGF"],
        ["rna:GRIA4", "rna:GRIN1", "rna:HIP1", "rna:BIN1", "rna:PICALM"],
        ["rna:CLU", "rna:CAMK2D", "rna:CAMK2N1", "rna:NEFM"],
    ],
    q=5,
    plot_umap=True,
    leiden_groups=["0", "1", "2", "3", "5", "6", "8", "9"],
    umap_size=60,
)

# %%
endo_dpt_agg_df, endo_dpt_agg_sem, endo_dpt_agg = plot_subclass_dpt_trends(
    rna_sub,
    "Endothelial",
    genes=[
        ["rna:VEGFA", "rna:RAP1A", "rna:NRP1", "rna:NRP2", "rna:ANGPT2", "rna:ANGPTL4"],
        ["rna:A2M", "rna:IFITM3", "rna:SERPINA3", "rna:SLC2A1", "rna:ABCB1", "rna:CLDN5", "rna:COL4A1"],
    ],
    q=5,
    plot_umap=True,
    leiden_groups=["0", "1", "2", "3", "5", "6", "8", "9"],
    umap_genes=["rna:VEGFA", "rna:RAP1A", "rna:NRP1", "rna:NRP2", "rna:ANGPT2", "rna:ANGPTL4"],
    umap_size=80,
    umap_ncols=3,
)

# %%
vlmc_dpt_agg_df, vlmc_dpt_agg_sem, vlmc_dpt_agg = plot_subclass_dpt_trends(
    rna_sub,
    "VLMC",
    genes=[
        ["rna:PDGFRB", "rna:RGS5", "rna:ACTA2"]
        + ["rna:DCN", "rna:COL1A2", "rna:PDGFRA", "rna:PTGDS"]
        + ["rna:NOTCH4", "rna:NOTCH2"],
    ],
    q=5,
    plot_umap=True,
    leiden_groups=["0", "1", "2", "3", "5", "6", "8", "9"],
    umap_size=80,
    umap_ncols=3,
)



# %%

# %% Vascular WM-end pseudotime gradients: pathology interaction test
# -------------------------------------------------------------------
# Question: do the white-matter-skewed endothelial/VLMC gradients steepen in CPS-high
# donors, or are they constitutive anatomy / identity-depth artifacts?
#
# The interaction term is the result:
#   z(gene) ~ z(DPT) * CPS_high + z(depth) + z(identity_marker)
#             + z(Age) + Sex + APOE4 + (1 | donor)
#
# CPS is split high/low within the donors available for each outcome. The current script
# consumes the per-donor CPS_AT8/CPS_6E10 table imported above. If AT8/6E10 CPS is
# rederived from Xenium-native IF, replace `vascular_cps_by_donor` here with that table;
# no HALO-derived field is used in this block.

import warnings


VASCULAR_DPT_KEY = "dpt_pseudotime"
VASCULAR_DONOR_KEY = "Donor ID"
VASCULAR_DEPTH_CANDIDATES = ("total_counts", "Number of UMIs", "n_counts", "transcript_counts")
VASCULAR_CPS_OUTCOMES = ("CPS_AT8", "CPS_6E10")
VASCULAR_MIN_DONORS = 8
VASCULAR_N_DPT_BINS = 5

VASCULAR_GENE_SETS = {
    "Endothelial": {
        "identity_marker": "rna:PECAM1",
        "genes": ["rna:RAP1A", "rna:NRP1", "rna:NRP2"],
        "adjust_arteriolosclerosis": False,
    },
    "VLMC": {
        "identity_marker": "rna:PDGFRB",
        "genes": ["rna:NOTCH2", "rna:ACTA2", "rna:COL1A2"],
        "adjust_arteriolosclerosis": True,
    },
}

vascular_cps_by_donor = kana2026_cps_by_donor.loc[:, list(VASCULAR_CPS_OUTCOMES)].copy()


def _first_present_column(obs, candidates):
    for col in candidates:
        if col in obs.columns:
            return col
    raise KeyError(f"None of these columns are present in adata.obs: {list(candidates)}")


def _as_vector(x):
    if sp.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def _gene_vector(adata, gene):
    if gene not in adata.var_names:
        raise KeyError(f"{gene} is absent from adata.var_names")
    return _as_vector(adata[:, gene].X)


def _zscore_series(values):
    values = pd.to_numeric(pd.Series(values), errors="coerce").astype(float)
    sd = values.std()
    if not np.isfinite(sd) or sd == 0:
        return None
    return (values - values.mean()) / sd


def _apoe4_carrier(values):
    return pd.Series(values).astype("string").str.contains("4", regex=False, na=False).astype(float)


def _arteriolosclerosis_score(values):
    mapping = {
        "None": 0,
        "Not observed": 0,
        "Not Observed": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
    }
    as_num = pd.to_numeric(pd.Series(values), errors="coerce")
    mapped = pd.Series(values).astype("string").map(mapping)
    return as_num.fillna(mapped)


def _split_cps_high_low(donor_cps):
    """Return donor -> Low/High after sorting by CPS; keeps 10 donors as 5 low / 5 high."""
    donor_cps = donor_cps.dropna().astype(float).sort_values()
    if donor_cps.empty:
        return pd.Series(dtype="object")
    split = len(donor_cps) // 2
    low = set(donor_cps.index[:split])
    return pd.Series(
        np.where(donor_cps.index.isin(low), "Low", "High"),
        index=donor_cps.index,
        dtype="object",
    )


def prepare_vascular_gradient_model_data(
    adata,
    *,
    subclass,
    gene,
    identity_marker,
    cps_outcome,
    adjust_arteriolosclerosis=False,
):
    required_obs = [VASCULAR_DONOR_KEY, "Subclass", VASCULAR_DPT_KEY, "Age at Death", "Sex", "APOE Genotype"]
    missing_obs = sorted(set(required_obs) - set(adata.obs.columns))
    if missing_obs:
        raise KeyError(f"Missing required obs columns for vascular model: {missing_obs}")
    if cps_outcome not in vascular_cps_by_donor.columns:
        raise KeyError(f"{cps_outcome} is absent from vascular_cps_by_donor")

    depth_col = _first_present_column(adata.obs, VASCULAR_DEPTH_CANDIDATES)
    mask = adata.obs["Subclass"].eq(subclass)
    if not mask.any():
        raise ValueError(f"No cells found for Subclass == {subclass!r}")

    needed_genes = list(dict.fromkeys([gene, identity_marker]))
    missing_genes = sorted(set(needed_genes) - set(adata.var_names))
    if missing_genes:
        raise KeyError(f"Missing required vascular model genes: {missing_genes}")

    sub = adata[mask, needed_genes].copy()
    data = pd.DataFrame({
        "donor": sub.obs[VASCULAR_DONOR_KEY].astype(str).values,
        "subclass": subclass,
        "gene": gene,
        "identity_marker_gene": identity_marker,
        "pseudotime": pd.to_numeric(sub.obs[VASCULAR_DPT_KEY], errors="coerce").values,
        "depth": pd.to_numeric(sub.obs[depth_col], errors="coerce").values,
        "age": pd.to_numeric(sub.obs["Age at Death"], errors="coerce").values,
        "sex_male": sub.obs["Sex"].astype(str).eq("Male").astype(float).values,
        "apoe4": _apoe4_carrier(sub.obs["APOE Genotype"]).values,
        "expr": _gene_vector(sub, gene),
        "identity_marker": _gene_vector(sub, identity_marker),
    }, index=sub.obs_names)
    data[cps_outcome] = data["donor"].map(vascular_cps_by_donor[cps_outcome])

    if adjust_arteriolosclerosis and "Arteriolosclerosis" in sub.obs.columns:
        data["arteriolosclerosis"] = _arteriolosclerosis_score(sub.obs["Arteriolosclerosis"]).values

    data["log_counts"] = np.log1p(data["depth"].clip(lower=0))
    donor_groups = _split_cps_high_low(data.groupby("donor")[cps_outcome].first())
    data["CPS_group"] = data["donor"].map(donor_groups)
    data["cps_high"] = data["CPS_group"].eq("High").astype(float)

    model_cols = [
        "expr", "identity_marker", "pseudotime", "log_counts", "age", "sex_male",
        "apoe4", "donor", "CPS_group", "cps_high", cps_outcome,
    ]
    if "arteriolosclerosis" in data:
        model_cols.append("arteriolosclerosis")
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=model_cols).copy()

    z_cols = {
        "expr_z": "expr",
        "identity_marker_z": "identity_marker",
        "pseudotime_z": "pseudotime",
        "log_counts_z": "log_counts",
        "age_z": "age",
    }
    if "arteriolosclerosis" in data:
        z_cols["arteriolosclerosis_z"] = "arteriolosclerosis"
    for out_col, in_col in z_cols.items():
        z = _zscore_series(data[in_col])
        if z is None:
            raise ValueError(f"Cannot z-score {in_col} for {subclass} {gene} {cps_outcome}")
        data[out_col] = z.to_numpy()

    if data["donor"].nunique() < VASCULAR_MIN_DONORS:
        raise ValueError(
            f"{subclass} {gene} {cps_outcome}: only {data['donor'].nunique()} donors after filtering"
        )
    if data["CPS_group"].nunique() < 2:
        raise ValueError(f"{subclass} {gene} {cps_outcome}: CPS split has fewer than two groups")

    return data


def fit_vascular_gradient_mixed_model(data, *, adjust_arteriolosclerosis=False):
    fixed_terms = ["pseudotime_z * cps_high", "log_counts_z", "identity_marker_z"]
    optional_terms = ["age_z", "sex_male", "apoe4"]
    if adjust_arteriolosclerosis and "arteriolosclerosis_z" in data:
        optional_terms.append("arteriolosclerosis_z")

    dropped_terms = []
    for term in optional_terms:
        if data[term].nunique(dropna=True) > 1:
            fixed_terms.append(term)
        else:
            dropped_terms.append(term)

    formula = "expr_z ~ " + " + ".join(fixed_terms)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.MixedLM.from_formula(
                formula,
                groups="donor",
                data=data,
            ).fit(reml=False, method="lbfgs", maxiter=500, disp=False)
        if not fit.converged:
            fit = sm.MixedLM.from_formula(
                formula,
                groups="donor",
                data=data,
            ).fit(reml=False, method="powell", maxiter=1000, disp=False)
    except Exception as err:
        return None, formula, dropped_terms, str(err)

    interaction_term = "pseudotime_z:cps_high"
    if interaction_term not in fit.params.index:
        return None, formula, dropped_terms, f"Missing interaction term {interaction_term!r}"

    # For the companion plot, remove depth/identity/demographic/injury variation but leave
    # pseudotime, CPS group, and their interaction in the displayed residual.
    covariate_terms = [
        term for term in ["log_counts_z", "identity_marker_z", "age_z", "sex_male", "apoe4", "arteriolosclerosis_z"]
        if term in data.columns and data[term].nunique(dropna=True) > 1
    ]
    cov_x = sm.add_constant(data[covariate_terms], has_constant="add")
    data["expr_adjusted"] = sm.OLS(data["expr_z"], cov_x).fit().resid

    ci = fit.conf_int()
    low_slope = float(fit.params.get("pseudotime_z", np.nan))
    interaction = float(fit.params[interaction_term])
    out = {
        "subclass": data["subclass"].iloc[0],
        "gene": data["gene"].iloc[0],
        "identity_marker": data["identity_marker_gene"].iloc[0],
        "cps_outcome": [c for c in VASCULAR_CPS_OUTCOMES if c in data.columns][0],
        "n_cells": int(data.shape[0]),
        "n_donors": int(data["donor"].nunique()),
        "n_low_donors": int(data.loc[data["CPS_group"].eq("Low"), "donor"].nunique()),
        "n_high_donors": int(data.loc[data["CPS_group"].eq("High"), "donor"].nunique()),
        "low_cps_slope": low_slope,
        "high_cps_slope": low_slope + interaction,
        "interaction_beta": interaction,
        "interaction_se": float(fit.bse[interaction_term]),
        "interaction_ci_low": float(ci.loc[interaction_term, 0]),
        "interaction_ci_high": float(ci.loc[interaction_term, 1]),
        "interaction_p": float(fit.pvalues[interaction_term]),
        "random_intercept_var": float(fit.cov_re.iloc[0, 0]) if fit.cov_re.size else np.nan,
        "formula": formula,
        "dropped_terms": ", ".join(dropped_terms),
        "fit_warnings": " | ".join(sorted({str(w.message) for w in caught})) if "caught" in locals() else "",
    }
    return out, formula, dropped_terms, None


def summarize_vascular_cell_counts(adata):
    subclasses = list(VASCULAR_GENE_SETS)
    counts = (
        adata.obs.loc[adata.obs["Subclass"].isin(subclasses)]
        .groupby(["Subclass", VASCULAR_DONOR_KEY], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    print("\n=== Per-donor vascular cell counts in DPT object ===")
    if counts.empty:
        print("No Endothelial/VLMC cells found.")
        return counts
    print(
        counts.groupby("Subclass")["n_cells"]
        .agg(["min", "median", "max", "sum"])
        .round(1)
        .to_string()
    )
    rare = counts[counts["n_cells"] < 20]
    if not rare.empty:
        print("\nDonor/subclass combinations with <20 cells:")
        print(rare.sort_values(["Subclass", "n_cells"]).to_string(index=False))
    return counts


vascular_cell_counts = summarize_vascular_cell_counts(rna_sub)

vascular_gradient_rows = []
vascular_gradient_model_data = {}
vascular_gradient_failures = []

for subclass, spec in VASCULAR_GENE_SETS.items():
    for gene in spec["genes"]:
        for cps_outcome in VASCULAR_CPS_OUTCOMES:
            try:
                model_data = prepare_vascular_gradient_model_data(
                    rna_sub,
                    subclass=subclass,
                    gene=gene,
                    identity_marker=spec["identity_marker"],
                    cps_outcome=cps_outcome,
                    adjust_arteriolosclerosis=spec["adjust_arteriolosclerosis"],
                )
                row, formula, dropped_terms, error = fit_vascular_gradient_mixed_model(
                    model_data,
                    adjust_arteriolosclerosis=spec["adjust_arteriolosclerosis"],
                )
            except Exception as err:
                row, formula, dropped_terms, error = None, None, None, str(err)

            if row is None:
                vascular_gradient_failures.append({
                    "subclass": subclass,
                    "gene": gene,
                    "cps_outcome": cps_outcome,
                    "error": error,
                })
                continue

            vascular_gradient_rows.append(row)
            vascular_gradient_model_data[(gene, cps_outcome)] = model_data

vascular_gradient_mixed_results = pd.DataFrame(vascular_gradient_rows)
if not vascular_gradient_mixed_results.empty:
    vascular_gradient_mixed_results["interaction_q"] = multipletests(
        vascular_gradient_mixed_results["interaction_p"],
        method="fdr_bh",
    )[1]
    vascular_gradient_mixed_results["interpretation"] = np.select(
        [
            (vascular_gradient_mixed_results["interaction_beta"] > 0)
            & (vascular_gradient_mixed_results["interaction_p"] < 0.05),
            (vascular_gradient_mixed_results["interaction_p"] >= 0.05),
        ],
        [
            "AD-associated WM-end steepening",
            "no interaction: constitutive/anatomy or underpowered",
        ],
        default="negative/reversed interaction",
    )
    vascular_gradient_mixed_results = vascular_gradient_mixed_results.sort_values(
        ["cps_outcome", "subclass", "interaction_p", "gene"]
    ).reset_index(drop=True)

    print("\n=== Vascular WM-end DPT gradient x CPS mixed models ===")
    display_cols = [
        "cps_outcome", "subclass", "gene", "identity_marker", "n_cells", "n_donors",
        "n_low_donors", "n_high_donors", "low_cps_slope", "high_cps_slope",
        "interaction_beta", "interaction_ci_low", "interaction_ci_high",
        "interaction_p", "interaction_q", "interpretation", "dropped_terms",
    ]
    print(vascular_gradient_mixed_results[display_cols].round(4).to_string(index=False))

if vascular_gradient_failures:
    print("\n=== Vascular CPS mixed-model failures ===")
    print(pd.DataFrame(vascular_gradient_failures).to_string(index=False))


def plot_vascular_cps_pseudotime_lines(model_data, *, cps_outcome, q=VASCULAR_N_DPT_BINS):
    keys = [(gene, outcome) for (gene, outcome) in model_data if outcome == cps_outcome]
    if not keys:
        print(f"No fitted vascular model data available for {cps_outcome}")
        return None

    keys = sorted(keys)
    ncols = 3
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    palette = {"Low": "#4C78A8", "High": "#D65F5F"}

    for ax, key in zip(axs.ravel(), keys):
        gene, _ = key
        data = model_data[key].copy()
        data["dpt_bin"] = pd.qcut(data["pseudotime"], q=q, labels=False, duplicates="drop")
        donor_bin = (
            data.dropna(subset=["dpt_bin"])
            .groupby(["CPS_group", "donor", "dpt_bin"], observed=True)
            .agg(
                y=("expr_adjusted", "mean"),
                x=("pseudotime", "mean"),
            )
            .reset_index()
        )
        summary = (
            donor_bin.groupby(["CPS_group", "dpt_bin"], observed=True)
            .agg(
                y_mean=("y", "mean"),
                y_sem=("y", "sem"),
                x_mean=("x", "mean"),
                n_donors=("donor", "nunique"),
            )
            .reset_index()
        )

        for group in ["Low", "High"]:
            g = summary[summary["CPS_group"].eq(group)].sort_values("x_mean")
            if g.empty:
                continue
            yerr = 1.96 * g["y_sem"].fillna(0)
            ax.plot(
                g["x_mean"],
                g["y_mean"],
                marker="o",
                linewidth=2,
                color=palette[group],
                label=f"{group} CPS",
            )
            ax.fill_between(
                g["x_mean"],
                g["y_mean"] - yerr,
                g["y_mean"] + yerr,
                color=palette[group],
                alpha=0.18,
                linewidth=0,
            )

        ax.axhline(0, color="0.8", linewidth=1, zorder=0)
        ax.set_title(gene.replace("rna:", ""))
        ax.set_xlabel("DPT pseudotime")
        ax.set_ylabel("depth/identity-adjusted z(expr)")
        ax.legend(frameon=False, fontsize=8)

    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")

    fig.suptitle(f"Vascular WM-end gene gradients split by {cps_outcome}", y=1.02)
    plt.tight_layout()
    plt.show()
    return fig


for _cps_outcome in VASCULAR_CPS_OUTCOMES:
    plot_vascular_cps_pseudotime_lines(vascular_gradient_model_data, cps_outcome=_cps_outcome)

# %% Neuronal WM-end pseudotime gradients: ST18/oligo-adjusted CPS interaction test
# --------------------------------------------------------------------------------
# This is the neuronal counterpart to the vascular CPS interaction test. The priority
# tests are the CAMK2D/CAMK2N1 pair and the D2 MSN immediate-early-gene decline. The
# receptor genes and ST18-co-tracking genes are kept in a separate exploratory/confounded
# family because their raw DPT trends can reflect boundary/oligo contamination.

NEURONAL_IDENTITY_ADJUSTMENTS = {
    "ST18": ["rna:ST18"],
    "oligo_marker_mean": ["rna:MBP", "rna:MOG", "rna:PLP1", "rna:OLIG2"],
}
NEURONAL_PRIORITY_GENE_SETS = {
    "STR D2 MSN": ["rna:FOS", "rna:JUN", "rna:JUNB", "rna:NR4A1"],
    "CN ST18 GABA": ["rna:CAMK2D", "rna:CAMK2N1"],
}
NEURONAL_EXPLORATORY_GENE_SETS = {
    "STR D2 MSN": ["rna:GRIA4", "rna:GRIN1", "rna:HIP1", "rna:PICALM"],
    "CN ST18 GABA": ["rna:GRIA4", "rna:GRIN1", "rna:HIP1", "rna:PICALM"],
}
NEURONAL_RUN_EXPLORATORY_CONFOUNDED = True


def _mean_identity_covariate(adata, features):
    present = [feature for feature in features if feature in adata.var_names]
    missing = sorted(set(features) - set(present))
    if not present:
        raise KeyError(f"None of the requested identity features are present: {features}")
    values = np.vstack([_gene_vector(adata, feature) for feature in present])
    return values.mean(axis=0), present, missing


def prepare_neuronal_gradient_model_data(
    adata,
    *,
    subclass,
    gene,
    identity_features,
    identity_name,
    cps_outcome,
    family,
):
    required_obs = [VASCULAR_DONOR_KEY, "Subclass", VASCULAR_DPT_KEY, "Age at Death", "Sex", "APOE Genotype"]
    missing_obs = sorted(set(required_obs) - set(adata.obs.columns))
    if missing_obs:
        raise KeyError(f"Missing required obs columns for neuronal model: {missing_obs}")

    depth_col = _first_present_column(adata.obs, VASCULAR_DEPTH_CANDIDATES)
    mask = adata.obs["Subclass"].eq(subclass)
    if not mask.any():
        raise ValueError(f"No cells found for Subclass == {subclass!r}")

    needed_genes = list(dict.fromkeys([gene, *identity_features]))
    present_genes = [feature for feature in needed_genes if feature in adata.var_names]
    if gene not in present_genes:
        raise KeyError(f"{gene} is absent from adata.var_names")
    sub = adata[mask, present_genes].copy()
    identity_values, identity_present, identity_missing = _mean_identity_covariate(sub, identity_features)

    data = pd.DataFrame({
        "donor": sub.obs[VASCULAR_DONOR_KEY].astype(str).values,
        "subclass": subclass,
        "candidate_family": family,
        "gene": gene,
        "identity_marker_gene": identity_name,
        "identity_features_used": ", ".join(identity_present),
        "identity_features_missing": ", ".join(identity_missing),
        "pseudotime": pd.to_numeric(sub.obs[VASCULAR_DPT_KEY], errors="coerce").values,
        "depth": pd.to_numeric(sub.obs[depth_col], errors="coerce").values,
        "age": pd.to_numeric(sub.obs["Age at Death"], errors="coerce").values,
        "sex_male": sub.obs["Sex"].astype(str).eq("Male").astype(float).values,
        "apoe4": _apoe4_carrier(sub.obs["APOE Genotype"]).values,
        "expr": _gene_vector(sub, gene),
        "identity_marker": identity_values,
    }, index=sub.obs_names)
    data[cps_outcome] = data["donor"].map(vascular_cps_by_donor[cps_outcome])
    data["log_counts"] = np.log1p(data["depth"].clip(lower=0))
    donor_groups = _split_cps_high_low(data.groupby("donor")[cps_outcome].first())
    data["CPS_group"] = data["donor"].map(donor_groups)
    data["cps_high"] = data["CPS_group"].eq("High").astype(float)

    model_cols = [
        "expr", "identity_marker", "pseudotime", "log_counts", "age", "sex_male",
        "apoe4", "donor", "CPS_group", "cps_high", cps_outcome,
    ]
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=model_cols).copy()

    for out_col, in_col in {
        "expr_z": "expr",
        "identity_marker_z": "identity_marker",
        "pseudotime_z": "pseudotime",
        "log_counts_z": "log_counts",
        "age_z": "age",
    }.items():
        z = _zscore_series(data[in_col])
        if z is None:
            raise ValueError(f"Cannot z-score {in_col} for {subclass} {gene} {identity_name} {cps_outcome}")
        data[out_col] = z.to_numpy()

    if data["donor"].nunique() < VASCULAR_MIN_DONORS:
        raise ValueError(
            f"{subclass} {gene} {cps_outcome}: only {data['donor'].nunique()} donors after filtering"
        )
    if data["CPS_group"].nunique() < 2:
        raise ValueError(f"{subclass} {gene} {cps_outcome}: CPS split has fewer than two groups")

    return data


def summarize_neuronal_cell_counts(adata):
    subclasses = sorted(set(NEURONAL_PRIORITY_GENE_SETS) | set(NEURONAL_EXPLORATORY_GENE_SETS))
    counts = (
        adata.obs.loc[adata.obs["Subclass"].isin(subclasses)]
        .groupby(["Subclass", VASCULAR_DONOR_KEY], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    print("\n=== Per-donor neuronal candidate cell counts in DPT object ===")
    if counts.empty:
        print("No neuronal candidate subclasses found.")
        return counts
    print(
        counts.groupby("Subclass")["n_cells"]
        .agg(["min", "median", "max", "sum"])
        .round(1)
        .to_string()
    )
    return counts


neuronal_cell_counts = summarize_neuronal_cell_counts(rna_sub)

neuronal_gradient_rows = []
neuronal_gradient_model_data = {}
neuronal_gradient_failures = []

neuronal_model_plan = []
for subclass, genes in NEURONAL_PRIORITY_GENE_SETS.items():
    for gene in genes:
        for identity_name, identity_features in NEURONAL_IDENTITY_ADJUSTMENTS.items():
            neuronal_model_plan.append(("priority", subclass, gene, identity_name, identity_features))
if NEURONAL_RUN_EXPLORATORY_CONFOUNDED:
    for subclass, genes in NEURONAL_EXPLORATORY_GENE_SETS.items():
        for gene in genes:
            neuronal_model_plan.append(("exploratory_confounded", subclass, gene, "ST18", NEURONAL_IDENTITY_ADJUSTMENTS["ST18"]))

for family, subclass, gene, identity_name, identity_features in neuronal_model_plan:
    for cps_outcome in VASCULAR_CPS_OUTCOMES:
        try:
            model_data = prepare_neuronal_gradient_model_data(
                rna_sub,
                subclass=subclass,
                gene=gene,
                identity_features=identity_features,
                identity_name=identity_name,
                cps_outcome=cps_outcome,
                family=family,
            )
            row, formula, dropped_terms, error = fit_vascular_gradient_mixed_model(
                model_data,
                adjust_arteriolosclerosis=False,
            )
        except Exception as err:
            row, formula, dropped_terms, error = None, None, None, str(err)

        if row is None:
            neuronal_gradient_failures.append({
                "candidate_family": family,
                "subclass": subclass,
                "gene": gene,
                "identity_adjustment": identity_name,
                "cps_outcome": cps_outcome,
                "error": error,
            })
            continue

        row["candidate_family"] = family
        row["identity_adjustment"] = identity_name
        row["identity_features_used"] = model_data["identity_features_used"].iloc[0]
        row["identity_features_missing"] = model_data["identity_features_missing"].iloc[0]
        neuronal_gradient_rows.append(row)
        neuronal_gradient_model_data[(family, subclass, gene, identity_name, cps_outcome)] = model_data

neuronal_gradient_mixed_results = pd.DataFrame(neuronal_gradient_rows)
if not neuronal_gradient_mixed_results.empty:
    neuronal_gradient_mixed_results["interaction_q"] = multipletests(
        neuronal_gradient_mixed_results["interaction_p"],
        method="fdr_bh",
    )[1]
    neuronal_gradient_mixed_results["interpretation"] = np.select(
        [
            (neuronal_gradient_mixed_results["interaction_p"] < 0.05)
            & (neuronal_gradient_mixed_results["interaction_beta"] < 0),
            (neuronal_gradient_mixed_results["interaction_p"] < 0.05)
            & (neuronal_gradient_mixed_results["interaction_beta"] > 0),
            neuronal_gradient_mixed_results["interaction_p"] >= 0.05,
        ],
        [
            "CPS-high DPT slope decreases after identity adjustment",
            "CPS-high DPT slope increases after identity adjustment",
            "no interaction after identity adjustment",
        ],
        default="undetermined",
    )
    neuronal_gradient_mixed_results = neuronal_gradient_mixed_results.sort_values(
        ["candidate_family", "cps_outcome", "subclass", "identity_adjustment", "interaction_p", "gene"]
    ).reset_index(drop=True)

    print("\n=== Neuronal DPT gradient x CPS mixed models, identity-adjusted ===")
    neuronal_display_cols = [
        "candidate_family", "cps_outcome", "subclass", "gene", "identity_adjustment",
        "n_cells", "n_donors", "n_low_donors", "n_high_donors",
        "low_cps_slope", "high_cps_slope", "interaction_beta",
        "interaction_ci_low", "interaction_ci_high", "interaction_p",
        "interaction_q", "interpretation", "identity_features_missing", "dropped_terms",
    ]
    print(neuronal_gradient_mixed_results[neuronal_display_cols].round(4).to_string(index=False))

if neuronal_gradient_failures:
    print("\n=== Neuronal CPS mixed-model failures ===")
    print(pd.DataFrame(neuronal_gradient_failures).to_string(index=False))


def plot_neuronal_cps_pseudotime_lines(
    model_data,
    *,
    cps_outcome,
    family="priority",
    identity_adjustment="ST18",
    q=VASCULAR_N_DPT_BINS,
):
    keys = [
        key for key in model_data
        if key[0] == family and key[3] == identity_adjustment and key[4] == cps_outcome
    ]
    if not keys:
        print(f"No neuronal model data available for {family}, {identity_adjustment}, {cps_outcome}")
        return None

    keys = sorted(keys)
    ncols = 3
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    palette = {"Low": "#4C78A8", "High": "#D65F5F"}

    for ax, key in zip(axs.ravel(), keys):
        _, subclass, gene, _, _ = key
        data = model_data[key].copy()
        data["dpt_bin"] = pd.qcut(data["pseudotime"], q=q, labels=False, duplicates="drop")
        donor_bin = (
            data.dropna(subset=["dpt_bin"])
            .groupby(["CPS_group", "donor", "dpt_bin"], observed=True)
            .agg(y=("expr_adjusted", "mean"), x=("pseudotime", "mean"))
            .reset_index()
        )
        summary = (
            donor_bin.groupby(["CPS_group", "dpt_bin"], observed=True)
            .agg(y_mean=("y", "mean"), y_sem=("y", "sem"), x_mean=("x", "mean"))
            .reset_index()
        )

        for group in ["Low", "High"]:
            g = summary[summary["CPS_group"].eq(group)].sort_values("x_mean")
            if g.empty:
                continue
            yerr = 1.96 * g["y_sem"].fillna(0)
            ax.plot(
                g["x_mean"],
                g["y_mean"],
                marker="o",
                linewidth=2,
                color=palette[group],
                label=f"{group} CPS",
            )
            ax.fill_between(
                g["x_mean"],
                g["y_mean"] - yerr,
                g["y_mean"] + yerr,
                color=palette[group],
                alpha=0.18,
                linewidth=0,
            )

        ax.axhline(0, color="0.8", linewidth=1, zorder=0)
        ax.set_title(f"{subclass}: {gene.replace('rna:', '')}")
        ax.set_xlabel("DPT pseudotime")
        ax.set_ylabel("depth/identity-adjusted z(expr)")
        ax.legend(frameon=False, fontsize=8)

    for ax in axs.ravel()[len(keys):]:
        ax.axis("off")

    fig.suptitle(
        f"Neuronal priority gradients split by {cps_outcome} ({identity_adjustment}-adjusted)",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()
    return fig


for _cps_outcome in VASCULAR_CPS_OUTCOMES:
    plot_neuronal_cps_pseudotime_lines(
        neuronal_gradient_model_data,
        cps_outcome=_cps_outcome,
        family="priority",
        identity_adjustment="ST18",
    )

# %%
feats = scrna_msi_mudata.mod['msi'].var_names.sort_values().str.split(':').str[1]
labelled_feats = feats[pd.to_numeric(feats, errors='coerce').isna()].tolist()
labelled_feats = ['msi:' + feat for feat in labelled_feats]

msi_sub = scrna_msi_mudata[rna_sub.obs_names].mod['msi']
msi_sub.obsm['X_umap'] = rna_sub.obsm['X_umap']

sc.pl.umap(msi_sub, color=labelled_feats, size=80, ncols=3)

# %%

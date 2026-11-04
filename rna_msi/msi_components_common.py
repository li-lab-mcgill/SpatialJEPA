try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% MOFA-free versions of the Fig. 5 / Fig. 6 dopamine analyses, from PLS decoders.
#
# All post-hoc PLS decoders in the manuscript share one input space, the student latent:
#   MSI decoder   latent -> 779 metabolites, fitted on the source section's MEASURED MSI
#                 (student_pls_decoder.joblib; it imputes the targets' MSI)
#   RNA decoder   latent -> target genes, fitted here in each target (as in Fig. 2)
#   ATAC decoder  latent -> target peaks, fitted here in each target (as in Fig. 2)
# A direction in the latent therefore has metabolite, gene and peak loadings at once,
# with no MOFA factor to choose or orient:
#
#   dopamine axis   the MSI decoder's dopamine direction (imputed dopamine = z . beta_DA).
#                   Recovered exactly by regressing the stored imputed dopamine on the
#                   stored target latent, so it is available for the manuscript's June model
#                   too, whose decoder file was overwritten.
#   MSI-PLS C1-C4   the decoder's own components (runs whose decoder file reproduces the
#                   stored imputation), oriented so dopamine loads positively. Their
#                   metabolite loadings come from measured source MSI (non-circular).
#
# Gene/peak loadings of a direction (filter b in the target's standardised latent units)
# are the RNA/ATAC decoders' response along its activation pattern Sigma.b (Haufe et al.,
# 2014): g = Q R^T Sigma b / sd(score). With K = latent dim this equals corr(feature,
# score); the structure correlation is reported alongside as a check.
#
# Analyses (Fig. 5 / Fig. 6 counterparts):
#   metabolites   per decoder component: dopamine share, ranks of the catecholamine-pathway
#                 metabolites (Dopamine, Dopamine (single), 3-MT, Norepinephrine), and a
#                 permutation p for their mean rank
#   genes (5j)    AUROC of dopamine-axis gene loadings vs the NMF program's top-50, in both
#                 targets; MSN-identity baseline; identity-adjusted partial correlations
#   ATAC (6b)     weighted GSEA of D2-MSN / MXD cCRE peaks over Fig. 6's 2,000 peaks ranked
#                 by dopamine-axis peak loading (permutation NES, sign-specific null), and
#                 over all 66,520 peaks by AUROC; leading-edge BEDs for the motif test (6c)
#   RNA vs ATAC   peaks linked to genes (10x annotation): concordance of peak and gene
#                 loadings (promoter / distal, gene-shuffle null), and whether peaks linked
#                 to the NMF top-50 genes sit at the top of the ATAC ranking
#
# Env: $PY_ANALYSIS

import argparse
import os
import re
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

OUT = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs")
PROJ = OUT / "spatialjepa_projection"
MODELS = OUT / "spatialjepa_models"
DATA = Path(f"{os.environ['SPATIALJEPA_HOME']}/data")
BASE = "V11L12-109_B1"
RUNS = {
    "manuscript": "",
    "old-behaviour retrain": "__induced2000",
    "fixed s0": "__fix2000",
    "fixed s1": "__fix2000_s1",
    "fixed s2": "__fix2000_s2",
    "fixed s3": "__fix2000_s3",
    "seed3": "_seed3",   # fixed s3 with the early-stopped teacher (SPATIALJEPA_TEACHER_EARLY_STOP=1)
}
TARGETS = {
    "multiome": dict(h5mu="multiome_target_rna_atac_msi.h5mu", emb="multiome_target_rna_embedded.h5ad",
                     npz="multiome_target_student_msi_imputed.npz", cluster="REF_arc_gex_graphclust_Cluster",
                     mofa="multiome_target_mofa_model.hdf5"),
    "spatial": dict(h5mu="spatial_target_rna_atac_msi.h5mu", emb="target_spatial_atac_rna_embedded.h5ad",
                    npz="spatial_target_student_msi_imputed.npz", cluster="RNA_clusters",
                    mofa="spatial_target_mofa_model.hdf5"),
}
DA = "msi:Dopamine"
CATECHOLAMINES = ["msi:Dopamine", "msi:Dopamine (single)", "msi:3-MT", "msi:Norepinephrine"]
K_FEATURE = 9          # latent -> RNA / ATAC decoders, as in the manuscript's Fig. 2
TOP_K = 50
N_PERM = 1000
MARKERS = {
    "MSN": ["Ppp1r1b", "Penk", "Tac1", "Drd1", "Drd2", "Adora2a", "Gpr88", "Rgs9", "Pde10a", "Pde1b"],
    "oligodendrocyte": ["Mobp", "Mog", "Mag", "Plp1", "Mbp", "Cldn11", "Mal", "Ermn"],
    "OPC": ["Pdgfra", "Cspg4", "Olig1"],
    "astrocyte": ["Gja1", "Aqp4", "Slc1a3", "Aldh1l1", "Gfap"],
    "microglia": ["Cx3cr1", "P2ry12", "Csf1r", "Hexb"],
    "pan-neuronal": ["Snap25", "Syt1", "Rbfox3", "Stmn2"],
    "interneuron": ["Sst", "Pvalb", "Npy", "Chat", "Lhx6"],
    "cortical excitatory": ["Slc17a7", "Satb2", "Neurod6"],
}
CCRE_SETS = {"D2MSN": lambda ct: ct.str.contains("D2MSN"), "MXD": lambda ct: ct.eq("MXD")}
BEDTOOLS_BIN = os.environ['SPATIALJEPA_BEDTOOLS_BIN']


def parse_args():
    p = argparse.ArgumentParser(description="MOFA-free PLS dopamine-axis analyses (Figs. 5-6).")
    p.add_argument("--out-dir", type=Path, default=OUT / "rna_msi_runs" / "pls_dopamine_axis")
    p.add_argument("--runs", nargs="+", default=list(RUNS), choices=list(RUNS))
    p.add_argument("--validate", action="store_true", help="check the PLS solver against sklearn")
    return p.parse_args()


# ----------------------------------------------------------------------------- PLS
def column_stats(Y):
    """Column means and sample SDs of a dense or sparse matrix."""
    n = Y.shape[0]
    mean = np.asarray(Y.mean(axis=0)).ravel()
    sq = np.asarray((Y.multiply(Y) if sp.issparse(Y) else Y ** 2).mean(axis=0)).ravel()
    var = np.maximum(sq - mean ** 2, 0.0) * n / (n - 1)
    return mean, np.sqrt(var)


def pls_crosscov(X, Y, K, y_stats=None):
    """PLSRegression(K, scale=True) from the cross-covariance alone.

    Every quantity of NIPALS PLS2 regression is a function of X0 and C = X0^T Y0, because
    the deflated X_{k-1} is orthogonal to earlier scores (so X_{k-1}^T Y_{k-1} =
    X_{k-1}^T Y0) and q_k = Y0^T t_k / t_k^T t_k. The converged NIPALS weight is the top
    left singular vector of X_{k-1}^T Y0. Y can be sparse and very wide (66k peaks).
    Returns x mean/std, R (D x K, x_rotations), Q (F x K, y_loadings), keep mask over Y.
    """
    xm, xs = X.mean(0), X.std(0, ddof=1)
    X0 = (X - xm) / xs
    ym, ys = y_stats if y_stats is not None else column_stats(Y)
    keep = ys > 0
    C = np.asarray((Y.T @ X0).T if sp.issparse(Y) else X0.T @ Y)[:, keep] / ys[keep]
    Xk, M = X0.copy(), C.copy()
    W, P, Q = [], [], []
    for _ in range(K):
        evals, evecs = np.linalg.eigh(M @ M.T)
        w = evecs[:, -1]
        w = w * (np.sign(w[np.argmax(np.abs(w))]) or 1.0)   # sklearn's _svd_flip_1d
        t = Xk @ w
        tt = t @ t
        p = Xk.T @ t / tt
        q = M.T @ w / tt
        Xk = Xk - np.outer(t, p)
        M = M - np.outer(p, q * tt)
        W.append(w), P.append(p), Q.append(q)
    W, P, Q = np.array(W).T, np.array(P).T, np.array(Q).T
    R = W @ np.linalg.pinv(P.T @ W)
    return dict(xm=xm, xs=xs, X0=X0, R=R, Q=Q, keep=keep, C=C)


def validate_solver():
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(800, 10)) @ rng.normal(size=(10, 10))
    Y = X @ rng.normal(size=(10, 60)) + rng.normal(size=(800, 60))
    ours = pls_crosscov(X, Y, 6)
    ref = PLSRegression(6, scale=True, max_iter=5000, tol=1e-12).fit(X, Y)
    dR = np.abs(np.abs(ours["R"]) - np.abs(ref.x_rotations_)).max()
    dQ = np.abs(np.abs(ours["Q"]) - np.abs(ref.y_loadings_)).max()
    Bo = ours["R"] @ ours["Q"].T
    Br = ref.x_rotations_ @ ref.y_loadings_.T
    print(f"[pls] solver vs sklearn: max|dR|={dR:.1e} max|dQ|={dQ:.1e} max|dB|={np.abs(Bo - Br).max():.1e}",
          flush=True)
    assert np.abs(Bo - Br).max() < 1e-6


def axis_loadings(pls, b, Y, y_stats):
    """Decoded and structure-correlation feature loadings for a latent filter b."""
    X0, R, Q, keep, C = pls["X0"], pls["R"], pls["Q"], pls["keep"], pls["C"]
    n = X0.shape[0]
    sigma = X0.T @ X0 / (n - 1)
    sd_s = np.sqrt(b @ sigma @ b)
    decoded = np.full(len(keep), np.nan)
    decoded[keep] = Q @ (R.T @ (sigma @ b)) / sd_s
    corr = np.full(len(keep), np.nan)
    corr[keep] = (C.T @ b) / ((n - 1) * sd_s)
    return decoded, corr


# ------------------------------------------------------------------ partial correlations
def partial_corr(Y, y_stats, s, G):
    """corr(Y_j, s | G) for every column j; Y sparse or dense, G small dense covariates."""
    n = Y.shape[0]
    ym, ys = y_stats
    Gc = G - G.mean(0)
    sc = s - s.mean()
    YtG = np.asarray(Y.T @ Gc) if sp.issparse(Y) else Y.T @ Gc        # centering of Y not
    Yts = np.asarray(Y.T @ sc).ravel() if sp.issparse(Y) else Y.T @ sc  # needed: Gc, sc centred
    cov_yg, cov_ys = YtG / (n - 1), Yts / (n - 1)
    Sgg = Gc.T @ Gc / (n - 1)
    Sgg_inv = np.linalg.pinv(Sgg)
    cov_gs = Gc.T @ sc / (n - 1)
    res_cov = cov_ys - cov_yg @ Sgg_inv @ cov_gs
    res_var_y = ys ** 2 - np.einsum("ij,jk,ik->i", cov_yg, Sgg_inv, cov_yg)
    res_var_s = sc @ sc / (n - 1) - cov_gs @ Sgg_inv @ cov_gs
    with np.errstate(invalid="ignore", divide="ignore"):
        return res_cov / np.sqrt(np.maximum(res_var_y, 1e-12) * res_var_s)


def plain_corr(Y, y_stats, s):
    n = Y.shape[0]
    sc = s - s.mean()
    Yts = np.asarray(Y.T @ sc).ravel() if sp.issparse(Y) else Y.T @ sc
    with np.errstate(invalid="ignore", divide="ignore"):
        return Yts / ((n - 1) * y_stats[1] * sc.std(ddof=1))


# ----------------------------------------------------------------------------- data
def read_index(group):
    return group[group.attrs.get("_index", "_index")][:].astype(str)


def read_csr(group):
    shape = tuple(group.attrs["shape"])
    return sp.csr_matrix((group["data"][:].astype(np.float32), group["indices"][:], group["indptr"][:]),
                         shape=shape)


def read_obs_column(group, key):
    col = group[key]
    if isinstance(col, h5py.Group):
        return pd.Categorical.from_codes(col["codes"][:], col["categories"][:].astype(str)).astype(str)
    return col[:].astype(str)


def load_target(target):
    spec = TARGETS[target]
    with h5py.File(PROJ / BASE / spec["h5mu"], "r") as f:
        cells = read_index(f["mod/rna/obs"])
        genes = read_index(f["mod/rna/var"])
        rna = read_csr(f["mod/rna/layers/SCT_data"])
        clusters = np.asarray(read_obs_column(f["mod/rna/obs"], spec["cluster"]))
        if not np.array_equal(read_index(f["mod/atac/obs"]), cells):
            raise ValueError(f"{target}: RNA and ATAC cells differ")
        peaks = read_index(f["mod/atac/var"])
        atac = read_csr(f["mod/atac/X"])
    with h5py.File(PROJ / BASE / spec["mofa"], "r") as f:
        fig_peaks = f["features/atac"][:].astype(str)
        fig_genes = f["features/rna"][:].astype(str)
    print(f"[data] {target}: {len(cells):,} cells, {len(genes):,} genes, {len(peaks):,} peaks", flush=True)
    return dict(cells=pd.Index(cells), genes=pd.Index(genes), rna=rna.tocsc(), clusters=clusters,
                peaks=pd.Index(peaks), atac=atac.tocsc(), fig_peaks=pd.Index(fig_peaks),
                fig_genes=pd.Index([g.split(":", 1)[-1] for g in fig_genes]))


def module_scores(rna, genes):
    out = {}
    for name, panel in MARKERS.items():
        idx = [genes.get_loc(g) for g in panel if g in genes]
        if len(idx) < 2:
            continue
        z = rna[:, idx].toarray()
        out[name] = ((z - z.mean(0)) / (z.std(0) + 1e-8)).mean(1)
    return pd.DataFrame(out)


def load_run(tag, target, cells):
    """Target latent, stored imputed MSI and, if it reproduces them, the decoder."""
    import anndata as ad

    spec = TARGETS[target]
    emb = ad.read_h5ad(PROJ / (BASE + tag) / spec["emb"], backed="r")
    Z = pd.DataFrame(np.asarray(emb.obsm["student_X_emb"]), index=emb.obs_names).reindex(cells).to_numpy()
    with np.load(PROJ / (BASE + tag) / spec["npz"], allow_pickle=True) as z:
        msi = pd.DataFrame(z["student_msi"], index=z["obs_names"].astype(str),
                           columns=z["msi_feature_names"].astype(str)).reindex(cells)
    if np.isnan(Z).any() or msi.isna().any().any():
        raise ValueError(f"{tag} {target}: latent / imputed MSI missing for some cells")
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    beta_da = np.linalg.lstsq(Z1, msi[DA].to_numpy(), rcond=None)[0][1:]
    resid = np.abs(Z1[:, 1:] @ beta_da + np.linalg.lstsq(Z1, msi[DA].to_numpy(), rcond=None)[0][0]
                   - msi[DA].to_numpy()).max()
    payload = joblib.load(MODELS / (BASE + tag) / "student_pls_decoder.joblib")
    model = payload["model"]
    reproduces = np.abs(model.predict(Z) - msi[payload["target_features"]].to_numpy()).max() < 1e-3
    return dict(Z=Z, msi=msi, beta_da=beta_da, beta_resid=resid,
                decoder=model if reproduces else None, features=pd.Index(payload["target_features"]))


def decoder_components(model, features):
    """Latent filters (raw latent units) and oriented metabolite loadings per component."""
    R, Q = model.x_rotations_, model.y_loadings_
    xs = getattr(model, "_x_std", None)
    if xs is None:
        xs = model.x_std_
    comps = []
    da = features.get_loc(DA)
    for k in range(R.shape[1]):
        sign = np.sign(Q[da, k]) or 1.0
        comps.append(dict(name=f"MSI-PLS C{k + 1}", beta=sign * R[:, k] / xs, q=pd.Series(sign * Q[:, k], index=features)))
    return comps


# ------------------------------------------------------------------------- statistics
def auroc_stats(values, labels):
    ok = np.isfinite(values)
    v, l = values[ok], labels[ok]
    a = roc_auc_score(l, v)
    p = mannwhitneyu(v[l], v[~l], alternative="greater").pvalue
    return a, p


def nmf_program():
    import anndata as ad

    nmf = joblib.load(PROJ / BASE / "nmf_dopamine_spatial_xcorr_no_panel_model.joblib")
    with np.load(PROJ / BASE / "nmf_dopamine_spatial_xcorr_no_panel_metrics.npz") as m:
        best = int(np.argmax(m["bivariate_moran_I"]))
    joint = ad.read_h5ad(MODELS / BASE / "joint_adata_no_panel.h5ad", backed="r")
    st = joint.var_names[joint.var["type"].eq("ST")]
    prog = pd.Series(nmf.components_[best], index=[g.split(":", 1)[1].upper() for g in st])
    return prog.groupby(level=0).max()


def nmf_test(gene_values, genes, universe, prog):
    """Panel-j statistics for gene loadings restricted to `universe` (gene symbols)."""
    s = pd.Series(gene_values, index=genes.str.upper()).groupby(level=0).max()
    shared = s.index.intersection(pd.Index([g.upper() for g in universe])).intersection(prog.index)
    s = s.reindex(shared).dropna()
    top = set(prog.reindex(s.index).nlargest(TOP_K).index)
    labels = np.array([g in top for g in s.index])
    a, p = auroc_stats(s.to_numpy(), labels)
    overlap = len(top & set(s.nlargest(TOP_K).index))
    return dict(n_genes=len(s), auroc=a, p=p, overlap=overlap,
                overlap_p=float(hypergeom.sf(overlap - 1, len(s), TOP_K, TOP_K)),
                top_genes=", ".join(s.nlargest(8).index))


def weighted_gsea(scores, member, n_perm=N_PERM, seed=0):
    """Weighted (p=1) running-sum ES with a sign-specific permutation NES and p."""
    order = np.argsort(-scores)
    s, m = scores[order], member[order]
    w = np.abs(s)

    def es_of(mask):
        hit = np.where(mask, w, 0.0)
        run = np.cumsum(hit / hit.sum() - (~mask) / (~mask).sum())
        i = np.argmax(np.abs(run))
        return run[i], i

    es, peak = es_of(m)
    rng = np.random.default_rng(seed)
    null = np.array([es_of(rng.permutation(m))[0] for _ in range(n_perm)])
    same = null[np.sign(null) == np.sign(es)]
    nes = es / np.abs(same).mean() if len(same) else es / np.abs(null).mean()
    p = (np.sum(np.abs(same) >= abs(es)) + 1) / (n_perm + 1)
    leading = order[: peak + 1][m[: peak + 1]] if es > 0 else order[peak:][m[peak:]]
    return dict(es=es, nes=nes, p=p, n_set=int(member.sum()), n_leading=len(leading)), leading


def peaks_bed(names):
    df = pd.Series(names).str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    df["name"] = list(names)
    return df.dropna().astype({"start": int, "end": int})


def ccre_membership(peaks):
    import pybedtools

    os.environ["PATH"] = BEDTOOLS_BIN + os.pathsep + os.environ["PATH"]
    (table,) = (DATA / "CATlas" / "supplementary_tables").glob("Supplementary Table 8*.txt")
    cc = pd.read_csv(table, sep=r"\t|\|", engine="python", names=["region", "id", "ct"], skiprows=1)
    peaks_bt = pybedtools.BedTool.from_dataframe(peaks_bed(peaks)).sort()
    out = {}
    for label, select in CCRE_SETS.items():
        regions = peaks_bed(cc.loc[select(cc["ct"]), "region"])
        hits = peaks_bt.intersect(pybedtools.BedTool.from_dataframe(regions[["chrom", "start", "end"]]).sort(), u=True)
        names = {iv.name for iv in hits}
        out[label] = np.asarray(pd.Index(peaks).isin(names))
    return out


def peak_gene_links(peaks):
    ann = pd.read_csv(DATA / "10x_mouse_brain" /
                      "M_Brain_Chromium_Nuc_Isolation_vs_SaltyEZ_vs_ComplexTissueDP_atac_peak_annotation.tsv",
                      sep="\t").dropna(subset=["gene"])
    ann["peak"] = ann["chrom"].astype(str) + ":" + ann["start"].astype(str) + "-" + ann["end"].astype(str)
    ann = ann[ann["peak"].isin(set(peaks)) & ann["peak_type"].isin(["promoter", "distal"])]
    return ann[["peak", "gene", "peak_type"]].drop_duplicates()


def link_concordance(peak_load, gene_load, links, n_perm=N_PERM, seed=0):
    """Spearman of peak vs linked-gene loadings; null shuffles gene loadings across genes."""
    rows = []
    rng = np.random.default_rng(seed)
    for ptype in ("promoter", "distal"):
        l = links[links["peak_type"].eq(ptype) & links["gene"].isin(gene_load.index)]
        pv = peak_load.reindex(l["peak"]).to_numpy()
        ok = np.isfinite(pv)
        l, pv = l[ok], pv[ok]
        genes = l["gene"].to_numpy()
        rho = spearmanr(pv, gene_load.reindex(genes).to_numpy()).correlation
        uniq = gene_load.index
        null = []
        for _ in range(200):
            shuffled = pd.Series(rng.permutation(gene_load.to_numpy()), index=uniq)
            null.append(spearmanr(pv, shuffled.reindex(genes).to_numpy()).correlation)
        null = np.array(null)
        rows.append(dict(peak_type=ptype, n_links=len(l), rho=rho,
                         null_mean=null.mean(), null_sd=null.std(),
                         p=(np.sum(null >= rho) + 1) / (len(null) + 1)))
    return rows


# ------------------------------------------------------------------------------ main
def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.validate:
        validate_solver()
    prog = nmf_program()
    nmf_top_genes = set(prog.nlargest(TOP_K).index)

    meta_rows, gene_rows, gsea_rows, auroc_rows, link_rows, atac_nmf_rows = [], [], [], [], [], []
    for target in TARGETS:
        T = load_target(target)
        genes, peaks, rna, atac = T["genes"], T["peaks"], T["rna"], T["atac"]
        rna_stats, atac_stats = column_stats(rna), column_stats(atac)
        ident = module_scores(rna, genes)
        G_ident = ident.to_numpy()
        G_clust = pd.get_dummies(T["clusters"]).to_numpy(dtype=float)[:, 1:]
        msn = ident["MSN"].to_numpy()
        universes = {"all NMF genes": prog.index, "Fig. 5j genes": T["fig_genes"]}
        # identity baselines (run-independent)
        msn_gene = plain_corr(rna, rna_stats, msn)
        msn_peak = plain_corr(atac, atac_stats, msn)
        for uname, universe in universes.items():
            gene_rows.append(dict(target=target, run="(baseline)", axis="MSN identity score", universe=uname,
                                  loading="corr", **nmf_test(msn_gene, genes, universe, prog)))
        ccre = ccre_membership(peaks)
        fig_mask = np.asarray(peaks.isin(T["fig_peaks"]))
        links = peak_gene_links(peaks)
        nmf_linked = np.asarray(peaks.isin(links.loc[links["gene"].str.upper().isin(nmf_top_genes), "peak"]))
        any_linked = np.asarray(peaks.isin(links["peak"]))
        for label, member in ccre.items():
            a, p = auroc_stats(msn_peak, member)
            auroc_rows.append(dict(target=target, run="(baseline)", axis="MSN identity score", ccre=label,
                                   auroc_all_peaks=a, p=p))
        a, p = auroc_stats(msn_peak[any_linked], nmf_linked[any_linked])
        atac_nmf_rows.append(dict(target=target, run="(baseline)", axis="MSN identity score",
                                  auroc=a, p=p, n_nmf_linked=int(nmf_linked.sum())))

        for run in args.runs:
            tag = RUNS[run]
            R = load_run(tag, target, T["cells"])
            pls_rna = pls_crosscov(R["Z"], rna, K_FEATURE, y_stats=rna_stats)
            pls_atac = pls_crosscov(R["Z"], atac, K_FEATURE, y_stats=atac_stats)
            xs_t = pls_rna["xs"]
            axes = [dict(name="dopamine axis", beta=R["beta_da"], q=None)]
            if R["decoder"] is not None:
                axes += decoder_components(R["decoder"], R["features"])
            # decoder metabolite side (source-fitted loadings): which component carries dopamine
            if R["decoder"] is not None and target == "multiome":
                comps = axes[1:]
                X0d = (R["Z"] - R["decoder"]._x_mean) / R["decoder"]._x_std
                share = np.array([(c["q"][DA] ** 2) * np.var(X0d @ (c["beta"] * R["decoder"]._x_std)) for c in comps])
                share = share / share.sum()
                for c, sh in zip(comps, share):
                    q = c["q"]
                    ranks = q.rank(ascending=False)
                    cat = [m for m in CATECHOLAMINES if m in q.index]
                    others = [m for m in cat if m != DA]
                    obs_rank = ranks[others].mean()
                    rng = np.random.default_rng(0)
                    null = np.array([ranks.drop(DA).sample(len(others), random_state=int(rng.integers(1e9))).mean()
                                     for _ in range(5000)])
                    meta_rows.append(dict(run=run, component=c["name"], dopamine_share=sh,
                                          **{f"rank {m.replace('msi:', '')}": int(ranks[m]) for m in cat},
                                          n_metabolites=len(q),
                                          catecholamine_perm_p=(np.sum(null <= obs_rank) + 1) / (len(null) + 1),
                                          top_metabolites=", ".join(m.replace("msi:", "") for m in q.nlargest(6).index)))
            for ax in axes:
                b = ax["beta"] * xs_t                 # filter in the target's standardised latent
                s = R["Z"] @ ax["beta"]
                g_dec, g_cor = axis_loadings(pls_rna, b, rna, rna_stats)
                p_dec, p_cor = axis_loadings(pls_atac, b, atac, atac_stats)
                agree_g = spearmanr(g_dec, g_cor, nan_policy="omit").correlation
                agree_p = spearmanr(p_dec, p_cor, nan_policy="omit").correlation
                g_part = {"identity-adjusted": partial_corr(rna, rna_stats, s, G_ident),
                          "within-cluster": partial_corr(rna, rna_stats, s, G_clust)}
                r_msn = np.corrcoef(s, msn)[0, 1]
                for uname, universe in universes.items():
                    for lname, vals in (("decoded", g_dec), ("corr", g_cor), *g_part.items()):
                        gene_rows.append(dict(target=target, run=run, axis=ax["name"], universe=uname, loading=lname,
                                              score_r_with_MSN=r_msn, decoded_vs_corr_rho=agree_g,
                                              **nmf_test(vals, genes, universe, prog)))
                if target == "multiome" and ax["name"] == "dopamine axis":
                    pd.DataFrame({"gene": genes, "decoded": g_dec, "corr": g_cor, **g_part}).to_csv(
                        args.out_dir / f"gene_loadings_{run.replace(' ', '_')}.csv", index=False)
                    pd.DataFrame({"peak": peaks, "decoded": p_dec, "corr": p_cor}).to_csv(
                        args.out_dir / f"peak_loadings_{run.replace(' ', '_')}.csv", index=False)
                # ATAC: cCRE enrichment (Fig. 6b), leading edge for motifs (Fig. 6c)
                for label, member in ccre.items():
                    res, leading = weighted_gsea(p_dec[fig_mask], member[fig_mask])
                    gsea_rows.append(dict(target=target, run=run, axis=ax["name"], ccre=label,
                                          peaks="Fig. 6 2,000", **res))
                    a, p = auroc_stats(p_dec, member)
                    auroc_rows.append(dict(target=target, run=run, axis=ax["name"], ccre=label,
                                           auroc_all_peaks=a, p=p, peaks_decoded_vs_corr_rho=agree_p))
                    if target == "multiome" and ax["name"] == "dopamine axis":
                        d = args.out_dir / "motif_inputs" / run.replace(" ", "_")
                        d.mkdir(parents=True, exist_ok=True)
                        fig_peaks = peaks[fig_mask]
                        fg = fig_peaks[leading]
                        bg = fig_peaks[member[fig_mask]].difference(fg)
                        peaks_bed(fg).to_csv(d / f"{label.lower()}_foreground.bed", sep="\t", header=False, index=False)
                        peaks_bed(bg).to_csv(d / f"{label.lower()}_background.bed", sep="\t", header=False, index=False)
                # RNA vs ATAC
                gene_load = pd.Series(g_dec, index=genes).dropna()
                gene_load = gene_load[~gene_load.index.duplicated()]
                peak_load = pd.Series(p_dec, index=peaks)
                for row in link_concordance(peak_load, gene_load, links):
                    link_rows.append(dict(target=target, run=run, axis=ax["name"], **row))
                a, p = auroc_stats(p_dec[any_linked], nmf_linked[any_linked])
                atac_nmf_rows.append(dict(target=target, run=run, axis=ax["name"], auroc=a, p=p,
                                          n_nmf_linked=int(nmf_linked.sum())))
                print(f"[axis] {target} {run} {ax['name']}: r(score, MSN)={r_msn:+.2f}; decoded~corr "
                      f"genes {agree_g:.3f} peaks {agree_p:.3f}", flush=True)
            meta_rows.append(dict(run=run, component=f"({target}) dopamine-axis fit residual",
                                  dopamine_share=R["beta_resid"],
                                  top_metabolites="decoder reproduces imputation" if R["decoder"] is not None
                                  else "decoder file != projection model; axis from stored imputation"))
        del T, rna, atac

    tables = dict(metabolites=meta_rows, gene_convergence=gene_rows, atac_ccre_gsea=gsea_rows,
                  atac_ccre_auroc=auroc_rows, rna_atac_concordance=link_rows, atac_nmf_linked=atac_nmf_rows)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(args.out_dir / f"{name}.csv", index=False)
    print(f"[done] wrote {', '.join(tables)} to {args.out_dir}")


if __name__ == "__main__":
    main()

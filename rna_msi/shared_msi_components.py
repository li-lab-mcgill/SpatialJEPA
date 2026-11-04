try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Shared PLS components across the source (RNA + MSI) and ONE target (RNA + ATAC).
#
# MOFA-free replacement for the Fig. 5 / Fig. 6 factor analyses. One latent, one SVD:
#
#   latent    the student's RNA-expert latent (H["st"]["q_mu"], obsm "student_X_emb_st"),
#             computed from RNA alone for every source spot and target cell, z-scored per
#             dimension within each dataset
#   blocks    source RNA (SCT data, all genes), source MSI (measured; the decoder's 779
#             metabolites, log1p as its targets), target RNA (SCT data), target ATAC
#   C_b       latent x feature correlation matrix of block b in its own dataset
#   SVD       [C_1/|C_1| | ... | C_4/|C_4|] = U S V^T  (equal block weight; multi-block
#             PLS-SVD). Column u_k of U is component k, shared by all four blocks
#   score     s_k = Z0 u_k in each dataset (same encoder, same coordinates)
#   weights   corr(feature, s_k) in the feature's own dataset = C_b^T u_k / sd(s_k), so
#             gene, metabolite and peak weights sit on one -1..1 scale per component
#   sign      each component oriented so measured source dopamine weighs positively
#   anchor    `--anchor msi`: U is the SVD of the source MSI block alone, C_srcMSI/|C_srcMSI|
#             (the latent directions carrying most latent-metabolite covariance). No dopamine
#             label and no target data enter U, so it is identical in every single-target
#             analysis; the dopamine component is the one with the largest |dopamine weight|
#             (C3 in the mouse run). `--anchor dopamine` (the earlier default): C1 fixed to
#             the OLS dopamine direction (u ~ Sigma_src^-1 corr(Z, dopamine)), C2..CK the SVD
#             of the stacked matrix after removing C1. `--anchor none`: plain stacked SVD.
#   source    `--source-spots intact` (default): source spots of the intact hemisphere only. The
#             6-OHDA lesion drops striatal dopamine to non-striatal levels while MSN identity is
#             unchanged, a dopamine-identity decoupling none of the (unlesioned) targets share;
#             in intact tissue dopamine follows striatal identity (r 0.64 with an MSN score) and is
#             better predicted by the latent (R^2 0.44 vs 0.25 on all spots). `all`: every spot
#   ported    metabolite weights in a dataset without MSI ("target MSI (ported)"): a reduced-rank
#             PLS regression of all metabolites on the source latent (rank by grouped CV: 5
#             spatial blocks of the source section; sections for the human source), applied to
#             the target, each prediction correlated with the component score there, times its
#             held-out source accuracy R_m (clipped at 0). If the prediction error is uncorrelated
#             with the score, R_m x corr(prediction, score) = corr(metabolite, score): the weight
#             expected if the metabolite were measured, given the source accuracy. latent_overlap
#             reports how far target cells sit from the source latent distribution.
#   OT        metabolite weights recovered in the target ("target MSI (OT)", panel 5i): every target
#             cell is OT-paired with the source spots in the latent (entropic, reg OT_REG = 0.01, the
#             value with the best held-out-animal fidelity in the animal-transfer benchmark) and given
#             their measured MSI; a reduced-rank PLS decoder is fitted ON THE TARGET (target latent ->
#             paired MSI; CV folds: 5 spatial blocks on P22, random on nuclei) and the weights are
#             R_m x corr_target(decoded metabolite, score), as for the ported weights (ot_msi_weights)
#   ATAC      `--atac-covariates depth_gc` (default): peak weights are corr(peak, score) with the score
#             residualised, in that dataset, on log ATAC depth and per-cell GC bias (count-weighted mean
#             GC of the cell's peak signal). The multiome C2 score correlates -0.12 with log depth and
#             +0.14 with GC bias; unadjusted, 78% of peak weights are negative and the weights track peak
#             GC; adjusted, r(weight, peak GC) = 0.00 and the D2-MSN cCRE AUROC rises 0.71 -> 0.79 (0.80
#             within GC bins). Unadjusted weights are kept as "target ATAC (unadjusted)". `none`: raw score
#   overview  per block, the share of its latent-feature correlation energy captured by each
#             component, |C_b^T u_k|^2 / |C_b|_F^2 (sums to 1 over a full basis); the mean
#             R^2 of features on the component scores is in components_summary.csv
#
# Comparisons: RNA weights exist in both datasets (source-vs-target Spearman per
# component); metabolite (source) and peak (target) weights belong to the same component.
# In the P22 spatial target the teacher's RNA-expert latent (run with P22's spatial graph)
# is scored on the same components (teacher vs student). Targets are analysed one at a time;
# `--target compare` matches the components of the two single-target analyses.
#
# Env: $PY_ANALYSIS

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
OT_REG = 0.01
sys.path.insert(0, str(HERE))
import msi_components_common as P  # noqa: E402
import dopamine_program_layouts as F  # noqa: E402

OUT_DEFAULT = P.OUT / "rna_msi_runs" / "shared_components"
SOURCE_H5MU = P.DATA / "vicari_2023" / "h5mu_export" / f"{P.BASE}_SCT.h5mu"
TARGET_LABEL = {"spatial": "P22 spatial RNA-ATAC", "multiome": "multiome nuclei"}
SHORT_LABEL = {"spatial": "P22 spatial", "multiome": "multiome"}
BLOCKS = ("source RNA", "source MSI", "target RNA", "target ATAC")
GENE_MARKERS = ["Ppp1r1b", "Penk", "Tac1", "Drd1", "Drd2", "Adora2a", "Gpr88", "Rgs9", "Pde10a",
                "Mobp", "Plp1", "Mbp", "Slc17a7", "Gja1"]


def parse_args():
    p = argparse.ArgumentParser(description="Shared PLS components, source + one target (Figs. 5-6).")
    p.add_argument("--target", required=True, choices=["spatial", "multiome", "compare"])
    p.add_argument("--run", default="fixed s3", choices=list(P.RUNS))
    p.add_argument("--k", type=int, default=6, help="components reported")
    p.add_argument("--anchor", default="msi", choices=["msi", "dopamine", "none"],
                   help="SVD of the source MSI block (default), C1 fixed to the source dopamine "
                        "direction, or plain stacked SVD")
    p.add_argument("--source-spots", default="intact", choices=["intact", "all"],
                   help="fit on the intact-hemisphere source spots (default) or on all of them")
    p.add_argument("--atac-covariates", default="depth_gc", choices=["depth_gc", "none"],
                   help="residualise the score on log ATAC depth + GC bias for the peak weights (default)")
    p.add_argument("--out-dir", type=Path, default=OUT_DEFAULT)
    p.add_argument("--redraw", action="store_true", help="redraw from the cached results.pkl")
    p.add_argument("--tables-only", action="store_true",
                   help="components, weights and cCRE GSEA only (no motif test, no figures)")
    return p.parse_args()


# ------------------------------------------------------------------------- algebra
def standardize(Z):
    Z = np.asarray(Z, dtype=float)
    return (Z - Z.mean(0)) / Z.std(0, ddof=1)


def corr_matrix(Z0, X, stats):
    """D x p correlations between standardised latent dims and the columns of X."""
    n = Z0.shape[0]
    sd = stats[1]
    XtZ = np.asarray(X.T @ Z0) if sp.issparse(X) else X.T @ Z0       # Z0 centred: X needs no centring
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (XtZ / ((n - 1) * sd[:, None])).T
    C[:, ~(sd > 0)] = 0.0
    return C


def direction_weights(C, Z0, u):
    """corr(feature, Z0 u) for every feature of a block."""
    sigma = Z0.T @ Z0 / (Z0.shape[0] - 1)
    return C.T @ u / np.sqrt(u @ sigma @ u)


def block_r2(C, Z0, U):
    """Mean over features of R^2(feature ~ all K component scores) and per component."""
    sigma = Z0.T @ Z0 / (Z0.shape[0] - 1)
    sd = np.sqrt(np.einsum("ik,ij,jk->k", U, sigma, U))
    W = C.T @ U / sd                                   # p x K correlations
    Rs = (U.T @ sigma @ U) / np.outer(sd, sd)          # score correlations
    joint = np.einsum("pk,kl,pl->p", W, np.linalg.pinv(Rs), W)
    return (W ** 2).mean(0), float(joint.mean())


def _standardize_cols(X):
    X = np.asarray(X, dtype=float)
    X = (X - X.mean(0)) / X.std(0, ddof=1)
    X[:, ~np.isfinite(X).all(0)] = 0.0
    return X


def _colcorr(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = (A * B).sum(0) / np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0))
    return np.nan_to_num(R)


def fit_msi_predictor(Z_src, msi, groups, method="pls"):
    """Latent -> metabolite predictor for porting (see header, `ported`). 'pls': reduced-rank
    PLS regression of all (standardised log1p) metabolites at once on the standardised latent,
    rank K chosen by grouped CV with the one-standard-error rule (smallest K whose mean held-out
    fold R^2 is within one fold-SE of the best; K = D is OLS); 'ols': per-metabolite
    least squares; 'simpls': the 'pls' recipe with a numpy SIMPLS fit (same predictions as
    sklearn's PLSRegression to r > 0.9998, much faster for many cells). Returns dict(B=D x p
    coefficients, R=held-out R per metabolite (clipped at 0), resid_var=held-out residual variance,
    k=rank, cv=table of the CV curve)."""
    from sklearn.cross_decomposition import PLSRegression

    X = _standardize_cols(msi)
    D = Z_src.shape[1]

    def fit(Ztr, Xtr, k):
        if method == "ols":
            coef = np.linalg.lstsq(np.column_stack([np.ones(len(Ztr)), Ztr]), Xtr, rcond=None)[0]
            return coef[0], coef[1:]
        if method == "simpls":
            return _simpls(Ztr, Xtr, k)
        m = PLSRegression(n_components=k, scale=False).fit(Ztr, Xtr)
        b0 = m.predict(np.zeros((1, D)))[0]
        return b0, m.predict(np.eye(D)) - b0

    rows, held = [], {}
    for k in (range(1, D + 1) if method in ("pls", "simpls") else [D]):
        pred, fold_r2 = np.zeros_like(X), []
        for g in np.unique(groups):
            h = groups == g
            b0, B = fit(Z_src[~h], X[~h], k)
            pred[h] = b0 + Z_src[h] @ B
            fold_r2.append(1 - ((X[h] - pred[h]) ** 2).sum() / ((X[h] - X[h].mean(0)) ** 2).sum())
        R = _colcorr(pred, X)
        held[k] = (R, ((X - pred) ** 2).mean(0))
        rows.append(dict(k=k, pooled_R2=1 - ((X - pred) ** 2).sum() / ((X - X.mean(0)) ** 2).sum(),
                         fold_R2_mean=float(np.mean(fold_r2)),
                         fold_R2_se=float(np.std(fold_r2, ddof=1) / np.sqrt(len(fold_r2))),
                         median_R=float(np.median(R)), n_R_above_0_5=int((R > 0.5).sum())))
    cv = pd.DataFrame(rows)
    best = cv.loc[cv["fold_R2_mean"].idxmax()]
    k = int(cv.loc[cv["fold_R2_mean"] >= best["fold_R2_mean"] - best["fold_R2_se"], "k"].min())
    _, B = fit(Z_src, X, k)
    R, resid = held[k]
    return dict(B=B, R=np.clip(R, 0.0, None), resid_var=resid, k=k, cv=cv, method=method)


def _simpls(X, Y, k):
    """Reduced-rank PLS regression (de Jong's SIMPLS) of Y on X; returns intercept, D x q B."""
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    S = Xc.T @ Yc
    R, Q, V = [], [], []
    for _ in range(k):
        r = np.linalg.svd(S, full_matrices=False)[0][:, 0]
        t = Xc @ r
        nt = np.linalg.norm(t)
        t, r = t / nt, r / nt
        p, q = Xc.T @ t, Yc.T @ t
        v = p - sum(vv * (vv @ p) for vv in V)
        v /= np.linalg.norm(v)
        S = S - np.outer(v, v @ S)
        R.append(r); Q.append(q); V.append(v)
    B = np.column_stack(R) @ np.column_stack(Q).T
    return ym - xm @ B, B


def ot_barycentre(Z_src, Y_src, Z_tgt, reg=0.01, device=None):
    """Source features carried to target cells by entropic OT between the two (standardised)
    latents: squared-Euclidean cost scaled by its mean, uniform marginals, log-domain Sinkhorn;
    each target cell gets the coupling-weighted mean of the source rows of Y_src."""
    import ot
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Zs, Zt = (torch.tensor(np.asarray(Z, dtype=np.float32), device=dev) for Z in (Z_src, Z_tgt))
    M = ot.dist(Zt, Zs)
    M /= M.mean()
    a = torch.full((len(Zt),), 1 / len(Zt), device=dev)
    b = torch.full((len(Zs),), 1 / len(Zs), device=dev)
    Pm = ot.sinkhorn(a, b, M, reg, method="sinkhorn_log", numItermax=3000, stopThr=1e-7)
    del M
    Y = torch.tensor(np.asarray(Y_src, dtype=np.float32), device=dev)
    out = ((Pm @ Y) / Pm.sum(1, keepdim=True)).cpu().numpy().astype(float)
    del Pm, Y
    torch.cuda.empty_cache()
    return out


def ot_barycentre_batched(Z_src, Y_src, Z_tgt, batches, reg=0.01, device=None):
    """ot_barycentre run once per source batch (section), with the latent and the features
    standardised within that batch, and the target values averaged over batches with equal
    weight. Section offsets in the features (and section shifts in the latent) then cannot
    reach the target as differences between the cells that happen to pair with one section."""
    out = []
    for b in np.unique(batches):
        m = np.asarray(batches) == b
        out.append(ot_barycentre(standardize(np.asarray(Z_src)[m]), _standardize_cols(np.asarray(Y_src)[m]),
                                 Z_tgt, reg=reg, device=device))
    return np.mean(out, axis=0)


def ot_msi_weights(Z_src, msi, Z_tgt, U, groups, reg=0.01, device=None, batches=None, return_paired=False):
    """Metabolite weights recovered in a dataset without MSI: OT-pair target cells with source
    spots (ot_barycentre of the standardised log1p MSI; with `batches`, ot_barycentre_batched on
    the unstandardised source latent), fit the reduced-rank PLS decoder ON THE TARGET (target
    latent -> paired MSI, rank by grouped CV, one-SE rule), weights as ported_msi_weights
    (R_m x corr_target(decoded metabolite, component score)). Returns (W, decoder), plus the
    paired target MSI with `return_paired`."""
    if batches is None:
        Y = ot_barycentre(Z_src, _standardize_cols(msi), Z_tgt, reg=reg, device=device)
    else:
        Y = ot_barycentre_batched(Z_src, msi, Z_tgt, batches, reg=reg, device=device)
    dec = fit_msi_predictor(Z_tgt, Y, groups, method="simpls")
    W = ported_msi_weights(dec, Z_tgt, U)[0]
    return (W, dec, Y) if return_paired else (W, dec)


def ported_msi_weights(pred, Z_tgt, U):
    """Metabolite weights carried into a dataset without MSI (see header, `ported`):
    R_m x corr_target(predicted metabolite, component score), from a fit_msi_predictor result.
    Also returns the target-adjusted reliability R_t (diagnostic only): var_t(m_hat) /
    (var_t(m_hat) + held-out residual variance), square-rooted."""
    B, R = pred["B"], pred["R"]
    sigma = Z_tgt.T @ Z_tgt / (len(Z_tgt) - 1)
    var_m = np.einsum("dp,de,ep->p", B, sigma, B)
    sd_s = np.sqrt(np.einsum("dk,de,ek->k", U, sigma, U))
    with np.errstate(invalid="ignore", divide="ignore"):
        W = (B.T @ sigma @ U) / np.outer(np.sqrt(var_m), sd_s)
        R_t = np.sqrt(var_m / (var_m + pred["resid_var"]))
    return np.nan_to_num(W) * R[:, None], np.nan_to_num(R_t)


def latent_overlap(Z_src, Z_tgt, n_max=20000, seed=0):
    """How far target cells sit from the source latent distribution (both standardised): CV AUC
    of a gradient-boosting source-vs-target classifier, the share of target cells beyond the
    source's 99th-percentile Mahalanobis distance, and |Sigma_src - Sigma_tgt|_F / |Sigma_src|_F."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    rng = np.random.default_rng(seed)
    pick = lambda Z: Z[rng.choice(len(Z), min(n_max, len(Z)), replace=False)]
    A, T = pick(Z_src), pick(Z_tgt)
    X, y = np.vstack([A, T]), np.r_[np.zeros(len(A)), np.ones(len(T))]
    prob = cross_val_predict(HistGradientBoostingClassifier(random_state=seed), X, y, cv=5,
                             method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score

    Ss, St = np.cov(Z_src, rowvar=False), np.cov(Z_tgt, rowvar=False)
    inv = np.linalg.inv(Ss)
    d_src = np.einsum("nd,de,ne->n", Z_src, inv, Z_src)
    d_tgt = np.einsum("nd,de,ne->n", Z_tgt, inv, Z_tgt)
    return dict(domain_auc=float(roc_auc_score(y, prob)),
                frac_beyond_src_q99=float((d_tgt > np.quantile(d_src, 0.99)).mean()),
                cov_rel_diff=float(np.linalg.norm(Ss - St) / np.linalg.norm(Ss)))


def atac_covariates(atac, peaks, gc_cache):
    """Per-cell technical covariates of an ATAC matrix: log depth and GC bias (count-weighted mean
    GC fraction of the cell's peak signal, z-scored). Peak GC from plot_dopamine_program.peak_gc."""
    from plot_dopamine_program import peak_gc

    X = atac.tocsr()
    depth = np.maximum(np.asarray(X.sum(1)).ravel(), 1e-12)
    gc = peak_gc(pd.Index(peaks), Path(gc_cache)).reindex(pd.Index(peaks)).to_numpy(float)
    bias = np.asarray(X @ np.nan_to_num(gc, nan=np.nanmean(gc))).ravel() / depth
    return np.column_stack([np.log(depth), (bias - bias.mean()) / bias.std()])


def covariate_adjusted_weights(X, x_stats, scores, covs, names, comps):
    """corr(feature, score residualised on covs) for every column of scores (header, `ATAC`)."""
    A = np.column_stack([np.ones(len(scores)), covs])
    Sr = scores - A @ np.linalg.lstsq(A, scores, rcond=None)[0]
    return pd.DataFrame(np.column_stack([np.nan_to_num(P.plain_corr(X, x_stats, Sr[:, k]))
                                         for k in range(Sr.shape[1])]), index=names, columns=comps)


def spatial_blocks(xy, n=5, seed=0):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n, n_init=10, random_state=seed).fit_predict(np.asarray(xy, dtype=float))


# ----------------------------------------------------------------------------- data
def load_source(tag):
    import anndata as ad
    import joblib

    joint = ad.read_h5ad(P.MODELS / (P.BASE + tag) / "joint_adata.h5ad")
    feats = list(joblib.load(P.MODELS / (P.BASE + tag) / "student_pls_decoder.joblib")["target_features"])
    X = joint[:, feats].X
    msi = np.log1p(X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float))
    with h5py.File(SOURCE_H5MU, "r") as f:
        cells = pd.Index(P.read_index(f["mod/rna/obs"]))
        genes = pd.Index(P.read_index(f["mod/rna/var"]))
        rna = P.read_csr(f["mod/rna/layers/SCT_data"])
    order = cells.get_indexer(joint.obs_names)
    if (order < 0).any():
        raise ValueError("source RNA spots do not cover the model's spots")
    rna = rna[order].tocsc()
    obs = joint.obs[["lesion", "region"]].astype(str)
    print(f"[data] source: {joint.n_obs:,} spots, {len(genes):,} genes, {len(feats)} metabolites", flush=True)
    return dict(cells=joint.obs_names, Z=np.asarray(joint.obsm["student_X_emb_st"]),
                Z_teacher=np.asarray(joint.obsm["teacher_X_emb_st"]), xy=np.asarray(joint.obsm["spatial"])[:, :2],
                rna=rna, genes=genes, msi=msi, metabolites=pd.Index(feats), obs=obs)


def subset_source(S, keep):
    out = dict(S, cells=S["cells"][keep], Z=S["Z"][keep], Z_teacher=S["Z_teacher"][keep], xy=S["xy"][keep],
               rna=S["rna"].tocsr()[keep].tocsc(), msi=S["msi"][keep], obs=S["obs"].loc[keep])
    print(f"[data] source restricted to {int(keep.sum()):,}/{len(keep):,} spots", flush=True)
    return out


def target_latents(tag, target, cells):
    import anndata as ad

    emb = ad.read_h5ad(P.PROJ / (P.BASE + tag) / P.TARGETS[target]["emb"], backed="r")
    out = {}
    for key in ("student_X_emb_st", "teacher_X_emb_st"):
        if key in emb.obsm:
            Z = pd.DataFrame(np.asarray(emb.obsm[key]), index=emb.obs_names).reindex(cells).to_numpy()
            if np.isnan(Z).any():
                raise ValueError(f"{target}: {key} missing for some cells")
            out[key.split("_")[0]] = Z
    return out


def upper_series(values, genes):
    s = pd.Series(np.asarray(values), index=pd.Index(genes).str.upper())
    return s[~s.index.duplicated()].dropna()


# ------------------------------------------------------------------------- analysis
def analyse(args, ns):
    tag = P.RUNS[args.run]
    S = load_source(tag)
    if args.source_spots == "intact":
        S = subset_source(S, S["obs"]["lesion"].eq("intact").to_numpy())
    T = P.load_target(args.target)
    lat = target_latents(tag, args.target, T["cells"])
    Z0 = {"source": standardize(S["Z"]), "target": standardize(lat["student"])}
    blocks = {
        "source RNA": ("source", S["rna"], S["genes"]),
        "source MSI": ("source", S["msi"], S["metabolites"]),
        "target RNA": ("target", T["rna"], T["genes"]),
        "target ATAC": ("target", T["atac"], T["peaks"]),
    }
    stats = {b: P.column_stats(X) for b, (_, X, _) in blocks.items()}
    C = {b: corr_matrix(Z0[d], X, stats[b]) for b, (d, X, _) in blocks.items()}
    M = np.hstack([C[b] / np.linalg.norm(C[b]) for b in BLOCKS])
    K = args.k
    da = S["metabolites"].get_loc(P.DA)
    if args.anchor == "dopamine":
        sigma_src = Z0["source"].T @ Z0["source"] / (Z0["source"].shape[0] - 1)
        u0 = np.linalg.solve(sigma_src, C["source MSI"][:, da])
        u0 /= np.linalg.norm(u0)
        rest, _, _ = np.linalg.svd(M - np.outer(u0, u0 @ M), full_matrices=False)
        U = np.column_stack([u0, rest[:, :K - 1]])
    elif args.anchor == "msi":
        Cm = C["source MSI"]
        U = np.linalg.svd(Cm / np.linalg.norm(Cm), full_matrices=False)[0][:, :K].copy()
    else:
        U = np.linalg.svd(M, full_matrices=False)[0][:, :K].copy()
    sv = np.linalg.norm(U.T @ M, axis=1)                    # |u_k^T M|; shares below
    share = {b: np.linalg.norm(U.T @ C[b], axis=1) ** 2 / np.linalg.norm(C[b]) ** 2 for b in BLOCKS}

    # orient, weights, variance
    W = {}
    for k in range(K):
        w_da = direction_weights(C["source MSI"], Z0["source"], U[:, k])[da]
        U[:, k] *= np.sign(w_da) or 1.0
    for b, (d, _, names) in blocks.items():
        W[b] = pd.DataFrame(np.column_stack([direction_weights(C[b], Z0[d], U[:, k]) for k in range(K)]),
                            index=names, columns=[f"C{k + 1}" for k in range(K)])
    comps = list(W["source MSI"].columns)
    atac_covs = None
    if args.atac_covariates == "depth_gc":
        atac_covs = atac_covariates(T["atac"], T["peaks"], out_dir(args) / "peak_gc.csv")
        W["target ATAC (unadjusted)"] = W["target ATAC"]
        W["target ATAC"] = covariate_adjusted_weights(T["atac"], stats["target ATAC"], Z0["target"] @ U,
                                                      atac_covs, T["peaks"], comps)
        print("[ATAC] peak weights against the score residualised on log depth + GC bias; "
              f"r(score, log depth) / r(score, GC bias): " + ", ".join(
                  f"{c} {np.corrcoef(Z0['target'] @ U[:, k], atac_covs[:, 0])[0, 1]:+.2f} / "
                  f"{np.corrcoef(Z0['target'] @ U[:, k], atac_covs[:, 1])[0, 1]:+.2f}" for k, c in enumerate(comps)),
              flush=True)
    predictor = fit_msi_predictor(Z0["source"], S["msi"], spatial_blocks(S["xy"]))
    msi_R = predictor["R"]
    ported, msi_R_t = ported_msi_weights(predictor, Z0["target"], U)
    W["target MSI (ported)"] = pd.DataFrame(ported, index=S["metabolites"], columns=comps)
    overlap = latent_overlap(Z0["source"], Z0["target"])
    print(f"[ported] PLS predictor rank {predictor['k']} (grouped CV); dopamine R_cv "
          f"{msi_R[da]:.2f}, target-adjusted {msi_R_t[da]:.2f}; median R_cv {np.median(msi_R):.2f}; overlap "
          + ", ".join(f"{k} {v:.2f}" for k, v in overlap.items()), flush=True)
    groups_t = (spatial_blocks(F.coordinates(tag, args.target, T["cells"])) if args.target == "spatial"
                else np.random.default_rng(0).integers(0, 5, len(T["cells"])))
    W_ot, dec_ot = ot_msi_weights(Z0["source"], S["msi"], Z0["target"], U, groups_t, reg=OT_REG)
    W["target MSI (OT)"] = pd.DataFrame(W_ot, index=S["metabolites"], columns=comps)
    print(f"[OT] target decoder rank {dec_ot['k']}; recovered vs source-measured weights r "
          + ", ".join(f"{c} {np.corrcoef(W['target MSI (OT)'][c], W['source MSI'][c])[0, 1]:.3f}" for c in comps),
          flush=True)
    r2 = {b: block_r2(C[b], Z0[blocks[b][0]], U) for b in BLOCKS}
    k_da = int(np.argmax(np.abs(W["source MSI"].loc[P.DA].to_numpy())))
    c_da = comps[k_da]
    scores = {d: Z0[d] @ U for d in Z0}

    # teacher (P22 only): same components, teacher RNA-expert latent on P22's spatial graph
    teacher = None
    if "teacher" in lat:
        Zt = standardize(lat["teacher"])
        teacher = dict(score=Zt @ U)
        for b in ("target RNA", "target ATAC"):
            Ct = corr_matrix(Zt, blocks[b][1], stats[b])
            teacher[b] = pd.DataFrame(np.column_stack([direction_weights(Ct, Zt, U[:, k]) for k in range(K)]),
                                      index=blocks[b][2], columns=comps)
        if atac_covs is not None:
            teacher["target ATAC (unadjusted)"] = teacher["target ATAC"]
            teacher["target ATAC"] = covariate_adjusted_weights(T["atac"], stats["target ATAC"], Zt @ U,
                                                                atac_covs, T["peaks"], comps)
        teacher["target MSI (ported)"] = pd.DataFrame(
            ported_msi_weights(predictor, Zt, U)[0], index=S["metabolites"], columns=comps)
        teacher["target MSI (OT)"] = pd.DataFrame(
            ot_msi_weights(Z0["source"], S["msi"], Zt, U, groups_t, reg=OT_REG)[0],
            index=S["metabolites"], columns=comps)

    # summaries per component
    prog = P.nmf_program()
    rows = []
    for k, c in enumerate(comps):
        src_g, tgt_g = upper_series(W["source RNA"][c], S["genes"]), upper_series(W["target RNA"][c], T["genes"])
        shared = src_g.index.intersection(tgt_g.index)
        row = dict(component=c, sv_share=sv[k] ** 2 / np.linalg.norm(M) ** 2,
                   **{f"share {b}": share[b][k] for b in BLOCKS},
                   **{f"R2 {b}": r2[b][0][k] for b in BLOCKS},
                   dopamine_weight=W["source MSI"].loc[P.DA, c],
                   dopamine_weight_ported=W["target MSI (ported)"].loc[P.DA, c],
                   dopamine_weight_ot=W["target MSI (OT)"].loc[P.DA, c],
                   dopamine_rank_ot=int((W["target MSI (OT)"][c] > W["target MSI (OT)"].loc[P.DA, c]).sum()) + 1,
                   r_ot_vs_source_msi=np.corrcoef(W["target MSI (OT)"][c], W["source MSI"][c])[0, 1],
                   rna_src_vs_tgt_rho=spearmanr(src_g[shared], tgt_g[shared]).correlation,
                   nmf_auroc_target=P.nmf_test(W["target RNA"][c].to_numpy(), T["genes"], prog.index, prog)["auroc"],
                   nmf_auroc_source=P.nmf_test(W["source RNA"][c].to_numpy(), S["genes"], prog.index, prog)["auroc"],
                   top_metabolites=", ".join(m.replace("msi:", "") for m in W["source MSI"][c].nlargest(5).index),
                   top_source_genes=", ".join(W["source RNA"][c].nlargest(6).index),
                   top_target_genes=", ".join(W["target RNA"][c].nlargest(6).index))
        if teacher is not None:
            for b in ("target RNA", "target ATAC"):
                row[f"teacher_vs_student {b} rho"] = spearmanr(teacher[b][c], W[b][c], nan_policy="omit").correlation
            row["teacher_vs_student score r"] = np.corrcoef(teacher["score"][:, k], scores["target"][:, k])[0, 1]
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = dict(component="all K (joint R2)", sv_share=(sv ** 2).sum() / np.linalg.norm(M) ** 2,
                                     **{f"share {b}": share[b].sum() for b in BLOCKS},
                                     **{f"R2 {b}": r2[b][1] for b in BLOCKS})
    print(summary.drop(columns=[c for c in summary if c.startswith("top_")]).round(3).to_string(index=False), flush=True)

    # Fig. 5 panels
    msn = P.module_scores(T["rna"], T["genes"])["MSN"].to_numpy()
    universe = T["fig_genes"]
    xy_t = F.coordinates(tag, args.target, T["cells"])
    panel5 = dict(
        target=args.target, run=args.run, anchor=args.anchor, comps=comps, c_da=c_da, summary=summary, sv=sv,
        source_xy=S["xy"], target_xy=xy_t, scores=scores, teacher_score=None if teacher is None else teacher["score"],
        metabolites=W["source MSI"][c_da], genes_source=W["source RNA"][c_da], genes_target=W["target RNA"][c_da],
        gene_scatter=_gene_scatter(W["source RNA"][c_da], W["target RNA"][c_da]),
        convergence=[F.convergence_stats(ns, f"{SHORT_LABEL[args.target]}\n(target)", W["target RNA"][c_da], universe),
                     F.convergence_stats(ns, "Visium\n(source)", W["source RNA"][c_da], universe),
                     F.convergence_stats(ns, "MSN identity\n(target)",
                                         pd.Series(P.plain_corr(T["rna"], stats["target RNA"], msn), index=T["genes"]),
                                         universe)],
    )

    # Fig. 6 panels: the dopamine component's peak weights
    out = out_dir(args)
    peaks = W["target ATAC"][c_da]
    ccre = P.ccre_membership(T["peaks"])
    fig_mask = np.asarray(T["peaks"].isin(T["fig_peaks"]))
    fig_peaks, fig_scores = T["peaks"][fig_mask], peaks.to_numpy()[fig_mask]
    gsea, gsea_rows, motif_inputs = {}, [], out / "motif_inputs"
    motif_inputs.mkdir(parents=True, exist_ok=True)
    for label in ("D2MSN", "MXD"):
        pre, leading, res = F.gsea_object(fig_scores, ccre[label][fig_mask], fig_peaks, f"{label}_cCRE")
        gsea[label] = pre
        a, p = P.auroc_stats(peaks.to_numpy(), ccre[label])
        gsea_rows.append(dict(ccre=label, **res, auroc_all_peaks=a, auroc_p=p))
        fg = fig_peaks[leading]
        bg = fig_peaks[ccre[label][fig_mask]].difference(fg)
        P.peaks_bed(fg).to_csv(motif_inputs / f"{label.lower()}_foreground.bed", sep="\t", header=False, index=False)
        P.peaks_bed(bg).to_csv(motif_inputs / f"{label.lower()}_background.bed", sep="\t", header=False, index=False)
    label = f"shared_{args.target}"
    motifs_dir = out / "motifs_gc"
    motifs = None
    if not args.tables_only:
        if not (motifs_dir / f"{label}_mxd_gc_controlled_motifs.tsv").exists():
            print(f"[motifs] GC-adjusted motif test ({label}, scenicplus env)", flush=True)
            subprocess.run([F.SCENICPLUS_PY, str(HERE / "gc_matched_motif_enrichment.py"), "--processes", "16",
                            "--out-dir", str(motifs_dir), "--inputs", f"{label}={motif_inputs}"],
                           check=True, stdout=subprocess.DEVNULL)
        motifs = {ct: F.motif_table(motifs_dir / f"{label}_{ct.lower()}_gc_controlled_motifs.tsv")
                  for ct in ("D2MSN", "MXD")}
    links = P.peak_gene_links(T["peaks"])
    gene_load = W["target RNA"][c_da]
    gene_load = gene_load[~gene_load.index.duplicated()]
    conc = []
    for row in P.link_concordance(peaks, gene_load, links):
        sub = links[links["peak_type"].eq(row["peak_type"]) & links["gene"].isin(gene_load.index)]
        x, y = gene_load.reindex(sub["gene"]).to_numpy(), peaks.reindex(sub["peak"]).to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)
        conc.append(dict(row, x=x[ok], y=y[ok]))
    nmf_linked = np.asarray(T["peaks"].isin(links.loc[links["gene"].str.upper().isin(set(prog.nlargest(P.TOP_K).index)), "peak"]))
    any_linked = np.asarray(T["peaks"].isin(links["peak"]))
    pv = peaks.to_numpy()
    auroc, p = P.auroc_stats(pv[any_linked], nmf_linked[any_linked])
    msn_peak = P.plain_corr(T["atac"], stats["target ATAC"], msn)
    baseline_auroc, _ = P.auroc_stats(msn_peak[any_linked], nmf_linked[any_linked])
    panel6 = dict(axis_title=f"Shared component {c_da}",
                  subtitle=f"(source RNA+MSI / target RNA+ATAC), {TARGET_LABEL[args.target]}",
                  gsea=gsea, motifs=motifs, concordance=conc,
                  nmf_peaks=dict(top=pv[any_linked & nmf_linked], rest=pv[any_linked & ~nmf_linked],
                                 auroc=auroc, p=p, baseline_auroc=baseline_auroc))

    # tables
    summary.to_csv(out / "components_summary.csv", index=False)
    pd.DataFrame(gsea_rows).to_csv(out / f"ccre_gsea_{c_da}.csv", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("x", "y")} for r in conc]).to_csv(
        out / f"peak_gene_concordance_{c_da}.csv", index=False)
    for b in [b for b in (*BLOCKS, "target MSI (ported)", "target MSI (OT)", "target ATAC (unadjusted)") if b in W]:
        W[b].to_csv(out / f"weights_{b.replace(' ', '_').replace('(', '').replace(')', '')}.csv.gz")
    if teacher is not None:
        for b in [b for b in ("target RNA", "target ATAC", "target MSI (ported)", "target MSI (OT)",
                              "target ATAC (unadjusted)") if b in teacher]:
            teacher[b].to_csv(out / f"teacher_weights_{b.replace(' ', '_').replace('(', '').replace(')', '')}.csv.gz")
    results = dict(U=U, sv=sv, comps=comps, c_da=c_da, W=W, teacher=teacher, summary=summary, anchor=args.anchor,
                   atac_covariates=args.atac_covariates,
                   source_spots=args.source_spots, predictor_k=predictor["k"], predictor_cv=predictor["cv"],
                   overlap=overlap, msi_R_target=pd.Series(msi_R_t, index=S["metabolites"]),
                   msi_R_cv=pd.Series(msi_R, index=S["metabolites"]))
    with open(out / "results.pkl", "wb") as handle:
        pickle.dump(results, handle)
    with open(out / "panels.pkl", "wb") as handle:
        pickle.dump((panel5, panel6), handle)
    return panel5, panel6


def _gene_scatter(src, tgt):
    s, t = upper_series(src, src.index), upper_series(tgt, tgt.index)
    shared = s.index.intersection(t.index)
    return dict(x=s[shared].to_numpy(), y=t[shared].to_numpy(), genes=shared,
                rho=spearmanr(s[shared], t[shared]).correlation)


# --------------------------------------------------------------------------- figure
def score_map(ax, xy, s, title, *, size, flip=(1, -1)):
    v = np.percentile(np.abs(s), 98)
    xy = xy * np.asarray(flip)
    order = np.argsort(np.abs(s))
    ax.scatter(xy[order, 0], xy[order, 1], c=s[order], cmap="RdBu_r", vmin=-v, vmax=v, s=size,
               linewidths=0, rasterized=True, marker="s")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#ededed")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11)


def pooled_by_location(xy, s):
    """Multiome nuclei sit on their MultiGATE-matched P22 pixel: average per pixel."""
    df = pd.DataFrame({"x": np.round(xy[:, 0], 3), "y": np.round(xy[:, 1], 3), "s": s})
    g = df.groupby(["x", "y"], sort=False)["s"].mean().reset_index()
    return g[["x", "y"]].to_numpy(), g["s"].to_numpy()


def draw_fig5(ns, panel, output_file):
    comps, c_da, summ = panel["comps"], panel["c_da"], panel["summary"]
    K = len(comps)
    tlabel = TARGET_LABEL[panel["target"]]
    map_rows = [("Visium source\n(RNA + MSI)", panel["source_xy"], panel["scores"]["source"], (1, -1), 22)]
    if panel["target"] == "multiome":
        map_rows.append((f"{tlabel}\n(student, on matched\nP22 pixels)", panel["target_xy"], panel["scores"]["target"],
                         (-1, -1), 9))
    else:
        map_rows.append((f"{tlabel}\n(student)", panel["target_xy"], panel["scores"]["target"], (1, -1), 9))
        map_rows.append((f"{tlabel}\n(teacher, P22 graph)", panel["target_xy"], panel["teacher_score"], (1, -1), 9))
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    with plt.rc_context(ns["FIG_RCPARAMS"]):
        heights = [4.3, 2.9 * len(map_rows), 4.6]
        fig = plt.figure(figsize=(19, sum(heights)), constrained_layout=True)
        rows = fig.subfigures(3, 1, height_ratios=heights)

        # row 1: variance overview + cross-dataset RNA agreement
        ax_h, ax_c = rows[0].subplots(1, 2, gridspec_kw={"width_ratios": [1.5, 1.0]})
        cols = [f"share {b}" for b in BLOCKS]
        body = summ[summ["component"].isin(comps)].set_index("component")
        H = body[cols].to_numpy().T * 100
        im = ax_h.imshow(H, cmap="Purples", aspect="auto")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                ax_h.text(j, i, f"{H[i, j]:.1f}", ha="center", va="center", fontsize=10,
                          color="white" if H[i, j] > 0.6 * H.max() else "black")
        ax_h.set_yticks(range(len(BLOCKS)))
        ax_h.set_yticklabels([b.replace("target", tlabel.split(" ")[0] if panel["target"] == "spatial" else "multiome")
                              for b in BLOCKS])
        ax_h.set_xticks(range(K))
        ax_h.set_xticklabels([f"{c}\nDA r={body.loc[c, 'dopamine_weight']:+.2f}" + ("\n(dopamine)" if c == c_da else "")
                              for c in comps], fontsize=10)
        for t in ax_h.get_xticklabels():
            if t.get_text().startswith(c_da + "\n"):
                t.set_fontweight("bold")
        ax_h.set_title("Shared components: share of each block's latent-linked signal (%)"
                       + ("\nC1 anchored to measured source dopamine" if panel.get("anchor") == "dopamine" else ""))
        cb = fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.01)
        cb.set_label("% of block's latent-feature correlation", fontsize=10)
        x = np.arange(K)
        width = 0.38 if panel["teacher_score"] is not None else 0.6
        ax_c.bar(x - (width / 2 if panel["teacher_score"] is not None else 0), body["rna_src_vs_tgt_rho"],
                 width=width, color="#4a6fa5", label="source vs target RNA weights")
        if panel["teacher_score"] is not None:
            ax_c.bar(x + width / 2, body["teacher_vs_student target RNA rho"], width=width, color="#c98b2b",
                     label="teacher vs student (P22) RNA weights")
        ax_c.axhline(0, color="#777777", lw=0.8)
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(comps)
        ax_c.set_ylim(min(-0.1, ax_c.get_ylim()[0]), 1.0)
        ax_c.set_ylabel("Spearman $\\rho$ of gene weights")
        ax_c.set_title("Do components mean the same in both datasets?")
        ax_c.legend(frameon=False, fontsize=9, loc="upper right")
        ns["_fig_panel_letter"](ax_h, next(letters))
        ns["_fig_panel_letter"](ax_c, next(letters))

        # row 2: component score maps (rows = datasets, columns = components)
        grid = rows[1].subplots(len(map_rows), K, squeeze=False)
        for r, (name, xy, S, flip, size) in enumerate(map_rows):
            for k in range(K):
                xy_k, s_k = (pooled_by_location(xy, S[:, k]) if "matched" in name else (xy, S[:, k]))
                score_map(grid[r, k], xy_k, s_k, comps[k] + (" (dopamine)" if comps[k] == c_da else "") if r == 0 else "",
                          size=size, flip=flip)
                if comps[k] == c_da:
                    for spine in grid[r, k].spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor("#222222")
                        spine.set_linewidth(1.6)
            grid[r, 0].set_ylabel(name, fontsize=11)
        ns["_fig_panel_letter"](grid[0, 0], next(letters), x=-0.25)
        rows[1].suptitle("Component scores (same latent direction in every dataset; colour clipped at the 98th "
                         "percentile of |score|, red = positive dopamine side)", fontsize=11)

        # row 3: the dopamine component
        axes = rows[2].subplots(1, 5, gridspec_kw={"width_ratios": [0.55, 0.55, 0.55, 1.2, 1.9]})
        ns["_fig_loadings_dotplot"](axes[0], ns["_fig_top_loadings"](panel["metabolites"], 6),
                                    title=f"{c_da}: top-6 metabolites\n(measured source MSI)")
        ns["_fig_loadings_dotplot"](axes[1], ns["_fig_top_loadings"](panel["genes_source"], 6),
                                    title=f"{c_da}: top-6 genes\n(Visium source)", italic=True)
        ns["_fig_loadings_dotplot"](axes[2], ns["_fig_top_loadings"](panel["genes_target"], 6),
                                    title=f"{c_da}: top-6 genes\n({tlabel})", italic=True)
        g = panel["gene_scatter"]
        ax = axes[3]
        hb = ax.hexbin(g["x"], g["y"], gridsize=50, bins="log", cmap="Greys", mincnt=1, linewidths=0, rasterized=True)
        marks = [m.upper() for m in GENE_MARKERS]
        idx = [i for i, gname in enumerate(g["genes"]) if gname in marks]
        ax.scatter(g["x"][idx], g["y"][idx], s=22, color="#c0392b", zorder=3)
        for i in idx:
            name = GENE_MARKERS[marks.index(g["genes"][i])]
            ax.annotate(name, (g["x"][i], g["y"][i]), xytext=(3, 2), textcoords="offset points", fontsize=8,
                        style="italic")
        ax.axhline(0, color="#999999", lw=0.6)
        ax.axvline(0, color="#999999", lw=0.6)
        ax.set_xlabel(f"{c_da} gene weight, Visium source")
        ax.set_ylabel(f"{c_da} gene weight, {tlabel}")
        ax.set_title(f"{c_da} gene weights across datasets\n(Spearman $\\rho$ = {g['rho']:.2f}, "
                     f"{len(g['genes']):,} shared genes)")
        fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02).set_label("genes per bin", fontsize=9)
        ns["_fig_draw_enrichment_convergence"](
            axes[4], panel["convergence"], P.TOP_K,
            title=f"Gene-level convergence with the\nNMF dopamine program (top {P.TOP_K})")
        axes[4].set_ylabel(f"{c_da} gene weight (corr. with score)")
        for slot, ax in enumerate(axes):
            if slot in (1, 2):
                continue
            ns["_fig_panel_letter"](ax, next(letters), x=-0.06 if slot else -0.3)
        fig.suptitle(f"Shared PLS components: Visium source (RNA + MSI) and {tlabel} (RNA + ATAC), "
                     f"student RNA-expert latent ({panel['run']})", fontsize=14)
        for suffix in (".png", ".svg"):
            fig.savefig(output_file.with_suffix(suffix), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"[fig] wrote {output_file.with_suffix('.png')}", flush=True)


# -------------------------------------------------------------------------- compare
def compare(args):
    R = {}
    base = anchor_base(args)
    for t in ("spatial", "multiome"):
        with open(base / t / "results.pkl", "rb") as handle:
            R[t] = pickle.load(handle)
    a, b = R["spatial"], R["multiome"]
    cos = np.abs(a["U"].T @ b["U"])
    rows = []
    for i, ci in enumerate(a["comps"]):
        for j, cj in enumerate(b["comps"]):
            rows.append(dict(spatial=ci, multiome=cj, cosine_u=float(a["U"][:, i] @ b["U"][:, j]),
                             source_MSI_r=np.corrcoef(a["W"]["source MSI"][ci], b["W"]["source MSI"][cj])[0, 1],
                             source_RNA_rho=spearmanr(a["W"]["source RNA"][ci], b["W"]["source RNA"][cj],
                                                      nan_policy="omit").correlation,
                             target_RNA_rho=spearmanr(a["W"]["target RNA"][ci],
                                                      b["W"]["target RNA"][cj].reindex(a["W"]["target RNA"].index),
                                                      nan_policy="omit").correlation,
                             peaks_rho=spearmanr(a["W"]["target ATAC"][ci],
                                                 b["W"]["target ATAC"][cj].reindex(a["W"]["target ATAC"].index),
                                                 nan_policy="omit").correlation))
    df = pd.DataFrame(rows)
    df.to_csv(base / "cross_target_matching.csv", index=False)
    print(df[df.spatial.eq(a["c_da"]) | df.multiome.eq(b["c_da"])].round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (key, title) in zip(axes, [("cosine_u", "latent direction (cosine of u)"),
                                       ("source_MSI_r", "source metabolite weights (r)"),
                                       ("peaks_rho", "peak weights (Spearman)")]):
        Mx = df.pivot(index="spatial", columns="multiome", values=key).loc[a["comps"], b["comps"]]
        im = ax.imshow(Mx.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        for i in range(Mx.shape[0]):
            for j in range(Mx.shape[1]):
                ax.text(j, i, f"{Mx.iat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(Mx.shape[1]))
        ax.set_xticklabels([c + ("*" if c == b["c_da"] else "") for c in Mx.columns])
        ax.set_yticks(range(Mx.shape[0]))
        ax.set_yticklabels([c + ("*" if c == a["c_da"] else "") for c in Mx.index])
        ax.set_xlabel("multiome analysis")
        ax.set_ylabel("P22 spatial analysis")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.02)
    fig.suptitle("Matching the two single-target analyses (* = dopamine component)")
    for suffix in (".png", ".svg"):
        fig.savefig(base / f"cross_target_matching{suffix}", dpi=200, bbox_inches="tight")
    print(f"[fig] wrote {base / 'cross_target_matching.png'}")


ANCHOR_DIR = {"dopamine": None, "none": "unanchored", "msi": "msi_svd"}


def components_subdir(anchor, source_spots):
    sub = ANCHOR_DIR[anchor]
    return f"{sub or 'anchored'}_intact" if source_spots == "intact" else sub


def anchor_base(args):
    sub = components_subdir(args.anchor, args.source_spots)
    return args.out_dir if sub is None else args.out_dir / sub


def out_dir(args):
    base = anchor_base(args)
    if args.run != "fixed s3":
        base = base / "runs" / args.run.replace(" ", "_")
    return base / args.target


def main():
    args = parse_args()
    if args.target == "compare":
        return compare(args)
    out = out_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    ns = F.script5_namespace(P.RUNS[args.run])
    if args.redraw:
        with open(out / "panels.pkl", "rb") as handle:
            panel5, panel6 = pickle.load(handle)
    else:
        panel5, panel6 = analyse(args, ns)
    if args.tables_only:
        return
    draw_fig5(ns, panel5, out / "SMA_mouse_shared_components")
    F.draw_fig6(ns, panel6, out / "SMA_mouse_shared_components_enrichment")


if __name__ == "__main__":
    main()

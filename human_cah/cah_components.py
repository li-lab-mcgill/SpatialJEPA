try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Manuscript Fig. 7 (SEA-AD CaH transfer) from shared PLS components instead of MOFA+.
#
# Same recipe as the mouse analysis (rna_msi/shared_msi_components.py):
#   latent   the student's RNA-expert latent H["st"]["q_mu"] for every source spot
#            (human Visium RNA+MSI, obsm student_X_emb_st) and every CaH nucleus
#            (obsm X_spatialmeta_student, which the hand-off writes from st.q_mu),
#            z-scored per dimension within each dataset
#   blocks   source RNA (SCT data, model genes), source MSI (measured; the decoder's
#            metabolites, log1p), CaH snRNA RNA (the aligned view MOFA used)
#   U        `--anchor msi` (default): SVD of the source MSI block, C_srcMSI/|C_srcMSI| (no
#            dopamine label, no CaH data); the dopamine component is the one with the largest
#            |dopamine weight| (C1 in every fixed run). Outputs pls_msi/, figures_pls_msi/.
#            `--anchor dopamine`: the earlier C1 = Sigma_src^-1 corr(Z, dopamine), C2..C5 the
#            stacked SVD after removing it (pls/, figures_pls/)
#   weights  corr(feature, component score) in the feature's own dataset; panel a's metabolites
#            `--msi-weights ot` (default): recovered in the CaH nuclei by OT pairing with the source
#            spots + a PLS decoder fitted on the nuclei (shared_msi_components.ot_msi_weights;
#            OT on a random 40,000-nucleus subsample, all source spots; a second subsample and a
#            random-rotation null in components_summary.csv); outputs pls_msi_ot/, figures_pls_msi_ot/.
#            `ported`: the source-fit predictor applied to the nuclei (rank and R_m from
#            leave-one-section-out on the three source sections), pls_msi/, figures_pls_msi/
#   sections `--source-batches section` (default): the three source sections are treated as batches,
#            as in model training (batch_keys=["section"]): latent, MSI and RNA are centred within
#            section before the source correlations (U, weights, predictor), and the OT transfer runs
#            per section (latent and MSI standardised within it) and is averaged over sections
#            (shared_msi_components.ot_barycentre_batched). Pooled, section C1's ~4x dopamine offset
#            and the latent's section signature (85% section accuracy) turned section membership into
#            apparent cell-type differences in the nuclei. Tag _sec. `none`: pooled, as before
#   validate `--stage validate`: each source section held out in turn, its metabolite weights
#            from measured MSI vs copied from / ported from / OT-recovered from the other two
#   Xenium   panel f: the CaH snRNA C1 gene weights applied to the section's z-scored
#            expression (replaces MOFA's RNA projection; see project_spatial_onto_factor).
#            panel e (with `--msi-weights ot`): the dopamine OT-transferred to the CaH nuclei, carried
#            to the Xenium cells by a ridge map on the genes shared by snRNA and the Xenium panel,
#            fitted on the nuclei (z-scored genes, 5-fold CV R reported) and applied to the
#            section's z-scored expression; replaces the student decoder on the Xenium latent
#
# `--stage components` writes <cah>/pls<tags>/results.pkl; `--stage figure` draws the manuscript
# layout with cah_fig7_driver's board and 2__'s panel functions, the MOFA model replaced
# by a mofax-like view of the PLS components, into <cah>/figures_pls<tags>/.
#
# Env: $PY_ANALYSIS

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

LIPID_GP = Path(__file__).resolve().parent
sys.path.insert(0, str(LIPID_GP))
sys.path.insert(0, os.environ["SPATIALJEPA_RNA_MSI"])
import cah_cell_runner as D  # noqa: E402
import msi_components_common as P  # noqa: E402
import shared_msi_components as SC  # noqa: E402
from plot_dopamine_program import PLSModel  # noqa: E402

OUTPUTS = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs")
RUNS = ("fix_s0", "fix_s1", "fix_s2", "induced_s0", "seed0", "seed1", "seed2")  # early_sK: fix_sK with the early-stopped teacher
BLOCKS = ("source RNA", "source MSI", "target RNA")
K = 5
NAME = "PLS C{}"


def parse_args():
    p = argparse.ArgumentParser(description="SEA-AD CaH Fig. 7 from shared PLS components.")
    p.add_argument("--stage", choices=("components", "figure", "validate"), required=True)
    p.add_argument("--run", choices=RUNS, required=True)
    p.add_argument("--anchor", choices=("msi", "dopamine"), default="msi")
    p.add_argument("--msi-weights", choices=("ot", "ported"), default="ot",
                   help="panel a metabolite weights: OT-recovered in the nuclei, or ported")
    p.add_argument("--ot-reg", type=float, default=0.001,
                   help="entropic OT regularisation (relative to mean cost); 0.001 chosen on held-out sections")
    p.add_argument("--source-batches", choices=("section", "none"), default="section",
                   help="treat the source sections as batches (header, `sections`) or pool them")
    return p.parse_args()


N_OT = 40_000          # nuclei per OT subsample
N_NULL = 10            # random rotations of the CaH latent before pairing


XENIUM_PATH = Path(f"{os.environ['SPATIALJEPA_HOME']}/data/SEA_AD/sea_ad_cah/CaH_Xenium_final.2026-01-07.h5ad")


def paths(run, anchor="msi", msi_weights="ported", ot_reg=0.01, source_batches="none"):
    cah = OUTPUTS / "human_cah_transfer" / run / "cah"
    tag = ("_msi" if anchor == "msi" else "") + ("_ot" if msi_weights == "ot" else "")
    if msi_weights == "ot" and ot_reg != 0.01:
        tag += f"_reg{ot_reg:g}"
    if source_batches == "section":
        tag += "_sec"
    return dict(cah=cah, bundle=OUTPUTS / "spatialjepa_models" / (f"V11T17-102_{run}" if run.startswith("seed") else f"V11T17-102__{run}"),
                pls=cah / f"pls{tag}", figures=cah / f"figures_pls{tag}")


# ----------------------------------------------------------------------- components
def load_source(bundle):
    import anndata as ad
    import joblib

    joint = ad.read_h5ad(bundle / "joint_adata.h5ad")
    feats = list(joblib.load(bundle / "student_pls_decoder.joblib")["target_features"])
    X = joint[:, feats].X
    msi = np.log1p(X.toarray() if sp.issparse(X) else np.asarray(X, dtype=float))
    st = joint.var["type"].eq("ST").to_numpy()
    rna = joint.layers["SCT_data"][:, st]
    rna = sp.csc_matrix(rna) if sp.issparse(rna) else np.asarray(rna, dtype=float)
    print(f"[pls] source: {joint.n_obs:,} spots ({', '.join(joint.obs['sample_id'].astype(str).unique())}), "
          f"{st.sum()} genes, {len(feats)} metabolites", flush=True)
    return dict(Z=np.asarray(joint.obsm["student_X_emb_st"]), rna=rna,
                sections=joint.obs["sample_id"].astype(str).to_numpy(),
                genes=joint.var_names[st].str.replace(r"^rna:", "", regex=True), msi=msi,
                metabolites=pd.Index(feats))


def load_target(cah):
    with h5py.File(cah / "snrna_spatialmeta_aligned.h5mu", "r") as f:
        cells = pd.Index(P.read_index(f["mod/rna/obs"]))
        genes = pd.Index(P.read_index(f["mod/rna/var"])).str.replace(r"^rna:", "", regex=True)
        rna = P.read_csr(f["mod/rna/X"]).tocsc()
        if not np.array_equal(P.read_index(f["mod/msi/obs"]), cells):
            raise ValueError("RNA and MSI nuclei differ")
        Z = f["mod/msi/obsm/X_spatialmeta_student"][:]
        idx = f["mod/msi/var"].attrs.get("_index", "_index")
        msi_names = f[f"mod/msi/var/{idx}"][:].astype(str)
        dopamine = f["mod/msi/X"][:, list(msi_names).index("msi:Dopamine")]
        subclass = pd.Series(np.asarray(P.read_obs_column(f["mod/rna/obs"], "Subclass"))).str.replace(
            "Subclass$", "", regex=True).to_numpy()
    print(f"[pls] CaH snRNA: {len(cells):,} nuclei, {len(genes)} genes", flush=True)
    return dict(cells=cells, genes=genes, rna=rna, Z=np.asarray(Z, dtype=float), dopamine=dopamine,
                subclass=subclass)


def centre_within(X, batches):
    """Dense copy of X with each batch's column means removed."""
    X = X.toarray() if sp.issparse(X) else np.array(X, dtype=float)
    for b in np.unique(batches):
        m = batches == b
        X[m] -= X[m].mean(0)
    return X


def msn_specificity(values, subclass):
    """AUROC of MSN vs other neuronal nuclei (Vip, Sst Chodl, LSX, GABA subclasses ...)."""
    from sklearn.metrics import roc_auc_score

    sub = pd.Series(subclass)
    msn = sub.str.contains("MSN").to_numpy()
    other = sub.str.contains("GABA|Gaba|Vip|Sst|LSX|Lamp5|Pvalb|Chodl").to_numpy() & ~msn
    keep = msn | other
    return float(roc_auc_score(msn[keep], np.asarray(values)[keep]))


def xenium_latent(cah):
    donor = ".".join(D.XENIUM_SECTION.split(".")[:3])
    bundle = np.load(cah / "spatialmeta_handoff" / "outputs" / "merfish" / donor / D.XENIUM_SECTION
                     / "embeddings.npz", allow_pickle=True)
    return pd.DataFrame(np.asarray(bundle["student"], dtype=float),
                        index=pd.Index(np.asarray(bundle["obs_names"], dtype=str)))


def ot_recovery(S, Z0, U, comps, W_src, reg=0.01, batches=None):
    """OT-recovered metabolite weights in the CaH nuclei (header, `--msi-weights ot`), plus their
    agreement with the source-measured weights, a second subsample and a rotation null. With
    `batches`, the transfer runs per source section (header, `sections`). Also returns the paired
    (standardised) MSI of the first subsample and its nucleus indices."""
    Zsrc = S["Z"] if batches is not None else Z0["source"]

    def pair(Zt):
        if batches is None:
            return SC.ot_barycentre(Zsrc, SC._standardize_cols(S["msi"]), Zt, reg=reg)
        return SC.ot_barycentre_batched(Zsrc, S["msi"], Zt, batches, reg=reg)

    rng = np.random.default_rng(0)
    n = len(Z0["target"])
    out = {}
    for seed in (0, 1):
        pick = np.random.default_rng(seed).choice(n, size=min(N_OT, n), replace=False)
        Zt = SC.standardize(Z0["target"][pick])
        groups = np.random.default_rng(seed).integers(0, 5, len(pick))
        W, dec, Y = SC.ot_msi_weights(Zsrc, S["msi"], Zt, U, groups, reg=reg, batches=batches,
                                      return_paired=True)
        out[seed] = (pd.DataFrame(W, index=S["metabolites"], columns=comps), dec, Zt, groups, Y, pick)
    W0, dec, Zt, groups, Y0, pick0 = out[0]
    null = []
    for _ in range(N_NULL):
        Q = np.linalg.qr(rng.standard_normal((Zt.shape[1], Zt.shape[1])))[0]
        Y = pair(Zt @ Q)
        dn = SC.fit_msi_predictor(Zt, Y, groups, method="simpls")
        Wn = SC.ported_msi_weights(dn, Zt, U)[0]
        null.append([np.corrcoef(Wn[:, k], W_src.iloc[:, k])[0, 1] for k in range(len(comps))])
    null = np.array(null)
    stats = pd.DataFrame(dict(
        r_ot_vs_source=[np.corrcoef(W0[c], W_src[c])[0, 1] for c in comps],
        r_ot_rotation_null_median=np.median(null, 0), r_ot_rotation_null_max=null.max(0),
        r_ot_subsample_1_vs_0=[np.corrcoef(W0[c], out[1][0][c])[0, 1] for c in comps]), index=comps)
    print(f"[ot] decoder rank {dec['k']} on {len(Zt):,} nuclei; recovery vs source-measured weights:\n"
          + stats.round(3).to_string(), flush=True)
    return W0, dec["k"], stats, null, Y0, pick0


def dopamine_xenium_map(T, pick, dopamine, alphas=np.logspace(0, 5, 11)):
    """Ridge map from z-scored snRNA genes shared with the Xenium panel to the nuclei's transferred
    dopamine (header, panel e). Returns (coefficients over upper-case gene names, 5-fold CV R)."""
    import anndata as ad
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict

    panel = pd.Index(ad.read_h5ad(XENIUM_PATH, backed="r").var_names.astype(str)).str.upper()
    genes = pd.Index(T["genes"]).str.upper()
    cols = np.flatnonzero(genes.isin(panel) & ~genes.duplicated())
    X = T["rna"].tocsr()[pick][:, cols].toarray().astype(float)
    X = (X - X.mean(0)) / X.std(0).clip(1e-6)
    ridge = RidgeCV(alphas=alphas).fit(X, dopamine)
    pred = cross_val_predict(RidgeCV(alphas=alphas), X, dopamine, cv=5)
    r_cv = float(np.corrcoef(pred, dopamine)[0, 1])
    print(f"[ot] Xenium dopamine map: ridge on {len(cols)} shared genes (alpha {ridge.alpha_:g}), "
          f"5-fold CV r {r_cv:.2f}", flush=True)
    return pd.Series(ridge.coef_, index=genes[cols]), r_cv


def stage_components(run, anchor, msi_weights="ot", ot_reg=0.01, source_batches="none"):
    from scipy.stats import spearmanr

    p = paths(run, anchor, msi_weights, ot_reg, source_batches)
    S, T = load_source(p["bundle"]), load_target(p["cah"])
    batches = S["sections"] if source_batches == "section" else None
    src_rna, src_msi, src_Z = S["rna"], S["msi"], S["Z"]
    if batches is not None:                     # sections as batches (header, `sections`)
        src_rna, src_msi, src_Z = (centre_within(X, batches) for X in (S["rna"], S["msi"], S["Z"]))
    Z0 = {"source": SC.standardize(src_Z), "target": SC.standardize(T["Z"])}
    blocks = {"source RNA": ("source", src_rna, S["genes"]),
              "source MSI": ("source", src_msi, S["metabolites"]),
              "target RNA": ("target", T["rna"], T["genes"])}
    stats = {b: P.column_stats(X) for b, (_, X, _) in blocks.items()}
    C = {b: SC.corr_matrix(Z0[d], X, stats[b]) for b, (d, X, _) in blocks.items()}
    M = np.hstack([C[b] / np.linalg.norm(C[b]) for b in BLOCKS])
    da = S["metabolites"].get_loc("msi:Dopamine")
    if anchor == "msi":
        Cm = C["source MSI"]
        U = np.linalg.svd(Cm / np.linalg.norm(Cm), full_matrices=False)[0][:, :K].copy()
    else:
        sigma = Z0["source"].T @ Z0["source"] / (len(Z0["source"]) - 1)
        u0 = np.linalg.solve(sigma, C["source MSI"][:, da])
        u0 /= np.linalg.norm(u0)
        rest = np.linalg.svd(M - np.outer(u0, u0 @ M), full_matrices=False)[0]
        U = np.column_stack([u0, rest[:, :K - 1]])
    comps = [NAME.format(k + 1) for k in range(K)]
    for k in range(K):                       # dopamine (source) loads positively
        U[:, k] *= np.sign(SC.direction_weights(C["source MSI"], Z0["source"], U[:, k])[da]) or 1.0
    W = {b: pd.DataFrame(np.column_stack([SC.direction_weights(C[b], Z0[d], U[:, k]) for k in range(K)]),
                         index=names, columns=comps) for b, (d, _, names) in blocks.items()}
    predictor = SC.fit_msi_predictor(Z0["source"], src_msi, S["sections"])
    msi_R = predictor["R"]
    ported, msi_R_t = SC.ported_msi_weights(predictor, Z0["target"], U)
    W["target MSI (ported)"] = pd.DataFrame(ported, index=S["metabolites"], columns=comps)
    overlap = SC.latent_overlap(Z0["source"], Z0["target"])
    print(f"[ported] PLS predictor rank {predictor['k']} (leave-one-section-out); dopamine R_cv "
          f"{msi_R[da]:.2f}, target-adjusted {msi_R_t[da]:.2f}; median R_cv {np.median(msi_R):.2f}; overlap "
          + ", ".join(f"{k} {v:.2f}" for k, v in overlap.items()), flush=True)
    c_da = comps[int(np.argmax(np.abs(W["source MSI"].loc["msi:Dopamine"].to_numpy())))]
    print(f"[pls] anchor {anchor}: dopamine component {c_da}", flush=True)
    ot_stats, ot_null, ot_k, ot_dopamine, xen_map, xen_r = None, None, None, None, None, None
    if msi_weights == "ot":
        W["target MSI (OT)"], ot_k, ot_stats, ot_null, Y0, pick0 = ot_recovery(
            S, Z0, U, comps, W["source MSI"], reg=ot_reg, batches=batches)
        ot_dopamine = pd.Series(Y0[:, da], index=T["cells"][pick0])
        xen_map, xen_r = dopamine_xenium_map(T, pick0, Y0[:, da])
        spec = {"transferred dopamine": msn_specificity(Y0[:, da], T["subclass"][pick0]),
                "old decoder dopamine": msn_specificity(T["dopamine"][pick0], T["subclass"][pick0]),
                **{c: msn_specificity((Z0["target"][pick0] @ U)[:, k], T["subclass"][pick0])
                   for k, c in enumerate(comps)}}
        print("[ot] AUROC MSN vs other neuronal nuclei: "
              + ", ".join(f"{k} {v:.2f}" for k, v in spec.items()), flush=True)
        g = pd.Series(Y0[:, da]).groupby(T["subclass"][pick0]).agg(["mean", "size"])
        print("[ot] transferred dopamine by subclass (z, >=50 nuclei): "
              + ", ".join(f"{i} {v:+.2f}" for i, v in g.loc[g["size"] >= 50, "mean"].sort_values(ascending=False).items()),
              flush=True)
    scores = pd.DataFrame((Z0["target"] @ U).astype(np.float32), index=T["cells"], columns=comps)
    Xz = xenium_latent(p["cah"])
    xenium = pd.DataFrame(SC.standardize(Xz.to_numpy()) @ U, index=Xz.index, columns=comps)

    rng = np.random.default_rng(0)
    pick = rng.choice(len(T["cells"]), size=min(100_000, len(T["cells"])), replace=False)
    rows = []
    for k, c in enumerate(comps):
        src_g, tgt_g = SC.upper_series(W["source RNA"][c], S["genes"]), SC.upper_series(W["target RNA"][c], T["genes"])
        shared = src_g.index.intersection(tgt_g.index)
        rows.append(dict(component=c,
                         share_of_stack=np.linalg.norm(U[:, k] @ M) ** 2 / np.linalg.norm(M) ** 2,
                         **{f"share {b}": np.linalg.norm(U[:, k] @ C[b]) ** 2 / np.linalg.norm(C[b]) ** 2 for b in BLOCKS},
                         dopamine_weight=W["source MSI"].loc["msi:Dopamine", c],
                         dopamine_weight_ported=W["target MSI (ported)"].loc["msi:Dopamine", c],
                         **({} if ot_stats is None else dict(
                             dopamine_weight_ot=W["target MSI (OT)"].loc["msi:Dopamine", c],
                             dopamine_rank_ot=int((W["target MSI (OT)"][c] > W["target MSI (OT)"].loc["msi:Dopamine", c]).sum()) + 1,
                             **ot_stats.loc[c].to_dict())),
                         r_imputed_dopamine=spearmanr(scores[c].to_numpy()[pick], T["dopamine"][pick]).correlation,
                         rna_src_vs_tgt_rho=spearmanr(src_g[shared], tgt_g[shared]).correlation,
                         top_msi=", ".join(m.replace("msi:", "") for m in W["source MSI"][c].nlargest(5).index),
                         top_msi_ported=", ".join(m.replace("msi:", "")
                                                  for m in W["target MSI (ported)"][c].nlargest(5).index),
                         **({} if ot_stats is None else dict(top_msi_ot=", ".join(
                             m.replace("msi:", "") for m in W["target MSI (OT)"][c].nlargest(5).index))),
                         top_rna=", ".join(W["target RNA"][c].nlargest(6).index),
                         bottom_rna=", ".join(W["target RNA"][c].nsmallest(6).index)))
    summary = pd.DataFrame(rows)
    tops = [c for c in summary if c.startswith(("top_", "bottom_"))]
    print(summary.drop(columns=tops).round(3).to_string(index=False))
    print(summary[["component", *tops]].to_string(index=False), flush=True)
    p["pls"].mkdir(parents=True, exist_ok=True)
    summary.to_csv(p["pls"] / "components_summary.csv", index=False)
    for b in [b for b in (*BLOCKS, "target MSI (ported)", "target MSI (OT)") if b in W]:
        W[b].to_csv(p["pls"] / f"weights_{b.replace(' ', '_').replace('(', '').replace(')', '')}.csv.gz")
    with open(p["pls"] / "results.pkl", "wb") as handle:
        pickle.dump(dict(U=U, comps=comps, c_da=c_da, W=W, scores=scores, xenium=xenium, summary=summary,
                         anchor=anchor, msi_R_cv=pd.Series(msi_R, index=S["metabolites"]),
                         msi_R_target=pd.Series(msi_R_t, index=S["metabolites"]), predictor_k=predictor["k"],
                         predictor_cv=predictor["cv"], overlap=overlap, msi_weights=msi_weights,
                         ot_stats=ot_stats, ot_null=ot_null, ot_decoder_k=ot_k, ot_reg=ot_reg,
                         source_batches=source_batches, ot_dopamine=ot_dopamine,
                         xenium_dopamine_map=xen_map, xenium_dopamine_map_cv_r=xen_r,
                         msn_specificity=None if ot_stats is None else spec), handle)
    print(f"[pls] wrote {p['pls']}", flush=True)


# --------------------------------------------------------------------------- figure
def stage_figure(run, anchor, msi_weights="ot", ot_reg=0.01, source_batches="none"):
    import scanpy as sc

    p = paths(run, anchor, msi_weights, ot_reg, source_batches)
    p["figures"].mkdir(parents=True, exist_ok=True)
    with open(p["pls"] / "results.pkl", "rb") as handle:
        R = pickle.load(handle)
    factor = R.get("c_da", R["comps"][0])
    k_da = R["comps"].index(factor)
    msi_block = next(b for b in ("target MSI (OT)", "target MSI (ported)", "source MSI") if b in R["W"])
    print(f"[pls] figure on {factor}; metabolite weights from '{msi_block}'", flush=True)

    ns = {"__name__": "fig7_pls", "__file__": str(D.DOWNSTREAM_SCRIPT)}
    root_patch = [(
        'CAH_OUTPUT_ROOT = Path(\n    __import__("os").environ["SPATIALJEPA_HOME"], "outputs/sea_ad_lipid_gps_minimal/cah"\n)',
        f'CAH_OUTPUT_ROOT = Path({str(p["cah"])!r})',
    )]
    D.run_cells(D.DOWNSTREAM_SCRIPT, ("load data", "load data and MOFA model"), ns,
                patches={"load data and MOFA model": root_patch})
    real = ns["mofa"]
    rna = ns["scrna_msi_mudata"].mod["rna"]
    weights = {}
    for view, block in (("rna", "target RNA"), ("msi", msi_block)):
        idx = real.get_weights(views=[view], df=True).index
        frame = R["W"][block]
        frame = frame[~frame.index.duplicated()]
        key = idx.str.replace(r"^rna:", "", regex=True) if view == "rna" else idx
        frame = frame.reindex(key)
        frame.index = idx
        print(f"[pls] {view}: {int(frame.iloc[:, 0].isna().sum())}/{len(idx)} MOFA features without a weight")
        weights[view] = frame.dropna()
    scores = R["scores"].reindex(rna.obs_names.astype(str))
    if scores.isna().any().any():
        raise ValueError("PLS scores missing for some nuclei")
    ns["mofa"] = PLSModel(scores, weights)
    ns["mofa_adata"] = sc.AnnData(
        scores.to_numpy(), obs=ns["scrna_msi_mudata"].mod["msi"].obs,
        obsm=rna.obsm, var=pd.DataFrame(index=R["comps"]))
    ns["PRIMARY"] = "Overall AD neuropathological Change"
    figure_cells = ("Reusable Factor 1 multi-panel figure", "Transfer-characterisation figure",
                    "spatial cross-correlation between the three panels")
    D.run_cells(D.DOWNSTREAM_SCRIPT, figure_cells, ns, defs_only=figure_cells)

    rna_weights = R["W"]["target RNA"]
    rna_weights = rna_weights[~rna_weights.index.duplicated()]

    def project_spatial_onto_factor(spatial, mofa_model=None, factor_index=k_da, key="mofa_factor1", **_):
        """PLS stand-in for MOFA's RNA projection: the snRNA C-k gene weights applied to the
        section's expression, z-scored per gene within the section (|w|-normalised weighted
        sum over the panel genes the weights cover). The Xenium RNA-expert latent, built from
        the 464-gene panel, tracked cell identity poorly and varied across seeds (r 0.67-0.80);
        this projection agrees across fixed seeds at r 0.93-0.99 and ranks MSNs top."""
        X = spatial.X.toarray() if sp.issparse(spatial.X) else np.asarray(spatial.X, dtype=float)
        Z = (X - X.mean(0)) / X.std(0).clip(1e-6)
        genes = pd.Index(spatial.var_names.astype(str)).str.upper()
        w = rna_weights.iloc[:, factor_index]
        w = w.groupby(w.index.str.upper()).first()
        shared = genes.isin(w.index)
        wv = w.reindex(genes[shared]).to_numpy()
        print(f"[pls] Xenium {rna_weights.columns[factor_index]}: gene-weight projection over "
              f"{int(shared.sum())}/{len(genes)} panel genes", flush=True)
        spatial.obs[key] = Z[:, shared] @ wv / np.abs(wv).sum()
        return spatial

    ns["project_spatial_onto_factor"] = project_spatial_onto_factor
    plot_spatial_continuous = ns["plot_spatial_continuous"]
    ns["plot_spatial_continuous"] = lambda ax, section, key, title, **kw: plot_spatial_continuous(
        ax, section, key, title.replace(f"projected Factor {factor}", factor), **kw)
    section = ns["load_xenium_factor_section"]()
    section = ns["attach_xenium_imputed_msi"](section)
    if R.get("xenium_dopamine_map") is not None:    # panel e from the OT-transferred dopamine (header)
        X = section.X.toarray() if sp.issparse(section.X) else np.asarray(section.X, dtype=float)
        Z = (X - X.mean(0)) / X.std(0).clip(1e-6)
        genes = pd.Index(section.var_names.astype(str)).str.upper()
        beta = R["xenium_dopamine_map"]
        shared = genes.isin(beta.index)
        section.obs["imputed_dopamine"] = Z[:, shared] @ beta.reindex(genes[shared]).to_numpy()
        print(f"[pls] Xenium dopamine: ridge map over {int(shared.sum())}/{len(beta)} genes "
              f"(CV r in nuclei {R['xenium_dopamine_map_cv_r']:.2f})", flush=True)
    section.obs["dens_oligo"] = ns["local_category_density"](section, "Oligodendrocyte_Subclass")
    moran, _ = ns["cah_xenium_spatial_crosscorrelation"](
        section, output_file=p["figures"] / "cah_xenium_bivariate_moran.csv")
    D.fig7_board(ns, section, factor, moran, p["figures"] / "sea_ad_cah_pls")


def stage_validate(run, anchor, msi_weights="ot", ot_reg=0.01, source_batches="none"):
    """Each source section held out in turn (its MSI never used): U from the other two sections'
    MSI-SVD, then the held-out section's metabolite weights from its measured MSI vs (i) copied from
    the other two sections, (ii) ported (predictor fitted on them), (iii) OT-recovered."""
    p = paths(run, anchor, msi_weights, ot_reg, source_batches)
    S = load_source(p["bundle"])
    da = S["metabolites"].get_loc("msi:Dopamine")
    rows = []
    for held in np.unique(S["sections"]):
        tr, te = S["sections"] != held, S["sections"] == held
        b_tr = S["sections"][tr] if source_batches == "section" else None
        Zt, Mt = SC.standardize(S["Z"][te]), S["msi"][te]
        Zs, Ms = S["Z"][tr], S["msi"][tr]
        if b_tr is not None:
            Zs, Ms = centre_within(Zs, b_tr), centre_within(Ms, b_tr)
        Zs = SC.standardize(Zs)
        Cs = SC.corr_matrix(Zs, Ms, P.column_stats(Ms))
        U = np.linalg.svd(Cs / np.linalg.norm(Cs), full_matrices=False)[0][:, :K].copy()
        U *= np.where(np.array([SC.direction_weights(Cs, Zs, U[:, k])[da] for k in range(K)]) < 0, -1, 1)
        w = lambda C, Z: np.column_stack([SC.direction_weights(C, Z, U[:, k]) for k in range(K)])
        W_meas = w(SC.corr_matrix(Zt, Mt, P.column_stats(Mt)), Zt)
        k_da = int(np.argmax(np.abs(w(Cs, Zs)[da])))
        groups = np.random.default_rng(0).integers(0, 5, te.sum())
        pred = SC.fit_msi_predictor(Zs, Ms, S["sections"][tr], method="simpls")
        methods = {"copy (other sections, measured)": w(Cs, Zs),
                   "ported": SC.ported_msi_weights(pred, Zt, U)[0],
                   "OT + target decoder": SC.ot_msi_weights(
                       S["Z"][tr] if b_tr is not None else Zs, S["msi"][tr] if b_tr is not None else Ms,
                       Zt, U, groups, reg=ot_reg, batches=b_tr)[0]}
        for m, Wm in methods.items():
            for k in range(K):
                rows.append(dict(held_out=held, method=m, component=NAME.format(k + 1), is_da=k == k_da,
                                 r_with_measured=np.corrcoef(Wm[:, k], W_meas[:, k])[0, 1],
                                 dopamine_rank=int((Wm[:, k] > Wm[da, k]).sum()) + 1,
                                 dopamine_rank_measured=int((W_meas[:, k] > W_meas[da, k]).sum()) + 1))
        print(f"[validate] held out {held}: dopamine component C{k_da + 1}", flush=True)
    V = pd.DataFrame(rows)
    p["pls"].mkdir(parents=True, exist_ok=True)
    V.to_csv(p["pls"] / "heldout_section_validation.csv", index=False)
    pd.set_option("display.width", 220)
    print(V[V["is_da"]].round(3).to_string(index=False))
    print(V.groupby(["method", "component"])["r_with_measured"].mean().unstack().round(3).to_string())


def main():
    args = parse_args()
    {"components": stage_components, "figure": stage_figure, "validate": stage_validate}[args.stage](
        args.run, args.anchor, args.msi_weights, args.ot_reg, args.source_batches)


if __name__ == "__main__":
    sys.exit(main())

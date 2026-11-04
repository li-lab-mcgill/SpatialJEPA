try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Fig. 6: per-nucleus TF-motif accessibility (chromVAR-style) vs the shared PLS components.
#
# The peak-level route (C1 peak weights -> cCRE GSEA -> leading-edge motif test) works on 66,520
# sparse peaks (median detected in 2% of nuclei). Here each nucleus instead gets one deviation
# z-score per motif, chromVAR's recipe:
#
#   scan        MOODS / JASPAR2024 CORE vertebrates, flat background; --pvalue 5e-5 (chromVAR /
#               motifmatchr default) by default. At 5e-4 (gc_matched_motif_enrichment.py's threshold, the
#               median motif hits half the peaks) its cached cCRE-pool scan is reused
#   annotation  motif hits in all peaks, and restricted to the CATlas D2-MSN cCRE peaks, to the
#               peaks outside them, and to the MXD cCRE peaks (the a priori cCRE route)
#   deviation   (observed - expected) / expected counts per nucleus, expected = depth x peak share
#   background  50 draws; each annotated peak replaced by a random peak of the same GC x mean
#               accessibility stratum (20 x 20 quantile bins); z = (dev - mean_bg) / sd_bg
#   association corr(z, component score) across nuclei, depth-partial; within the MSN RNA cluster
#               also partial on a D1-vs-D2 score; calibrated against random latent directions
#               (oriented towards MSN identity) and the ported axes of the other metabolites
#
# Stages (separate envs):
#   --stage scan   scenicplus env      motif hits for every multiome peak
#   --stage dev    nichecompass_liana  deviation z-scores, own implementation (GPU if available)
#   --stage prep        nichecompass_liana  counts + cCRE flags -> <out>/atac_counts.h5ad (for pychromvar)
#   --stage pychromvar  pychromvar          the same deviations with pychromvar (its own motif matching,
#                                           GC bias and background peaks); --engine pychromvar for assoc
#   --stage assoc  nichecompass_liana  association tables (pychromvar deviations by default;
#                                       --engine custom for the own implementation) for the dopamine component of
#                                       --components msi_intact (default; fig567 defaults, written
#                                       to <out>/msi_svd_intact/), msi (all source spots) or
#                                       dopamine (anchored C1, <out>/)
#
# Env: $CONDA_ENVS/{scenicplus,nichecompass_liana,pychromvar}/bin/python

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs")
SHARED = OUT / "rna_msi_runs" / "shared_components"
CACHED_SCAN = SHARED / "manuscript_layout_full" / "motifs_gc" / "peak_motif_hits.pkl"
H5MU = OUT / "spatialjepa_projection" / "V11L12-109_B1" / "multiome_target_rna_atac_msi.h5mu"   # as P.load_target
N_BG, N_STRATA, N_RANDOM = 50, 20, 300
COMPONENT_DIRS = {"msi_intact": "msi_svd_intact", "msi": "msi_svd", "dopamine": None}
D1_MARKERS, D2_MARKERS = ["Drd1", "Tac1", "Pdyn"], ["Drd2", "Penk", "Adora2a"]


def parse_args():
    p = argparse.ArgumentParser(description="chromVAR-style motif deviations vs PLS components (multiome).")
    p.add_argument("--stage", choices=("scan", "dev", "prep", "pychromvar", "assoc"), required=True)
    p.add_argument("--engine", choices=("custom", "pychromvar"), default="pychromvar",
                   help="which deviations assoc reads (default pychromvar; scan/dev are the custom engine, "
                        "prep/pychromvar the pychromvar one)")
    p.add_argument("--out-dir", type=Path, default=SHARED / "chromvar_multiome")
    p.add_argument("--processes", type=int, default=32)
    p.add_argument("--pvalue", type=float, default=5e-5, help="MOODS motif-match p-value")
    p.add_argument("--components", choices=tuple(COMPONENT_DIRS), default="msi_intact",
                   help="assoc: which shared-components analysis supplies the dopamine component")
    p.add_argument("--run", default="fixed s3",
                   help="assoc: model run (msi_components_common.RUNS); other runs read and write runs/<run>/")
    args = p.parse_args()
    pychromvar = args.stage in ("prep", "pychromvar") or (args.stage == "assoc" and args.engine == "pychromvar")
    args.out_dir = args.out_dir / (f"p{args.pvalue:g}" + ("_pychromvar" if pychromvar else ""))
    return args


def atac_peaks():
    import h5py

    with h5py.File(H5MU, "r") as f:
        var = f["mod/atac/var"]
        return pd.Index(var[var.attrs.get("_index", "_index")][:].astype(str))


# ----------------------------------------------------------------------------- scan
def stage_scan(args):
    import gc_matched_motif_enrichment as G
    from motif_enrichment_worker import extract_sequences, fetch_jaspar_motifs, motif_to_log_odds
    import MOODS.tools as mt

    motifs = fetch_jaspar_motifs("CORE", "vertebrates")
    bg_flat = mt.flat_bg(4)
    matrices, thresholds, names = [], [], []
    for motif in motifs:
        fwd, rc = motif_to_log_odds(motif, bg_flat, G.PSEUDOCOUNT)
        thr = mt.threshold_from_p(fwd, bg_flat, args.pvalue)
        matrices += [fwd, rc]
        thresholds += [thr, thr]
        names.append((motif.name, motif.matrix_id))
    counts = pd.Series([n for n, _ in names]).value_counts()
    names = [n if counts[n] == 1 else f"{n} ({i})" for n, i in names]

    peaks = atac_peaks()
    cache = pd.read_pickle(CACHED_SCAN) if args.pvalue == G.PVALUE_THRESHOLD else pd.DataFrame(columns=range(len(names)))
    if cache.shape[1] != len(names):
        raise ValueError("cached scan has a different motif set")
    todo = peaks.difference(cache.index)
    print(f"[scan] {len(peaks):,} peaks: {len(peaks) - len(todo):,} cached, scanning {len(todo):,} "
          f"with {len(names)} motifs on {args.processes} processes", flush=True)
    seqs = extract_sequences(G.peaks_bed(todo), str(G.FASTA))
    new = G.scan(seqs, matrices, thresholds, args.processes)
    hits = pd.concat([cache.loc[cache.index.intersection(peaks)], new])
    hits = hits.reindex(peaks).astype(float).fillna(0).astype(bool)
    hits.columns = names
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hits.to_pickle(args.out_dir / "peak_motif_hits_all.pkl")
    print(f"[scan] wrote {hits.shape} (median {hits.mean().median():.2f} of peaks hit per motif)", flush=True)


# ------------------------------------------------------------------------ deviations
def stage_dev(args):
    import h5py
    import torch
    import msi_components_common as P

    hits = pd.read_pickle(args.out_dir / "peak_motif_hits_all.pkl")
    peaks = hits.index
    with h5py.File(H5MU, "r") as f:
        X = P.read_csr(f["mod/atac/layers/counts"]).astype(np.float32)
    ccre = P.ccre_membership(peaks)
    gc = pd.read_csv(SHARED / "manuscript_layout_full" / "peak_gc.csv", index_col=0)["gc"].reindex(peaks).to_numpy()
    gc = np.where(np.isfinite(gc), gc, np.nanmedian(gc))

    H = hits.to_numpy(np.float32)
    variants = {"all": np.ones(len(peaks), bool), "D2MSN_cCRE": ccre["D2MSN"],
                "outside_D2MSN_cCRE": ~ccre["D2MSN"], "MXD_cCRE": ccre["MXD"]}
    A = np.hstack([H * m[:, None] for m in variants.values()])
    cols = [f"{v}|{n}" for v in variants for n in hits.columns]
    keep = A.sum(0) >= 50                                         # >= 50 annotated peaks
    A, cols = A[:, keep], [c for c, k in zip(cols, keep) if k]
    print(f"[dev] {X.shape[0]:,} nuclei x {len(peaks):,} peaks; {len(cols):,} annotations", flush=True)

    depth = np.asarray(X.sum(1)).ravel()
    share = np.asarray(X.sum(0)).ravel() / depth.sum()
    acc = np.log1p(np.asarray(X.mean(0)).ravel())
    q = lambda v: np.searchsorted(np.quantile(v, np.linspace(0, 1, N_STRATA + 1)[1:-1]), v, side="right")
    strata = q(gc) * N_STRATA + q(acc)
    rng = np.random.default_rng(0)
    order = np.argsort(strata, kind="stable")
    starts = np.searchsorted(strata[order], np.arange(N_STRATA ** 2))
    ends = np.searchsorted(strata[order], np.arange(N_STRATA ** 2), side="right")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.sparse_csr_tensor(torch.from_numpy(X.indptr.astype(np.int64)), torch.from_numpy(X.indices.astype(np.int64)),
                                 torch.from_numpy(X.data), size=X.shape).to(dev)
    At = torch.from_numpy(A).to(dev)
    dt = torch.from_numpy(depth.astype(np.float32)).to(dev)[:, None]
    st = torch.from_numpy(share.astype(np.float32)).to(dev)

    def deviation(Ann):
        exp = dt * (st @ Ann)[None, :]
        return (Xt @ Ann - exp) / exp

    obs = deviation(At)
    mean = torch.zeros_like(obs); m2 = torch.zeros_like(obs)
    for i in range(N_BG):
        r = rng.random(len(peaks))
        bg = order[starts[strata] + np.minimum((r * (ends[strata] - starts[strata])).astype(int),
                                               ends[strata] - starts[strata] - 1)]
        Abg = torch.zeros_like(At).index_add_(0, torch.from_numpy(bg).to(dev), At)
        d = deviation(Abg)
        delta = d - mean
        mean += delta / (i + 1)
        m2 += delta * (d - mean)
        if (i + 1) % 10 == 0:
            print(f"[dev] background {i + 1}/{N_BG}", flush=True)
    z = ((obs - mean) / torch.sqrt(m2 / (N_BG - 1)).clamp_min(1e-8)).cpu().numpy().astype(np.float32)
    np.save(args.out_dir / "deviation_z.npy", z)
    pd.Series(cols).to_csv(args.out_dir / "deviation_columns.csv", index=False, header=["annotation"])
    pd.Series(depth, name="depth").to_csv(args.out_dir / "nucleus_depth.csv", index=False)
    print(f"[dev] wrote z {z.shape}", flush=True)


# ------------------------------------------------------------------------ pychromvar
FASTA = Path.home() / ".local/share/genomes/mm10/mm10.fa"


def stage_prep(args):
    import anndata as ad
    import h5py
    import msi_components_common as P

    peaks = atac_peaks()
    with h5py.File(H5MU, "r") as f:
        X = P.read_csr(f["mod/atac/layers/counts"]).astype(np.float32)
        cells = pd.Index(P.read_index(f["mod/atac/obs"]))
    ccre = P.ccre_membership(peaks)
    sizes = pd.read_csv(str(FASTA) + ".fai", sep="\t", header=None, index_col=0)[1]
    loc = peaks.str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    ok = loc["chrom"].isin(sizes.index).to_numpy()
    ok[ok] = loc.loc[ok, "end"].astype(int).to_numpy() <= sizes.reindex(loc.loc[ok, "chrom"]).to_numpy()
    var = pd.DataFrame({"D2MSN": ccre["D2MSN"], "MXD": ccre["MXD"]}, index=peaks)
    data = ad.AnnData(X=X[:, ok], obs=pd.DataFrame(index=cells), var=var[ok])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(args.out_dir / "atac_counts.h5ad")
    print(f"[prep] {data.shape} ({(~ok).sum()} peaks off the FASTA dropped)", flush=True)


_MOTIFS = None


def _jaspar():
    from pyjaspar import jaspardb

    return jaspardb(release="JASPAR2024").fetch_motifs(collection="CORE", tax_group=["vertebrates"])


def _match_shard(job):
    """pychromvar.match_motif on a shard of peak sequences (it scans serially)."""
    global _MOTIFS
    import anndata as ad
    import pychromvar as pc

    seqs, p_value = job
    if _MOTIFS is None:
        _MOTIFS = _jaspar()
    sub = ad.AnnData(X=np.zeros((1, len(seqs)), np.float32))
    sub.uns["peak_seq"] = list(seqs)
    pc.match_motif(sub, motifs=_MOTIFS, p_value=p_value)
    return sub.varm["motif_match"], sub.uns["motif_name"]


def stage_pychromvar(args):
    import anndata as ad
    from multiprocessing import Pool
    import pychromvar as pc

    data = ad.read_h5ad(args.out_dir / "atac_counts.h5ad")
    pc.add_peak_seq(data, genome_file=str(FASTA), delimiter=":|-")
    pc.add_gc_bias(data)
    pc.get_bg_peaks(data, niterations=N_BG)
    seqs = data.uns["peak_seq"]
    shards = np.array_split(np.arange(len(seqs)), args.processes * 4)
    with Pool(args.processes) as pool:
        res = pool.map(_match_shard, [([seqs[i] for i in sh], args.pvalue) for sh in shards])
    M = np.vstack([m for m, _ in res]).astype(np.float32)
    data.uns["motif_name"] = list(res[0][1])
    data.varm["motif_match"] = M
    ids = [str(n).split(".", 2) for n in data.uns["motif_name"]]          # "MA0004.1.Arnt"
    names = pd.Series([n for _, _, n in ids])
    dup = names.duplicated(keep=False).to_numpy()
    names[dup] = [f"{n} ({a}.{b})" for (a, b, n) in np.asarray(ids, dtype=object)[dup]]
    print(f"[pychromvar] {data.shape}; {M.shape[1]} motifs, median {np.median(M.mean(0)):.2f} of peaks hit", flush=True)
    variants = {"all": np.ones(data.n_vars, bool), "D2MSN_cCRE": data.var["D2MSN"].to_numpy(),
                "outside_D2MSN_cCRE": ~data.var["D2MSN"].to_numpy(), "MXD_cCRE": data.var["MXD"].to_numpy()}
    zs, cols = [], []
    for v, mask in variants.items():
        Mv = M * mask[:, None]
        keep = Mv.sum(0) >= 50
        data.varm["motif_match"] = Mv[:, keep]
        data.uns["motif_name"] = list(names[keep])
        dev = pc.compute_deviations(data, n_jobs=1)     # n_jobs>1 pickles the dense count matrix per draw
        zs.append(np.asarray(dev.X, dtype=np.float32))
        cols += [f"{v}|{n}" for n in names[keep]]
        print(f"[pychromvar] {v}: {keep.sum()} annotations", flush=True)
    z = np.hstack(zs)
    np.save(args.out_dir / "deviation_z.npy", z)
    pd.Series(cols).to_csv(args.out_dir / "deviation_columns.csv", index=False, header=["annotation"])
    pd.Series(np.asarray(data.X.sum(1)).ravel(), name="depth").to_csv(args.out_dir / "nucleus_depth.csv", index=False)
    print(f"[pychromvar] wrote z {z.shape}", flush=True)


# ----------------------------------------------------------------------- association
def stage_assoc(args):
    import pickle

    import anndata as ad
    import joblib
    import scipy.sparse as sp
    from scipy.stats import norm
    from statsmodels.stats.multitest import multipletests
    import msi_components_common as P
    import shared_msi_components as SC

    tag = P.RUNS[args.run]
    z = np.load(args.out_dir / "deviation_z.npy")
    cols = pd.read_csv(args.out_dir / "deviation_columns.csv")["annotation"].tolist()
    depth = pd.read_csv(args.out_dir / "nucleus_depth.csv")["depth"].to_numpy()
    T = P.load_target("multiome")
    sub = COMPONENT_DIRS[args.components]
    comp_dir, res_dir = (SHARED / sub, args.out_dir / sub) if sub else (SHARED, args.out_dir)
    if args.run != "fixed s3":
        comp_dir, res_dir = (d / "runs" / args.run.replace(" ", "_") for d in (comp_dir, res_dir))
    res_dir.mkdir(parents=True, exist_ok=True)
    R = pickle.load(open(comp_dir / "multiome" / "results.pkl", "rb"))
    c = R["c_da"]
    print(f"[assoc] components {comp_dir / 'multiome'}: dopamine component {c}", flush=True)
    Z0 = SC.standardize(SC.target_latents(tag, "multiome", T["cells"])["student"])
    n = len(Z0)
    ld = np.log(depth)

    # axes: components, MSN identity, random latent directions, ported metabolite axes
    joint = ad.read_h5ad(P.MODELS / (P.BASE + tag) / "joint_adata.h5ad")
    if R.get("source_spots") == "intact":                     # metabolite axes from the same spots
        joint = joint[joint.obs["lesion"].eq("intact").to_numpy()].copy()
    feats = list(joblib.load(P.MODELS / (P.BASE + tag) / "student_pls_decoder.joblib")["target_features"])
    Xm = joint[:, feats].X
    msi = np.log1p(Xm.toarray() if sp.issparse(Xm) else np.asarray(Xm, float))
    Zs = SC.standardize(joint.obsm["student_X_emb_st"])
    msi_z = (msi - msi.mean(0)) / msi.std(0, ddof=1)
    msi_z[:, ~np.isfinite(msi_z).all(0)] = 0
    Cs = Zs.T @ msi_z / (len(Zs) - 1)
    Bm = np.linalg.solve(Zs.T @ Zs / (len(Zs) - 1), Cs)
    R_m = np.sqrt(np.einsum("dm,dm->m", Cs, Bm))
    msn = P.module_scores(T["rna"], T["genes"])["MSN"].to_numpy()
    axes = {c: Z0 @ R["U"][:, k] for k, c in enumerate(R["comps"])}
    axes["MSN identity"] = msn
    rng = np.random.default_rng(1)
    rand = []
    for _ in range(N_RANDOM):
        u = rng.standard_normal(Z0.shape[1])
        s = Z0 @ u
        rand.append(s if np.corrcoef(s, msn)[0, 1] >= 0 else -s)
    good = np.where(R_m > 0.5)[0]
    met = {feats[m]: Z0 @ Bm[:, m] for m in good}
    for name, s in met.items():
        if np.corrcoef(s, msn)[0, 1] < 0:
            met[name] = -s

    # within the MSN cluster: depth + D1-vs-D2 score as covariates
    cl = pd.Series(T["clusters"]).astype(str).to_numpy()
    cm = pd.Series(msn).groupby(cl).mean()
    msn_cl = cm.index[cm > 0.5 * msn.std() + msn.mean()].tolist()
    inm = np.isin(cl, msn_cl)
    genes = pd.Index(T["genes"])
    expr = lambda gs: np.asarray(T["rna"][:, [genes.get_loc(g) for g in gs if g in genes]].mean(1)).ravel()
    d1d2 = expr(D1_MARKERS) - expr(D2_MARKERS)
    print(f"[assoc] MSN cluster(s) {msn_cl}: {inm.sum():,} nuclei", flush=True)

    def resid(M, covs):
        Xc = np.column_stack([np.ones(len(covs[0]))] + list(covs))
        return M - Xc @ np.linalg.lstsq(Xc, M, rcond=None)[0]

    def standardize_cols(M):
        M = M - M.mean(0)
        return M / np.clip(M.std(0, ddof=1), 1e-12, None)

    zo = standardize_cols(resid(z, [ld]))
    zw = standardize_cols(resid(z[inm], [ld[inm], d1d2[inm]]))

    def corr_with(S, within=False):
        S = np.column_stack(S)
        if within:
            Sr = standardize_cols(resid(S[inm], [ld[inm], d1d2[inm]]))
            return zw.T @ Sr / (inm.sum() - 1)
        Sr = standardize_cols(resid(S, [ld]))
        return zo.T @ Sr / (n - 1)

    out = pd.DataFrame(index=pd.Index(cols, name="annotation"))
    for within, tag_ in ((False, "all"), (True, "MSN")):
        A = corr_with(list(axes.values()), within)
        for j, name in enumerate(axes):
            out[f"r_{tag_}|{name}"] = A[:, j]
        Rr = corr_with(rand, within)
        rc = out[f"r_{tag_}|{c}"].to_numpy()
        out[f"pct_random_{tag_}|{c}"] = (Rr < rc[:, None]).mean(1)
        Mr = corr_with(list(met.values()), within)
        out[f"rank_among_metabolites_{tag_}|{c}"] = (Mr > rc[:, None]).sum(1) + 1
        pd.DataFrame(Mr, index=out.index, columns=list(met)).to_csv(res_dir / f"metabolite_axes_x_motifs_{tag_}.csv.gz")
        m = inm.sum() if within else n
        zf = np.arctanh(np.clip(rc, -0.999999, 0.999999)) * np.sqrt(m - 3 - (2 if within else 1))
        out[f"p_{tag_}|{c}"] = 2 * norm.sf(np.abs(zf))
        out[f"padj_{tag_}|{c}"] = multipletests(out[f"p_{tag_}|{c}"], method="fdr_bh")[1]
    out["variant"] = [c.split("|", 1)[0] for c in cols]
    out["motif"] = [c.split("|", 1)[1] for c in cols]
    out.to_csv(res_dir / "motif_deviation_associations.csv.gz")

    pd.set_option("display.width", 250)
    show = [f"r_all|{c}", f"pct_random_all|{c}", "r_all|MSN identity", f"r_MSN|{c}", "r_MSN|MSN identity",
            f"padj_MSN|{c}", f"pct_random_MSN|{c}", f"rank_among_metabolites_MSN|{c}"]
    for v in ("all", "D2MSN_cCRE", "outside_D2MSN_cCRE"):
        sub = out[out["variant"].eq(v)].set_index("motif")
        print(f"\n[{v}] top 15 motifs by r with {c} across nuclei")
        print(sub.nlargest(15, f"r_all|{c}")[show].round(3).to_string())
        print(f"[{v}] top 15 motifs by r with {c} within MSNs")
        print(sub.nlargest(15, f"r_MSN|{c}")[show].round(3).to_string())
        print(f"[{v}] bottom 10 motifs within MSNs")
        print(sub.nsmallest(10, f"r_MSN|{c}")[show].round(3).to_string())


def main():
    args = parse_args()
    {"scan": stage_scan, "dev": stage_dev, "prep": stage_prep, "pychromvar": stage_pychromvar,
     "assoc": stage_assoc}[args.stage](args)


if __name__ == "__main__":
    main()

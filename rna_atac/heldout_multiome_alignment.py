"""Does the held-out 10x Multiome land on the P22 tissue, and does that recover the teacher's clusters?

The Multiome has no coordinates, so spatial information can only be read through P22.
For each model, Multiome nuclei are coupled to P22 spots in that model's own embedding
(models trained on P22, applied zero-shot to the Multiome):
  knn  spot-side 15-NN vote (back-projection) / nucleus-side 15-NN rows (placement)
  ot   balanced entropic OT, Euclidean cost / mean, eps 0.01 (the manuscript's OT step)
  uot  unbalanced entropic OT (KL marginals, rho 1.0, ~80% of mass kept): composition can differ
per view: rna (RNA-view embeddings), atac (ATAC-view), joint (both views, each scaled to
unit mean-square norm on P22).

Readouts, against teacher clusters = KMeans(K) of the spatial teacher's joint P22 embedding:
  backproject  Multiome labels carried to P22 spots through the coupling (argmax of
               transported mass): (a) the model's own KMeans(K) clusters of the Multiome,
               (b) CellTypist cell types (model-free). ARI/NMI vs teacher clusters, and the
               spatial kappa of the back-projected map on the P22 6-NN grid.
  transfer     teacher clusters carried to the Multiome, ARI vs the Multiome's own
               KMeans(K) and NMI vs CellTypist.
  in_sample    KMeans(K) of the model's own P22 embedding vs teacher clusters (ceiling).
  placement    known pairing: each nucleus is placed on P22 twice, through its RNA view and
               through its ATAC view; W1 (P22 coordinate units, pitch 23) between the two
               placements, vs a random nucleus's ATAC placement and vs a same-CellTypist
               nucleus's ATAC placement. Placement closer than the same-type partner means
               the embedding locates nuclei more finely than their cell type.

Models: teacher_nograph (teacher weights, identity graph), student (distilled, graph-free),
student_in_teacher (student Multiome vs teacher P22), nonspatial (graph-free, no teacher),
multivi (P22-trained), pca (untrained RNA PCA / ATAC LSI fitted on P22).

Run in the nichecompass_liana env after encode_heldout_controls.py.
--clustering leiden: Leiden (scanpy, 150-NN umap graph, seed 0, as for main-text Fig. 4a) in place of
KMeans, with the resolution bisected to give exactly K clusters; teacher reference cached once;
outputs in heldout_alignment_leiden/ (W1 placement is clustering-independent and is skipped).
Outputs in $SPATIALJEPA_OUT/MultiGATE/rna_atac_analysis/20260505_190037/heldout_alignment/.
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import argparse
import os
from pathlib import Path

import h5py
import anndata as ad
import numpy as np
import ot
import pandas as pd
import scipy.sparse as sp
import torch
from anndata.io import read_elem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

BASE = Path(os.getenv("SPATIALJEPA_HOME", os.environ['SPATIALJEPA_HOME']))
DATA = BASE / "data" / "aligned_data"
RUN = BASE / "outputs/MultiGATE/rna_atac_analysis/20260505_190037"
SAVED = RUN / "metrics_bootstrap" / "metric_inputs"
MULTIVI_FULL = RUN / "metrics_bootstrap_v3" / "multivi_zero_shot_full_target_latents.npz"
OUT = RUN / "heldout_alignment"
KS = (5, 10, 20)
CLUSTERING = "kmeans"
LEIDEN_NEIGHBORS = 150  # main-text Fig. 4a target Leiden graph


def leiden_k(X, ks, seed=0):
    """Leiden on a 150-NN umap graph (Fig. 4a settings); resolution bisected to hit each K exactly."""
    import scanpy as sc
    a = ad.AnnData(np.asarray(X, dtype=np.float32))
    sc.pp.neighbors(a, n_neighbors=LEIDEN_NEIGHBORS, use_rep="X", knn=True, method="umap",
                    metric="euclidean", random_state=seed)
    out = {}
    for K in ks:
        lo, hi, best = 1e-3, 5.0, None
        for _ in range(30):
            r = np.sqrt(lo * hi)
            sc.tl.leiden(a, resolution=r, random_state=seed, key_added="l")
            lab = a.obs["l"].astype(str).to_numpy()
            n = len(np.unique(lab))
            if best is None or abs(n - K) < abs(best[0] - K):
                best = (n, lab, r)
            if n == K:
                break
            lo, hi = (r, hi) if n < K else (lo, r)
        if best[0] != K:
            print(f"   leiden: K={K} not reached exactly, using {best[0]} clusters (res {best[2]:.4f})", flush=True)
        out[K] = best[1]
    return out


def cluster_k(X, ks):
    if CLUSTERING == "leiden":
        return leiden_k(X, ks)
    return {K: KMeans(K, n_init=10, random_state=0).fit_predict(X).astype(str) for K in ks}
K_NN = 15
EPS, RHO = 0.01, 1.0
EPS_SCHEDULE = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)
TOP_R = 32
N_PLACE = 4000
VIEWS = ("rna", "atac", "joint")


# ---------------------------------------------------------------- data
def saved(domain, view, key):
    a = ad.read_h5ad(SAVED / f"{domain}__{view}.h5ad")
    return np.asarray(a.obs_names), np.asarray(a.obsm[key], dtype=np.float32), a.obs


def h5_obs_column(path, col):
    with h5py.File(path) as f:
        return np.asarray(read_elem(f["obs"][col])).astype(str)


def h5_obsm(path, key):
    with h5py.File(path) as f:
        return np.asarray(read_elem(f["obsm"][key]))


def pca_baseline(p22_obs, mo_obs, n_comps=30):
    """Untrained baselines fitted on P22 only: PCA of SCT residuals, LSI of binarised peaks."""
    def load(name, layer=None):
        a = ad.read_h5ad(DATA / f"{name}_aligned_with_latents.h5ad")
        X = a.layers[layer] if layer else a.X
        return pd.Index(a.obs_names), X
    obs_s, Xs = load("source_rna")
    obs_t, Xt = load("target_rna")
    Xs, Xt = (np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32) for X in (Xs, Xt))
    pca = PCA(n_comps, random_state=0).fit(Xs)
    rna = {"p22": pca.transform(Xs)[obs_s.get_indexer(p22_obs)], "mo": pca.transform(Xt)[obs_t.get_indexer(mo_obs)]}

    obs_s, Cs = load("source_atac", "counts")
    obs_t, Ct = load("target_atac", "counts")
    Bs, Bt = (sp.csr_matrix(C).astype(bool).astype(np.float32) for C in (Cs, Ct))
    idf = np.log1p(Bs.shape[0] / (1.0 + np.asarray(Bs.sum(0)).ravel()))

    def tfidf(B):
        tf = sp.diags(1.0 / np.maximum(np.asarray(B.sum(1)).ravel(), 1)) @ B
        X = (tf @ sp.diags(idf)).tocsr()
        X.data = np.log1p(1e4 * X.data)
        return X
    svd = TruncatedSVD(n_comps + 1, random_state=0).fit(tfidf(Bs))
    atac = {"p22": svd.transform(tfidf(Bs))[:, 1:][obs_s.get_indexer(p22_obs)],
            "mo": svd.transform(tfidf(Bt))[:, 1:][obs_t.get_indexer(mo_obs)]}
    return rna, atac


def load_all():
    p22_obs, T_rna, p22_meta = saved("teacher_source", "rna", "MultiGATE_full_teacher")
    _, T_atac, _ = saved("teacher_source", "atac", "MultiGATE_full_teacher")
    mo_obs, S_mo_rna, mo_meta = saved("target", "rna", "MultiGATE")
    _, S_mo_atac, _ = saved("target", "atac", "MultiGATE")
    src = DATA / "source_rna_aligned_with_latents.h5ad"
    with h5py.File(src) as f:
        index_key = f["obs"].attrs.get("_index", "_index")
        assert np.array_equal(np.asarray(read_elem(f["obs"][index_key])).astype(str), p22_obs)
    coords = h5_obsm(src, "spatial").astype(np.float64)
    labels = dict(RNA_clusters=h5_obs_column(src, "RNA_clusters"), ATAC_clusters=h5_obs_column(src, "ATAC_clusters"))
    celltypist = mo_meta["scib_shared_label"].astype(str).to_numpy()

    def views(p22_rna, p22_atac, mo_rna, mo_atac):
        return {"p22": {"rna": p22_rna, "atac": p22_atac}, "mo": {"rna": mo_rna, "atac": mo_atac}}

    models = {"student": views(saved("source", "rna", "MultiGATE_source_aligned")[1],
                               saved("source", "atac", "MultiGATE_source_aligned")[1], S_mo_rna, S_mo_atac),
              "student_in_teacher": views(T_rna, T_atac, S_mo_rna, S_mo_atac)}

    enc_path = RUN / "heldout_alignment" / "multigate_encodings.npz"  # shared by kmeans and leiden runs
    if enc_path.exists():
        enc = np.load(enc_path, allow_pickle=True)
        ps = pd.Index(enc["source_obs_names"].astype(str)).get_indexer(p22_obs)
        pt = pd.Index(enc["target_obs_names"].astype(str)).get_indexer(mo_obs)
        assert (ps >= 0).all() and (pt >= 0).all()
        for m in ("teacher_nograph", "nonspatial"):
            models[m] = views(enc[f"{m}__source__rna"][ps], enc[f"{m}__source__atac"][ps],
                              enc[f"{m}__target__rna"][pt], enc[f"{m}__target__atac"][pt])
    else:
        print("multigate_encodings.npz missing: run encode_heldout_controls.py; skipping its models")
    mv = np.load(MULTIVI_FULL, allow_pickle=True)
    pm = pd.Index(mv["obs_names"].astype(str)).get_indexer(mo_obs)
    assert (pm >= 0).all()
    models["multivi"] = views(saved("source_multivi", "rna", "X_multivi")[1], saved("source_multivi", "atac", "X_multivi")[1],
                              mv["expression"][pm], mv["accessibility"][pm])
    rna, atac = pca_baseline(p22_obs, mo_obs)
    models["pca"] = views(rna["p22"], atac["p22"], rna["mo"], atac["mo"])

    for m in models.values():  # joint view: both views, each with unit mean-square norm on P22
        s = {v: np.sqrt((m["p22"][v] ** 2).sum(1).mean()) for v in ("rna", "atac")}
        for d in ("p22", "mo"):
            m[d] = {v: np.asarray(x, dtype=np.float32) for v, x in m[d].items()}
            m[d]["joint"] = np.hstack([m[d]["rna"] / s["rna"], m[d]["atac"] / s["atac"]]).astype(np.float32)
    teacher_joint = models["student_in_teacher"]["p22"]["joint"]
    return dict(p22_obs=p22_obs, mo_obs=mo_obs, coords=coords, p22_labels=labels,
                celltypist=celltypist, models=models, teacher_joint=teacher_joint)


# ---------------------------------------------------------------- couplings
def sinkhorn_log(C, eps, rho=None, n_iter=300, tol=1e-4):
    """Log-domain entropic OT between uniform marginals with eps-scaling; rho = KL marginal weight."""
    n, m = C.shape
    loga, logb = -np.log(n), -np.log(m)
    f = torch.zeros(n, device=C.device)
    g = torch.zeros(m, device=C.device)
    for e in [x for x in EPS_SCHEDULE if x >= eps]:
        lam = 1.0 if rho is None else rho / (rho + e)
        for it in range(n_iter):
            f = -lam * e * torch.logsumexp((g[None, :] - C) / e + logb, dim=1)
            g = -lam * e * torch.logsumexp((f[:, None] - C) / e + loga, dim=0)
            if rho is None and it % 25 == 24:
                row = torch.logsumexp((f[:, None] + g[None, :] - C) / e + logb, dim=1)
                if (row.exp() - 1).abs().max() < tol:
                    break
    return (f[:, None] + g[None, :] - C) / eps + loga + logb  # log plan


def onehot(labels):
    codes, uniq = pd.factorize(labels)
    H = np.zeros((len(codes), len(uniq)), dtype=np.float32)
    H[np.arange(len(codes)), codes] = 1
    return H, np.asarray(uniq)


def readouts_from_plan(P, mo_labels, p22_labels):
    """P: n_mo x n_p22 plan (torch). argmax of transported label mass on either side."""
    back = {}
    for name, lab in mo_labels.items():
        H, uniq = onehot(lab)
        back[name] = uniq[(P.T @ torch.as_tensor(H, device=P.device)).argmax(1).cpu().numpy()]
    fwd = {}
    for name, lab in p22_labels.items():
        H, uniq = onehot(lab)
        fwd[name] = uniq[(P @ torch.as_tensor(H, device=P.device)).argmax(1).cpu().numpy()]
    return back, fwd


def vote(labels, nbrs):
    codes, uniq = pd.factorize(np.asarray(labels))
    rows = np.repeat(np.arange(len(nbrs)), nbrs.shape[1])
    counts = sp.csr_matrix((np.ones(nbrs.size), (rows, codes[nbrs].ravel())), shape=(len(nbrs), len(uniq)))
    return uniq[np.asarray(counts.argmax(1)).ravel()]


def knn_readouts(Xmo, Xp22, mo_labels, p22_labels, place_idx):
    """Spot-side vote for back-projection; nucleus-side 15-NN for transfer and placement."""
    nn_mo = NearestNeighbors(n_neighbors=K_NN).fit(Xmo)
    _, spot_nbrs = nn_mo.kneighbors(Xp22)
    back = {name: vote(lab, spot_nbrs) for name, lab in mo_labels.items()}
    nn_p = NearestNeighbors(n_neighbors=K_NN).fit(Xp22)
    _, mo_nbrs = nn_p.kneighbors(Xmo)
    fwd = {name: vote(lab, mo_nbrs) for name, lab in p22_labels.items()}
    rows = (mo_nbrs[place_idx], np.full((len(place_idx), K_NN), 1.0 / K_NN))
    return back, fwd, rows, None


def ot_readouts(Xmo, Xp22, mo_labels, p22_labels, place_idx, rho, device):
    X = torch.as_tensor(Xmo, device=device)
    Y = torch.as_tensor(Xp22, device=device)
    C = torch.cdist(X, Y)
    C = C / C.mean()
    logP = sinkhorn_log(C, EPS, rho)
    del C
    P = logP.exp()
    back, fwd = readouts_from_plan(P, mo_labels, p22_labels)
    mass = float(P.sum().cpu())
    col = P.sum(0)
    coverage = float((col > 0.1 / P.shape[1]).float().mean().cpu())  # spots receiving >=10% of uniform mass
    top = torch.topk(logP[torch.as_tensor(place_idx, device=device)], TOP_R, dim=1)
    w = torch.softmax(top.values, dim=1).cpu().numpy().astype(np.float64)
    rows = (top.indices.cpu().numpy(), w)
    del P, logP, X, Y
    torch.cuda.empty_cache()
    return back, fwd, rows, dict(mass=mass, coverage=coverage)


# ---------------------------------------------------------------- metrics
def spatial_kappa(labels, nbrs):
    """Neighbour label agreement on the P22 grid, chance-corrected by label frequencies."""
    lab = pd.factorize(np.asarray(labels))[0]
    agree = (lab[nbrs] == lab[:, None]).mean()
    p = np.bincount(lab) / len(lab)
    chance = (p ** 2).sum()
    return (agree - chance) / (1 - chance)


def w1(rows_a, rows_b, coords):
    (ia, wa), (ib, wb) = rows_a, rows_b
    out = np.empty(len(ia))
    for k in range(len(ia)):
        M = ot.dist(coords[ia[k]], coords[ib[k]], metric="euclidean")
        out[k] = ot.emd2(wa[k], wb[k], M)
    return out


def placement_rows(rows, order):
    return rows[0][order], rows[1][order]


# ---------------------------------------------------------------- summary
MODEL_ORDER = ["student", "student_in_teacher", "teacher_nograph", "nonspatial", "multivi", "pca"]


def summarize():
    """Merge per-run CSVs; print the joint-view tables and the known-pairing placement table."""
    long = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("alignment_long*.csv"))]).drop_duplicates(
        ["model", "view", "method", "readout", "labels", "reference", "metric"], keep="last")
    pfiles = [f for f in sorted(OUT.glob("placement_w1_long*.csv")) if f.stat().st_size > 1]
    place = (pd.concat([pd.read_csv(f) for f in pfiles]).drop_duplicates(["model", "method", "partner", "nucleus"], keep="last")
             if pfiles else None)
    order = [m for m in MODEL_ORDER if m in set(long.model)]
    pd.set_option("display.width", 220)

    def pick(readout, labels, reference, metric, view="joint"):
        s = long[(long.readout == readout) & (long.view == view) & (long.metric == metric)
                 & long["labels"].astype(str).str.fullmatch(labels) & long.reference.astype(str).str.fullmatch(reference)]
        return s.groupby(["model", "method"])["value"].mean()  # mean over K where several match

    ins = long[(long.readout == "in_sample")].groupby("model")["value"].mean()
    rows = []
    for (model, method), v in pick("backproject", r"own_K\d+", r"teacher_K\d+", "ari").items():
        if method == "none":
            continue
        rows.append(dict(model=model, method=method, in_sample_ari=ins.get(model), backproject_ari=v))
    t = pd.DataFrame(rows).set_index(["model", "method"])
    t["retained"] = t.backproject_ari / t.in_sample_ari
    t["celltypist_nmi"] = pick("backproject", "celltypist", r"teacher_K\d+", "nmi")
    published = long[(long.readout == "backproject") & (long.view == "joint") & (long["labels"] == "own_K10")
                     & long.reference.isin(["RNA_clusters", "ATAC_clusters"]) & (long.metric == "ari")]
    t["published_ari"] = published.groupby(["model", "method"])["value"].mean()  # vs Zhang 2023 RNA/ATAC clusters
    t["kappa_own_K10"] = pick("backproject", "own_K10", "grid", "spatial_kappa")
    t["kappa_celltypist"] = pick("backproject", "celltypist", "grid", "spatial_kappa")
    t["transfer_ari"] = pick("transfer", r"teacher_K\d+", r"own_K\d+", "ari")
    t["transfer_ct_nmi"] = pick("transfer", r"teacher_K\d+", "celltypist", "nmi")
    t = t.reindex(pd.MultiIndex.from_product([order, ["knn", "ot", "uot"]])).dropna(how="all")
    print("\n[joint view; ARI/NMI averaged over K = 5/10/20 unless noted]")
    print(t.round(3).to_string())
    ref = long[(long.readout == "reference") & (long.metric == "spatial_kappa")].set_index("labels")["value"]
    print("teacher map spatial kappa:", ref.round(3).to_dict())
    plan = long[long.readout == "plan"].pivot_table(index=["model", "view"], columns=["method", "metric"], values="value")
    print("\n[plan mass / spot coverage]\n" + plan.round(3).to_string())

    print("\n[per-view back-projection ARI of own clusters vs teacher, mean over K]")
    pv = long[(long.readout == "backproject") & long["labels"].str.startswith("own_K") & (long.metric == "ari")
              & long.reference.str.startswith("teacher_K")].groupby(["model", "method", "view"])["value"].mean().unstack("view")
    print(pv.reindex(pd.MultiIndex.from_product([order, ["knn", "ot", "uot"]])).dropna(how="all").round(3).to_string())

    t.to_csv(OUT / "alignment_summary.csv")
    if place is None:
        return
    rng = np.random.default_rng(0)
    rows = []
    for (model, method), g in place.groupby(["model", "method"]):
        w = g.pivot_table(index="nucleus", columns="partner", values="w1")
        diff = (w["same_celltype"] - w["paired"]).to_numpy()
        boot = [np.median(rng.choice(diff, len(diff))) for _ in range(1000)]
        rows.append(dict(model=model, method=method, paired=w.paired.median(), same_celltype=w.same_celltype.median(),
                         random=w.random.median(), frac_closer_than_same_type=(diff > 0).mean(),
                         median_gain=np.median(diff), gain_lo=np.percentile(boot, 2.5), gain_hi=np.percentile(boot, 97.5)))
    p = pd.DataFrame(rows).set_index(["model", "method"])
    p = p.reindex(pd.MultiIndex.from_product([order, ["knn", "ot", "uot"]])).dropna(how="all")
    print("\n[known pairing: W1 (P22 coordinate units; spot pitch 23) between a nucleus's RNA-view and ATAC-view placements]")
    print(p.round(3).to_string())
    t.to_csv(OUT / "alignment_summary.csv")
    p.to_csv(OUT / "placement_w1_summary.csv")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--clustering", choices=("kmeans", "leiden"), default="kmeans")
    ap.add_argument("--teacher_only", action="store_true", help="leiden: compute and cache the teacher reference, then exit")
    args = ap.parse_args()
    global CLUSTERING, OUT
    CLUSTERING = args.clustering
    if CLUSTERING == "leiden":
        OUT = RUN / "heldout_alignment_leiden"
    if args.summarize:
        return summarize()
    torch.set_grad_enabled(False)
    d = load_all()
    coords = d["coords"]
    _, grid_nbrs = NearestNeighbors(n_neighbors=7).fit(coords).kneighbors(coords)
    grid_nbrs = grid_nbrs[:, 1:]
    cache = OUT / "teacher_reference.npz"
    if CLUSTERING == "leiden" and cache.exists():
        z = np.load(cache, allow_pickle=True)
        teacher = {K: z[f"teacher_K{K}"].astype(str) for K in KS}
    else:
        teacher = cluster_k(d["teacher_joint"], KS)
        if CLUSTERING == "leiden":
            OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **{f"teacher_K{K}": teacher[K] for K in KS})
    if args.teacher_only:
        print("teacher reference written to", cache, flush=True)
        return
    p22_refs = {**{f"teacher_K{K}": teacher[K] for K in KS}, **d["p22_labels"]}

    rng = np.random.default_rng(0)
    n_mo = len(d["mo_obs"])
    place_idx = np.sort(rng.choice(n_mo, N_PLACE, replace=False))
    ct = d["celltypist"][place_idx]
    random_partner = rng.permutation(N_PLACE)
    same_type_partner = np.arange(N_PLACE)
    for t in np.unique(ct):
        idx = np.flatnonzero(ct == t)
        if len(idx) > 1:
            same_type_partner[idx] = idx[np.roll(rng.permutation(len(idx)), 1)]
    nulls = {"paired": np.arange(N_PLACE), "random": random_partner, "same_celltype": same_type_partner}

    rows_out, place_out, maps = [], [], {}
    for ref, lab in p22_refs.items():
        rows_out.append(dict(model="teacher", view="joint", method="none", readout="reference", labels=ref,
                             reference="grid", metric="spatial_kappa", value=spatial_kappa(lab, grid_nbrs)))
    for model, emb in d["models"].items():
        if args.models and model not in args.models:
            continue
        print(f"== {model}", flush=True)
        own = cluster_k(emb["mo"]["joint"], KS)
        ins_k = cluster_k(emb["p22"]["joint"], KS)
        for K in KS:
            ins = ins_k[K]
            rows_out.append(dict(model=model, view="joint", method="none", readout="in_sample", labels=f"own_K{K}",
                                 reference=f"teacher_K{K}", metric="ari", value=adjusted_rand_score(teacher[K], ins)))
        mo_labels = {**{f"own_K{K}": own[K] for K in KS}, "celltypist": d["celltypist"]}
        placements = {}
        for view in VIEWS:
            Xmo, Xp22 = emb["mo"][view], emb["p22"][view]
            for method in ("knn", "ot", "uot"):
                if method == "knn":
                    back, fwd, rows, info = knn_readouts(Xmo, Xp22, mo_labels, p22_refs, place_idx)
                else:
                    back, fwd, rows, info = ot_readouts(Xmo, Xp22, mo_labels, p22_refs, place_idx,
                                                        None if method == "ot" else RHO, args.device)
                tag = dict(model=model, view=view, method=method)
                if info:
                    for k, v in info.items():
                        rows_out.append(dict(**tag, readout="plan", labels="", reference="", metric=k, value=v))
                for name, lab in back.items():
                    K = int(name.split("K")[1]) if name.startswith("own") else None
                    refs = [f"teacher_K{K}"] if K else [f"teacher_K{k}" for k in KS] + list(d["p22_labels"])
                    for ref in refs + (list(d["p22_labels"]) if K == 10 else []):
                        for metric, fn in (("ari", adjusted_rand_score), ("nmi", normalized_mutual_info_score)):
                            rows_out.append(dict(**tag, readout="backproject", labels=name, reference=ref,
                                                 metric=metric, value=fn(p22_refs[ref], lab)))
                    rows_out.append(dict(**tag, readout="backproject", labels=name, reference="grid",
                                         metric="spatial_kappa", value=spatial_kappa(lab, grid_nbrs)))
                    if view == "joint" and name in ("own_K10", "celltypist"):
                        maps[f"{model}__{method}__{name}"] = np.asarray(lab).astype(str)
                for K in KS:
                    lab = fwd[f"teacher_K{K}"]
                    rows_out.append(dict(**tag, readout="transfer", labels=f"teacher_K{K}", reference=f"own_K{K}",
                                         metric="ari", value=adjusted_rand_score(own[K], lab)))
                    rows_out.append(dict(**tag, readout="transfer", labels=f"teacher_K{K}", reference="celltypist",
                                         metric="nmi", value=normalized_mutual_info_score(d["celltypist"], lab)))
                if view in ("rna", "atac"):
                    placements[(view, method)] = rows
                print(f"   {view:5s} {method:4s} done {info or ''}", flush=True)
        for method in (("knn", "ot", "uot") if CLUSTERING == "kmeans" else ()):
            r_rna, r_atac = placements[("rna", method)], placements[("atac", method)]
            for null, order in nulls.items():
                dist = w1(r_rna, placement_rows(r_atac, order), coords)
                for k, i in enumerate(place_idx):
                    place_out.append(dict(model=model, method=method, partner=null, nucleus=i,
                                          celltypist=d["celltypist"][i], w1=dist[k]))

    OUT.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(rows_out)
    tag = "" if not args.models else "_" + "_".join(args.models)
    long.to_csv(OUT / f"alignment_long{tag}.csv", index=False)
    place = pd.DataFrame(place_out, columns=["model", "method", "partner", "nucleus", "celltypist", "w1"])
    place.to_csv(OUT / f"placement_w1_long{tag}.csv", index=False)
    np.savez_compressed(OUT / f"backprojected_maps{tag}.npz", coords=coords, p22_obs=d["p22_obs"],
                        **{f"teacher_K{K}": teacher[K] for K in KS}, **maps)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()

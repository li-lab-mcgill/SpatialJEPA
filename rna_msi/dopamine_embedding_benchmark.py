try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Benchmark SMA joint RNA+MSI embeddings: SpatialMETA teacher/student/nonspatial vs Multigrate.
#
# Loads joint (per-spot) embeddings of one SMA sample and scores them with a benchmark
# modeled on the RNA-ATAC block in rna_atac/analyze_rna_atac.py:
#
#   A. Bio-conservation        -> scib silhouette_label + Leiden NMI/ARI vs
#                                 RNA_clusters / MSI_clusters.
#   B. Spatial structure       -> niche recovery, within-cluster coherence, and an
#                                 (adapted) orthogonal-target recovery, all on the joint
#                                 embedding with spatial-block CV.
#   C. Cross-modal alignment   -> 1-FOSCTTM and iLISI modality mixing between each model's
#                                 per-modality RNA-only / MSI-only latents.
#
# All latents come precomputed from disk (no model is reloaded):
#   - SpatialMETA teacher/student/nonspatial: joint_adata.obsm["{teacher,student,nonspatial}_X_emb{,_st,_sm}"],
#     written by train_rna_msi.py.
#   - Multigrate: ${SPATIALJEPA_OUT}/multigrate_mouse_sma/<sample>/multigrate_modality_latents.npz
#     (joint/rna/msi), written by baseline_multigrate.py. Multigrate exposes per-modality
#     latents, so (unlike the old TOTALVI baseline) it also joins the cross-modal metric.
#
# Env: `conda activate nichecompass_liana` (has scib_metrics + jax). FOSCTTM uses the jax
# evals_utils.foscttm_moscot; bio-conservation / iLISI use scib_metrics' functional API.

import argparse
import os
import sys
from pathlib import Path

'''
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "numba_cache"),
)
'''

from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from scipy.stats import hypergeom, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from scib_metrics import ilisi_knn, nmi_ari_cluster_labels_leiden, silhouette_label
from scib_metrics.nearest_neighbors import NeighborsResults

import warnings
warnings.filterwarnings("ignore")

ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
load_dotenv(dotenv_path=ENV_FILE_PATH)

SPATIALJEPA_OUT = Path(os.environ["SPATIALJEPA_OUT"])
BAKLAVA_ROOT = os.environ["SPATIALJEPA_SHARED"]
# evals_utils.foscttm_moscot (jax) lives at the BAKLAVA repo root.
sys.path.insert(0, os.environ["SPATIALJEPA_SHARED"])
from evals_utils import foscttm_moscot

FIG_DIR = Path(os.getenv("SPATIALJEPA_FIG_DIR",
                          f"{os.environ['SPATIALJEPA_MANUSCRIPT']}/figures"))
# Optional suffix (as in train_rna_msi.py) selecting a rerun's model dir;
# outputs go to multigrate_mouse_sma/<sample><tag>, Multigrate latents stay per sample.
RUN_TAG = os.getenv("SPATIALJEPA_RUN_TAG", "")

MM_LABEL_KEY = "MM_clusters"
RNA_LABEL_KEY = "RNA_clusters"
MSI_LABEL_KEY = "MSI_clusters"

# per-modality latent obsm keys written by train_rna_msi.py
PER_MODALITY_KEYS = {
    "teacher": ("teacher_X_emb_st", "teacher_X_emb_sm"),
    "student": ("student_X_emb_st", "student_X_emb_sm"),
    "nonspatial": ("nonspatial_X_emb_st", "nonspatial_X_emb_sm"),
}

# Spatial-structure metric hyperparameters (match analyze_rna_atac.py).
N_SPATIAL_BLOCKS = 10
NICHE_K = 15
SSM_RNG = np.random.default_rng(0)

# ── Metric D: dopamine-representation hyperparameters ─────────────────────────
# Quantify whether the SPATIAL models (teacher/student) arrange the dopamine input
# feature into a latent structure that aligns with the intact-vs-lesioned striatal
# axis better than the nonspatial baseline. Dopamine is an input to all three models,
# so the claim is structural (Metrics 1-3), not "the latent can decode dopamine"
# (the decodability control, Metric 4, is expected to be at parity across models).
DOPA_MODELS = ("teacher", "student", "nonspatial", "multigrate")  # others added if present
MODEL_COLORS = {
    "teacher": "#66C2A5",      # Teal/Green from Set2
    "student": "#FC8D62",      # Orange from Set2
    "nonspatial": "#8DA0CB",   # Blue/Gray from Set2
    "multigrate": "#E78AC3",   # Pink from Set2 -- an external method, not a \spajepa arm
}
DOPA_FEATURE = "msi:Dopamine"        # the clean intact-striatum-specific peak (vs the
                                      # weaker, ~0.2-correlated "msi:Dopamine (single)").
DOPA_LAYER = "normalized"            # the per-spot values the models were trained on; no
                                      # extra transform (AUROC is rank-invariant; Moran's I
                                      # is reported on these same values for all models).
DOPA_K = 15                          # kNN graph degree on the joint latent (NOT the UMAP).
DOPA_LATENT_DIMS = 10                # every model's latent is reduced to this width before
                                      # the kNN graphs are built. SpatialMETA's arms are
                                      # natively 10-D but Multigrate's joint latent is 16-D,
                                      # and neighbourhood geometry depends on how many
                                      # dimensions the distance is taken in, so comparing
                                      # them unmatched would confound model with latent
                                      # width. Set to None to use native widths.
DOPA_K_SWEEP = (10, 15, 30, 50)      # robustness sweep for the headline metrics.
DOPA_TOP_Q = 0.90                    # "dopamine-high" latent region = top 10% smoothed score.
DOPA_N_BOOT = 1000                   # block-bootstrap resamples for CIs.
DOPA_N_BOOT_BLOCKS = 25              # spatial blocks resampled with replacement. Only ONE
                                      # section is available, so this within-section spatial
                                      # block bootstrap replaces the (impossible) by-section
                                      # bootstrap; it respects spatial autocorrelation but is
                                      # anti-conservative — see note in the printed output.
DOPA_RNG = np.random.default_rng(0)


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(
        description="Benchmark SMA joint RNA+MSI embeddings.",
        allow_abbrev=False,
    )
    parser.add_argument("--sample-id", type=str, default="V11L12-109_B1")
    parser.add_argument("--lisi-perplexity", type=int, default=30)
    if notebook:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def _dense(mat):
    return mat.toarray() if sp.issparse(mat) else np.asarray(mat)


def _standardize(X, identity=True):
    if identity:
        return np.asarray(X, dtype=float)
    return StandardScaler().fit_transform(np.asarray(X, dtype=float))


def _to_pca(X, n_comps=30):
    """Top components of a feature block (TruncatedSVD == PCA on centred input)."""
    X = _dense(X)
    n = min(n_comps, X.shape[1] - 1)
    return TruncatedSVD(n_components=n, random_state=0).fit_transform(
        StandardScaler(with_mean=False).fit_transform(X)
    )


def _neighbors(X, k):
    """Build a scib_metrics NeighborsResults from an embedding."""
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, idx = nn.kneighbors(X)
    return NeighborsResults(indices=idx, distances=dist)


# ── data loading ─────────────────────────────────────────────────────────────
def load_inputs(sample_id):
    jepa_dir = SPATIALJEPA_OUT / "spatialjepa_models" / (sample_id + RUN_TAG)
    mg_npz = SPATIALJEPA_OUT / "multigrate_mouse_sma" / sample_id / "multigrate_modality_latents.npz"

    import scanpy as sc
    joint = sc.read_h5ad(jepa_dir / "joint_adata.h5ad")
    for key in ("teacher_X_emb", "student_X_emb", "spatial"):
        if key not in joint.obsm:
            raise KeyError(f"joint_adata missing obsm['{key}']")
    if "nonspatial_X_emb" not in joint.obsm:
        warnings.warn(
            "joint_adata missing obsm['nonspatial_X_emb']; the vanilla SpatialMETA "
            "baseline will be excluded. Re-run train_rna_msi.py to add it."
        )
    for lab in (MM_LABEL_KEY, RNA_LABEL_KEY, MSI_LABEL_KEY):
        if lab not in joint.obs.columns:
            raise KeyError(f"joint_adata missing obs['{lab}']")

    # Per-modality SpatialMETA latents, precomputed by train_rna_msi.py.
    per_modality = {}
    for name, (st_key, sm_key) in PER_MODALITY_KEYS.items():
        if st_key in joint.obsm and sm_key in joint.obsm:
            per_modality[name] = {
                "st": np.asarray(joint.obsm[st_key]),
                "sm": np.asarray(joint.obsm[sm_key]),
            }
        else:
            warnings.warn(
                f"per-modality latents for '{name}' ({st_key}/{sm_key}) not in "
                f"joint_adata.obsm; its cross-modal metrics will be skipped. Re-run "
                f"train_rna_msi.py for sample '{sample_id}' to add them."
            )

    # Multigrate joint + per-modality latents (baseline_multigrate.py), aligned to the
    # joint_adata spot order. Multigrate exposes per-modality latents, so it also joins the
    # cross-modal metric (the old TOTALVI baseline could not).
    x_multigrate = None
    if mg_npz.exists():
        d = np.load(mg_npz, allow_pickle=True)
        mg_obs = pd.Index([str(x) for x in d["obs_names"]])
        pos = mg_obs.get_indexer(joint.obs_names)
        if (pos < 0).any():
            raise ValueError("Multigrate latents do not cover all joint_adata spots.")
        x_multigrate = np.asarray(d["joint"])[pos]
        per_modality["multigrate"] = {
            "st": np.asarray(d["rna"])[pos],
            "sm": np.asarray(d["msi"])[pos],
        }
        # Also expose it the way the SpatialMETA arms are exposed. The dopamine block
        # reaches for obsm["<model>_X_emb"] by name, so without this Multigrate is
        # silently absent from that comparison while joining every other metric.
        joint.obsm["multigrate_X_emb"] = x_multigrate
    else:
        warnings.warn(
            f"Multigrate latents not found at {mg_npz}; Multigrate will be excluded. "
            f"Run baseline_multigrate.py for sample '{sample_id}' first."
        )

    return joint, x_multigrate, per_modality


# ── Metric A: bio-conservation (scib_metrics) ────────────────────────────────
def bio_conservation(emb, labels_by_key):
    """scib silhouette_label (rescaled to [0,1]) + Leiden NMI/ARI vs each label set."""
    nr = _neighbors(emb, 15)
    rows = []
    for lab_name, labels in labels_by_key.items():
        labels = np.asarray(labels)
        rows.append((f"silhouette_{lab_name}",
                     float(silhouette_label(emb, labels, rescale=True))))
        na = nmi_ari_cluster_labels_leiden(nr, labels)
        rows.append((f"nmi_{lab_name}", float(na["nmi"])))
        rows.append((f"ari_{lab_name}", float(na["ari"])))
    return rows


# ── Metric C: cross-modal alignment ─────────────────────────────────────────
def foscttm(x, y):
    """Mean Fraction Of Samples Closer Than the True Match (jax, evals_utils)."""
    return float(np.asarray(foscttm_moscot(_standardize(x), _standardize(y))).mean())


def ilisi_modality_mixing(st, sm, perplexity=30):
    """iLISI in [0,1] of the modality label over the stacked RNA/MSI latents (scib)."""
    Z = np.vstack([_standardize(st), _standardize(sm)])
    batches = np.concatenate([np.zeros(len(st)), np.ones(len(sm))])
    nr = _neighbors(Z, 3 * perplexity)
    return float(ilisi_knn(nr, batches, perplexity=perplexity))


# ── Metric B: spatial structure (ported from analyze_rna_atac.py) ──────────
def _codes_and_onehot(labels):
    cat = pd.Series(np.asarray(labels)).astype("category")
    return cat.cat.codes.to_numpy(), pd.get_dummies(cat).to_numpy().astype(float)


def _mean_spearman(y_true, y_pred):
    rhos = []
    for j in range(y_true.shape[1]):
        rho, _ = spearmanr(y_true[:, j], y_pred[:, j])
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


def _sample_pairs(idx, max_pairs, rng):
    m = idx.size
    n_pairs = min(max_pairs, m * (m - 1) // 2)
    a = rng.integers(0, m, size=n_pairs)
    b = rng.integers(0, m, size=n_pairs)
    keep = a != b
    return idx[a[keep]], idx[b[keep]]


class SpatialScorer:
    """Niche recovery, within-cluster coherence, and (adapted) orthogonal-target
    recovery for a set of joint embeddings, scored with spatial-block CV.

    Adaptation vs the RNA-ATAC reference: cluster partitions are RNA_clusters +
    MSI_clusters, and orthogonal-target predicts neighbour-averaged MSI PCA from
    the JOINT embedding (the joint emb already encodes MSI, so this measures
    MSI-decodability of the embedding, not held-out cross-modal transfer).
    """

    def __init__(self, coords, rna_labels, msi_labels, msi_pca):
        self.coords = np.asarray(coords, dtype=float)
        self.n_ref = self.coords.shape[0]
        self.rna_codes, self.rna_onehot = _codes_and_onehot(rna_labels)
        self.msi_codes, self.msi_onehot = _codes_and_onehot(msi_labels)
        self.cluster_partitions = [self.rna_codes, self.msi_codes]
        self.msi_pca = msi_pca
        self.block_labels = KMeans(
            n_clusters=N_SPATIAL_BLOCKS, random_state=0, n_init=10
        ).fit_predict(self.coords)

    def _build_targets(self, coords_for_targets):
        nn = NearestNeighbors(n_neighbors=NICHE_K + 1).fit(coords_for_targets)
        nbr = nn.kneighbors(coords_for_targets, return_distance=False)[:, 1:]
        y_niche = np.concatenate(
            [self.rna_onehot[nbr].mean(axis=1), self.msi_onehot[nbr].mean(axis=1)],
            axis=1,
        )
        y_orth = _standardize(self.msi_pca[nbr].mean(axis=1))
        return y_niche, y_orth

    def _cv_predict_per_block(self, X, Y, scorer):
        out = {}
        for b in np.unique(self.block_labels):
            test = self.block_labels == b
            train = self.block_labels != b
            if test.sum() < 5 or train.sum() < 20:
                continue
            scaler = StandardScaler().fit(X[train])
            ridge = Ridge(alpha=1.0).fit(scaler.transform(X[train]), Y[train])
            pred = ridge.predict(scaler.transform(X[test]))
            out[b] = scorer(Y[test], pred)
        return out

    def _coherence_per_block(self, Xjoint, coords_for_dist):
        out = {}
        for b in np.unique(self.block_labels):
            idx = np.where(self.block_labels == b)[0]
            if idx.size < 30:
                continue
            rhos, weights = [], []
            for codes in self.cluster_partitions:
                for c in np.unique(codes[idx]):
                    ci = idx[codes[idx] == c]
                    if ci.size < 8:
                        continue
                    pa, pb = _sample_pairs(ci, max_pairs=2000, rng=SSM_RNG)
                    if pa.size < 10:
                        continue
                    ed = np.linalg.norm(Xjoint[pa] - Xjoint[pb], axis=1)
                    pdist = np.linalg.norm(coords_for_dist[pa] - coords_for_dist[pb], axis=1)
                    rho, _ = spearmanr(ed, pdist)
                    if np.isfinite(rho):
                        rhos.append(rho)
                        weights.append(ci.size)
            if rhos:
                out[b] = float(np.average(rhos, weights=weights))
        return out

    def score_lineup(self, lineup, coords_for_dist, coords_for_targets, tag):
        y_niche, y_orth = self._build_targets(coords_for_targets)
        rows = []
        for name, joint_emb in lineup.items():
            niche = self._cv_predict_per_block(joint_emb, y_niche, _mean_spearman)
            orth = self._cv_predict_per_block(joint_emb, y_orth, _mean_spearman)
            coher = self._coherence_per_block(joint_emb, coords_for_dist)
            for metric, per_block in (
                ("niche_recovery", niche),
                ("orthogonal_target", orth),
                ("within_cluster_coherence", coher),
            ):
                for fold, value in per_block.items():
                    rows.append({"setting": tag, "model": name, "metric": metric,
                                 "fold": int(fold), "value": value})
        return pd.DataFrame(rows)


# ── Metric D: dopamine representation across models ──────────────────────────
# Implements the dopamine-representation plan: structural smoothness of dopamine on
# each model's latent graph (Moran's I), the embedding's structural gain in register
# with the lesion axis (ΔAUROC), intact-striatal specificity of the dopamine-high
# latent region (enrichment), and a decodability control showing parity. All metrics
# use the standardized 10-D latent, an identical k and euclidean metric across models,
# and the same dopamine vector / transform for every model.
def get_dopamine(joint, feature=DOPA_FEATURE, layer=DOPA_LAYER):
    """The dopamine input feature for every spot (same vector for all models)."""
    if feature not in joint.var_names:
        raise KeyError(f"dopamine feature '{feature}' not in joint_adata.var_names")
    col = joint.layers[layer][:, joint.var_names.get_loc(feature)]
    return _dense(col).ravel().astype(float)


def standardized_latents(joint, models=DOPA_MODELS, n_dims=DOPA_LATENT_DIMS):
    """z-score (per dim) each model's joint latent, so kNN distances compare.

    Latents wider than ``n_dims`` are first reduced by PCA to the narrowest width in
    the lineup. Every metric below reads dopamine off a kNN graph, and how coherent
    that graph looks depends on how many dimensions the distance is taken in --
    Multigrate's joint latent is 16-D against SpatialMETA's 10-D, so scoring them at
    native width would confound the model with its latent width and penalise the
    wider one through distance concentration alone. Pass ``n_dims=None`` to compare
    at native widths instead (and say so, if the numbers are reported that way).
    """
    raw = {m: np.asarray(joint.obsm[f"{m}_X_emb"], float)
           for m in models if f"{m}_X_emb" in joint.obsm}
    if n_dims is not None and raw:
        target = min([n_dims] + [Z.shape[1] for Z in raw.values()])
        for m, Z in list(raw.items()):
            if Z.shape[1] > target:
                print(f"[dopamine] {m}: {Z.shape[1]}-D latent -> {target}-D by PCA, "
                      "to match the narrowest model in the lineup")
                raw[m] = PCA(n_components=target, random_state=0).fit_transform(Z)
    return {m: StandardScaler().fit_transform(Z) for m, Z in raw.items()}


def _knn_idx(Z, k):
    """Neighbor index array (n, k) on the latent, self excluded."""
    k_eff = min(k + 1, len(Z))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(Z)
    return nn.kneighbors(Z, return_distance=False)[:, 1:]


def _neigh_mean(values, idx):
    """Neighborhood mean of `values` over the kNN graph, excluding self."""
    return values[idx].mean(axis=1)


def morans_I_knn(values, idx):
    """Moran's I of `values` over a symmetric binary kNN weight matrix."""
    n = len(values)
    if n < 3:
        return np.nan
    z = values - values.mean()
    rows = np.repeat(np.arange(n), idx.shape[1])
    W = sp.csr_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=(n, n))
    W = W.maximum(W.T)  # symmetric binary adjacency
    S0 = W.sum()
    den = float((z * z).sum())
    if den == 0 or S0 == 0:
        return np.nan
    return float((n / S0) * (z @ (W @ z)) / den)


def delta_auroc(Z, dopa, y, k):
    """ΔAUROC = AUROC(neighbor-smoothed dopamine) − AUROC(raw dopamine) for label y."""
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    idx = _knn_idx(Z, k)
    smoothed = _neigh_mean(dopa, idx)
    a_raw = roc_auc_score(y, dopa)
    a_sm = roc_auc_score(y, smoothed)
    return a_sm - a_raw, a_raw, a_sm


def intact_enrichment(Z, dopa, intact_str_mask, k, q=DOPA_TOP_Q):
    """Intact-striatal enrichment of the dopamine-high latent region (full graph)."""
    idx = _knn_idx(Z, k)
    smoothed = _neigh_mean(dopa, idx)
    top = smoothed >= np.quantile(smoothed, q)
    N, K, n_top = len(smoothed), int(intact_str_mask.sum()), int(top.sum())
    hit = int((top & intact_str_mask).sum())
    p = float(hypergeom.sf(hit - 1, N, K, n_top)) if n_top and K else np.nan
    frac, base, eps = hit / max(n_top, 1), K / N, 1e-9
    log_odds = float(np.log(((frac + eps) / (1 - frac + eps)) /
                            ((base + eps) / (1 - base + eps))))
    return {"frac_intact_striatal": frac, "base_rate": base,
            "log_odds": log_odds, "hypergeom_p": p}


def decodability_cv(Z_sub, dopa_sub, blocks_sub):
    """Spatial-block CV regression latent→dopamine (ridge + kNN). Control for parity."""
    rows = []
    for b in np.unique(blocks_sub):
        te, tr = blocks_sub == b, blocks_sub != b
        if te.sum() < 5 or tr.sum() < 20:
            continue
        scaler = StandardScaler().fit(Z_sub[tr])
        for lname, learner in (("ridge", Ridge(alpha=1.0)),
                               ("knn", KNeighborsRegressor(n_neighbors=15))):
            learner.fit(scaler.transform(Z_sub[tr]), dopa_sub[tr])
            pred = learner.predict(scaler.transform(Z_sub[te]))
            rho = spearmanr(dopa_sub[te], pred).correlation
            ss_tot = float(((dopa_sub[te] - dopa_sub[te].mean()) ** 2).sum())
            r2 = 1.0 - float(((dopa_sub[te] - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
            rows.append({"learner": lname, "spearman": float(rho) if np.isfinite(rho) else np.nan,
                         "r2": r2})
    return rows


def _striatal_metrics(Zs_dict, dopa, y, k):
    """Point metrics on a (possibly resampled) striatal cell set, all models paired.

    Zs_dict maps model -> latent for the SAME striatal cells; `y` is intact==1; the
    intact / lesioned Moran's I rebuild the kNN graph on that group's own subset.
    """
    intact, lesioned = y == 1, y == 0
    out = {}
    for m, Z in Zs_dict.items():
        d_auc = delta_auroc(Z, dopa, y, k)[0]
        m_int = (morans_I_knn(dopa[intact], _knn_idx(Z[intact], k))
                 if intact.sum() > k else np.nan)
        m_les = (morans_I_knn(dopa[lesioned], _knn_idx(Z[lesioned], k))
                 if lesioned.sum() > k else np.nan)
        out[m] = {"delta_auroc": d_auc, "moran_intact": m_int, "moran_lesioned": m_les}
    return out


def dopamine_representation_analysis(joint, sample_id, k=DOPA_K, n_boot=DOPA_N_BOOT):
    """Full dopamine-representation comparison: returns (results_long, comparison, decod,
    ksweep) tidy DataFrames and writes the headline figures."""
    dopa = get_dopamine(joint)
    latents = standardized_latents(joint)
    if not latents:
        warnings.warn("No SpatialMETA latents found for dopamine analysis; skipping.")
        return None
    lesion = joint.obs["lesion"].astype(str).to_numpy()
    region = joint.obs["region"].astype(str).to_numpy()
    coords = np.asarray(joint.obsm["spatial"], float)
    intact_all = lesion == "intact"
    striatum = region == "striatum"
    intact_str = intact_all & striatum
    groups = {
        "intact_striatum": intact_all & striatum,
        "lesioned_striatum": (~intact_all) & striatum,
        "intact_not_striatum": intact_all & (~striatum),
        "lesioned_not_striatum": (~intact_all) & (~striatum),
    }
    print(f"\n[dopamine] feature='{DOPA_FEATURE}' layer='{DOPA_LAYER}', k={k}; "
          f"models={list(latents)}")
    print("[dopamine] group sizes: " +
          ", ".join(f"{g}={int(m.sum())}" for g, m in groups.items()))
    print("[dopamine] raw within-group dopamine variance: " +
          ", ".join(f"{g}={dopa[m].var():.3f}" for g, m in groups.items()))

    # ── point estimates ──────────────────────────────────────────────────────
    rows = []
    # Metric 1: Moran's I of dopamine per model × group (graph on each group's subset).
    for m, Z in latents.items():
        for g, mask in groups.items():
            mi = morans_I_knn(dopa[mask], _knn_idx(Z[mask], k)) if mask.sum() > k else np.nan
            rows.append({"model": m, "group": g, "metric": "morans_I", "value": mi})
    # Metric 2/3: striatum-restricted ΔAUROC and full-graph intact enrichment.
    s_idx = np.where(striatum)[0]
    y_str = intact_all[s_idx].astype(int)
    for m, Z in latents.items():
        d_auc, a_raw, a_sm = delta_auroc(Z[s_idx], dopa[s_idx], y_str, k)
        rows.append({"model": m, "group": "striatum", "metric": "delta_auroc", "value": d_auc})
        rows.append({"model": m, "group": "striatum", "metric": "auroc_raw", "value": a_raw})
        rows.append({"model": m, "group": "striatum", "metric": "auroc_smoothed", "value": a_sm})
        enr = intact_enrichment(Z, dopa, intact_str, k)
        rows.append({"model": m, "group": "all", "metric": "enrich_log_odds",
                     "value": enr["log_odds"]})
        rows.append({"model": m, "group": "all", "metric": "enrich_frac_intact_striatal",
                     "value": enr["frac_intact_striatal"]})
        rows.append({"model": m, "group": "all", "metric": "enrich_hypergeom_p",
                     "value": enr["hypergeom_p"]})
    results_long = pd.DataFrame(rows)

    # ── spatial-block bootstrap CIs (paired across models) ───────────────────
    Zs = {m: Z[s_idx] for m, Z in latents.items()}
    dopa_s, coords_s = dopa[s_idx], coords[s_idx]
    n_blk = min(DOPA_N_BOOT_BLOCKS, len(s_idx) // 10)
    blocks = KMeans(n_clusters=n_blk, random_state=0, n_init=10).fit_predict(coords_s)
    ublocks = np.unique(blocks)
    block_cells = {b: np.where(blocks == b)[0] for b in ublocks}
    boot = {m: {met: [] for met in ("delta_auroc", "moran_intact", "moran_lesioned")}
            for m in latents}
    boot_diff = {m: {met: [] for met in ("delta_auroc", "moran_intact")}
                 for m in latents if m != "nonspatial"}
    has_baseline = "nonspatial" in latents
    for _ in range(n_boot):
        chosen = DOPA_RNG.choice(ublocks, size=len(ublocks), replace=True)
        cell = np.concatenate([block_cells[b] for b in chosen])
        y_b = y_str[cell]
        if len(np.unique(y_b)) < 2:
            continue
        per = _striatal_metrics({m: Z[cell] for m, Z in Zs.items()}, dopa_s[cell], y_b, k)
        for m, vals in per.items():
            for met, v in vals.items():
                boot[m][met].append(v)
        if has_baseline:
            for m in boot_diff:
                boot_diff[m]["delta_auroc"].append(per[m]["delta_auroc"] - per["nonspatial"]["delta_auroc"])
                boot_diff[m]["moran_intact"].append(per[m]["moran_intact"] - per["nonspatial"]["moran_intact"])

    # ── full-graph spatial-block bootstrap for enrichment log-odds ───────────
    # Enrichment is computed on the FULL cell graph (not the striatal subset), so it
    # needs its own bootstrap over spatial blocks spanning all cells. Paired across
    # models on each resample, so the diffs vs nonspatial share sampling noise.
    n_blk_all = min(DOPA_N_BOOT_BLOCKS, len(dopa) // 10)
    blocks_all = KMeans(n_clusters=n_blk_all, random_state=0, n_init=10).fit_predict(coords)
    ublocks_all = np.unique(blocks_all)
    block_cells_all = {b: np.where(blocks_all == b)[0] for b in ublocks_all}
    boot_enrich = {m: [] for m in latents}
    boot_enrich_diff = {m: [] for m in latents if m != "nonspatial"}
    for _ in range(n_boot):
        chosen = DOPA_RNG.choice(ublocks_all, size=len(ublocks_all), replace=True)
        cell = np.concatenate([block_cells_all[b] for b in chosen])
        ism = intact_str[cell]
        if ism.sum() == 0:
            continue
        vals = {m: intact_enrichment(Z[cell], dopa[cell], ism, k)["log_odds"]
                for m, Z in latents.items()}
        for m, v in vals.items():
            boot_enrich[m].append(v)
        if has_baseline:
            for m in boot_enrich_diff:
                boot_enrich_diff[m].append(vals[m] - vals["nonspatial"])

    def _ci(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        return (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) if v.size else (np.nan, np.nan)

    # attach CIs to the bootstrapped striatal metrics
    results_long["ci_low"], results_long["ci_high"] = np.nan, np.nan
    boot_metric_group = {("delta_auroc", "striatum"): "delta_auroc",
                         ("morans_I", "intact_striatum"): "moran_intact",
                         ("morans_I", "lesioned_striatum"): "moran_lesioned"}
    for i, r in results_long.iterrows():
        bkey = boot_metric_group.get((r["metric"], r["group"]))
        if bkey is not None and r["model"] in boot:
            lo, hi = _ci(boot[r["model"]][bkey])
            results_long.at[i, "ci_low"], results_long.at[i, "ci_high"] = lo, hi
        elif r["metric"] == "enrich_log_odds" and r["model"] in boot_enrich:
            lo, hi = _ci(boot_enrich[r["model"]])
            results_long.at[i, "ci_low"], results_long.at[i, "ci_high"] = lo, hi

    # ── model-comparison table (spatial vs nonspatial, paired diffs) ─────────
    comp_rows = []
    if has_baseline:
        for m in boot_diff:
            for met, label in (("delta_auroc", "delta_auroc"),
                               ("moran_intact", "morans_I_intact_striatum")):
                diffs = np.asarray([x for x in boot_diff[m][met] if np.isfinite(x)])
                lo, hi = _ci(diffs)
                comp_rows.append({
                    "comparison": f"{m} - nonspatial", "metric": label,
                    "diff_mean": float(np.mean(diffs)) if diffs.size else np.nan,
                    "ci_low": lo, "ci_high": hi,
                    "win_fraction": float((diffs > 0).mean()) if diffs.size else np.nan,
                })
        for m in boot_enrich_diff:
            diffs = np.asarray([x for x in boot_enrich_diff[m] if np.isfinite(x)])
            lo, hi = _ci(diffs)
            comp_rows.append({
                "comparison": f"{m} - nonspatial", "metric": "enrich_log_odds",
                "diff_mean": float(np.mean(diffs)) if diffs.size else np.nan,
                "ci_low": lo, "ci_high": hi,
                "win_fraction": float((diffs > 0).mean()) if diffs.size else np.nan,
            })
    comparison = pd.DataFrame(comp_rows)

    # ── decodability control (parity expected) ───────────────────────────────
    decod_rows = []
    blocks_full = KMeans(n_clusters=N_SPATIAL_BLOCKS, random_state=0,
                         n_init=10).fit_predict(coords_s)
    for m, Z in Zs.items():
        for d in decodability_cv(Z, dopa_s, blocks_full):
            decod_rows.append({"model": m, **d})
    decod = pd.DataFrame(decod_rows)
    decod_summary = (decod.groupby(["model", "learner"])[["spearman", "r2"]].mean()
                     .reset_index() if not decod.empty else decod)

    # ── k-sensitivity sweep (point estimates only) ───────────────────────────
    ksweep_rows = []
    for kk in DOPA_K_SWEEP:
        per = _striatal_metrics(Zs, dopa_s, y_str, kk)
        for m, vals in per.items():
            ksweep_rows.append({"model": m, "k": kk, "delta_auroc": vals["delta_auroc"],
                                "moran_intact": vals["moran_intact"]})
    ksweep = pd.DataFrame(ksweep_rows)

    # ── console summary ──────────────────────────────────────────────────────
    print("\n[dopamine] results (point estimate [95% block-bootstrap CI]):")
    pe = results_long[results_long.metric.isin(
        ["delta_auroc", "morans_I", "enrich_log_odds"])]
    for met in ["delta_auroc", "morans_I", "enrich_log_odds"]:
        sub = pe[pe.metric == met]
        print(f"  {met}:")
        for _, r in sub.iterrows():
            ci = (f" [{r.ci_low:.3f}, {r.ci_high:.3f}]"
                  if np.isfinite(r.ci_low) else "")
            print(f"    {r.model:11s} {r.group:22s} {r.value:.3f}{ci}")
    if not comparison.empty:
        print("\n[dopamine] spatial vs nonspatial (paired block bootstrap):")
        print(comparison.round(3).to_string(index=False))
    if not decod_summary.empty:
        print("\n[dopamine] decodability control (spatial-block CV, parity expected):")
        print(decod_summary.round(3).to_string(index=False))
    print("\n[dopamine] NOTE: only one section is available, so CIs come from a within-"
          "section spatial-block bootstrap (resampling whole spatial blocks). This "
          "respects spatial autocorrelation but is anti-conservative relative to a "
          "by-section bootstrap; treat CIs as indicative, not inferential.")

    _dopamine_figures(results_long, comparison, decod, ksweep, latents, sample_id)
    return results_long, comparison, decod, ksweep


def _nonneg_err(parts):
    """Stack lower/upper error-bar lengths, clipped at 0: a percentile-bootstrap CI
    need not contain the point estimate, and matplotlib rejects negative lengths."""
    return np.clip(np.vstack(parts), 0, None)


def _finish_svg(fig, output_file):
    """Show a figure or save it as SVG, regardless of the supplied suffix."""
    if output_file is None:
        plt.show()
    else:
        output_path = Path(output_file).with_suffix(".svg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] wrote {output_path}")
    return fig


def _dopamine_model_order(df, model_order=None):
    """Return the requested model order followed by any additional models."""
    present = list(pd.unique(df["model"]))
    requested = list(DOPA_MODELS if model_order is None else model_order)
    return [m for m in requested if m in present] + [
        m for m in present if m not in requested
    ]


def FIG_dopamine_structural_gain(
    results_long,
    *,
    sample_id="V11L12-109_B1",
    model_order=None,
    output_file=None,
):
    """ΔAUROC per model with spatial-block-bootstrap confidence intervals."""
    order = _dopamine_model_order(results_long, model_order)
    d = results_long[(results_long.metric == "delta_auroc")].set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    yerr = _nonneg_err([d.value - d.ci_low, d.ci_high - d.value])
    colors = [MODEL_COLORS.get(m, "#888888") for m in order]
    ax.bar(order, d.value, yerr=yerr, capsize=4, color=colors,
           edgecolor="#555", linewidth=1)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_ylabel("ΔAUROC (smoothed − raw dopamine)")
    ax.set_title("Dopamine structural gain")
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    return _finish_svg(fig, output_file)


def FIG_dopamine_latent_graph_smoothness(
    results_long,
    *,
    sample_id="V11L12-109_B1",
    model_order=None,
    output_file=None,
):
    """Dopamine Moran's I per model and anatomical/lesion group."""
    order = _dopamine_model_order(results_long, model_order)
    mi = results_long[results_long.metric == "morans_I"]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    grp_order = ["intact_striatum", "lesioned_striatum",
                 "intact_not_striatum", "lesioned_not_striatum"]
    grp_labels = ["intact striatum", "lesioned striatum",
                  "intact non-striatum", "lesioned non-striatum"]
    x = np.arange(len(grp_order))
    w = 0.8 / max(len(order), 1)
    for i, m in enumerate(order):
        sub = mi[mi.model == m].set_index("group").reindex(grp_order)
        err = _nonneg_err([(sub.value - sub.ci_low).fillna(0),
                         (sub.ci_high - sub.value).fillna(0)])
        ax.bar(x + i * w, sub.value, width=w, yerr=err, capsize=3, label=m,
               color=MODEL_COLORS.get(m, "#888888"), edgecolor="#555", linewidth=0.8)
    ax.set_xticks(x + w * (len(order) - 1) / 2)
    ax.set_xticklabels(grp_labels, rotation=15, ha="right")
    ax.set_ylabel("Moran's I of dopamine")
    ax.set_title("Dopamine latent-graph smoothness")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _finish_svg(fig, output_file)


def FIG_dopamine_intact_striatal_enrichment(
    results_long,
    *,
    sample_id="V11L12-109_B1",
    model_order=None,
    output_file=None,
):
    """Intact-striatal enrichment of the dopamine-high latent region."""
    order = _dopamine_model_order(results_long, model_order)
    en = (results_long[results_long.metric == "enrich_log_odds"]
          .set_index("model").reindex(order))
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    en_err = _nonneg_err([(en.value - en.ci_low).fillna(0),
                        (en.ci_high - en.value).fillna(0)])
    colors = [MODEL_COLORS.get(m, "#888888") for m in order]
    ax.bar(order, en.value, yerr=en_err, capsize=4,
           color=colors, edgecolor="#555", linewidth=1)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_ylabel("log-odds (intact-striatal | dopamine-high)")
    ax.set_title("Intact-striatal specificity")
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    return _finish_svg(fig, output_file)


def FIG_dopamine_composite(
    results_long,
    *,
    sample_id="V11L12-109_B1",
    model_order=None,
    output_file=None,
):
    """3-panel horizontal composite figure of dopamine representation metrics."""
    order = _dopamine_model_order(results_long, model_order)
    colors = [MODEL_COLORS.get(m, "#888888") for m in order]
    
    # 1 row, 3 columns with custom width ratios to give the smoothness plot more space
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), gridspec_kw={'width_ratios': [1.6, 1.0, 1.0]})
    
    # ── Panel 1: Latent-graph smoothness (Moran's I) ─────────────────────────
    ax = axes[0]
    mi = results_long[results_long.metric == "morans_I"]
    grp_order = ["intact_striatum", "lesioned_striatum",
                 "intact_not_striatum", "lesioned_not_striatum"]
    grp_labels = ["intact striatum", "lesioned striatum",
                  "intact non-striatum", "lesioned non-striatum"]
    x = np.arange(len(grp_order))
    w = 0.8 / max(len(order), 1)
    
    for i, m in enumerate(order):
        sub = mi[mi.model == m].set_index("group").reindex(grp_order)
        err = _nonneg_err([(sub.value - sub.ci_low).fillna(0),
                         (sub.ci_high - sub.value).fillna(0)])
        ax.bar(x + i * w, sub.value, width=w, yerr=err, capsize=3, label=m,
               color=colors[i], edgecolor="#555", linewidth=0.8)
               
    ax.set_xticks(x + w * (len(order) - 1) / 2)
    ax.set_xticklabels(grp_labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Moran's I of dopamine", fontsize=10)
    ax.set_title("Dopamine latent-graph smoothness", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    
    # ── Panel 2: Structural gain (ΔAUROC) ────────────────────────────────────
    ax = axes[1]
    d = results_long[(results_long.metric == "delta_auroc")].set_index("model").loc[order]
    yerr = _nonneg_err([d.value - d.ci_low, d.ci_high - d.value])
    ax.bar(order, d.value, yerr=yerr, capsize=4, color=colors,
           edgecolor="#555", linewidth=1)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_ylabel("ΔAUROC (smoothed − raw dopamine)", fontsize=10)
    ax.set_title("Dopamine structural gain", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(9)
    
    # ── Panel 3: Intact-striatal enrichment (log-odds) ───────────────────────
    ax = axes[2]
    en = (results_long[results_long.metric == "enrich_log_odds"]
          .set_index("model").reindex(order))
    en_err = _nonneg_err([(en.value - en.ci_low).fillna(0),
                        (en.ci_high - en.value).fillna(0)])
    ax.bar(order, en.value, yerr=en_err, capsize=4,
           color=colors, edgecolor="#555", linewidth=1)
    ax.axhline(0, color="#999", lw=0.8, ls="--")
    ax.set_ylabel("log-odds (intact-striatal | dopamine-high)", fontsize=10)
    ax.set_title("Intact-striatal specificity", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=15)
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_fontsize(9)
    
    fig.tight_layout()
    return _finish_svg(fig, output_file)


def FIG_dopamine_k_sensitivity(
    ksweep,
    *,
    sample_id="V11L12-109_B1",
    model_order=None,
    output_file=None,
):
    """Sensitivity of ΔAUROC and intact-striatal Moran's I to kNN degree."""
    order = _dopamine_model_order(ksweep, model_order)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    met_labels = {
        "delta_auroc": "ΔAUROC",
        "moran_intact": "Moran's I (intact striatum)"
    }
    for met, ax in zip(("delta_auroc", "moran_intact"), axes):
        for m in order:
            sub = ksweep[ksweep.model == m].sort_values("k")
            ax.plot(sub.k, sub[met], marker="o", label=m, color=MODEL_COLORS.get(m, "#888888"))
        ax.set_xlabel("k (kNN degree)")
        ax.set_ylabel(met_labels[met])
        ax.set_title(met_labels[met])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Dopamine metric k-sensitivity", y=1.02)
    fig.tight_layout()
    return _finish_svg(fig, output_file)


def _dopamine_figures(results_long, comparison, decod, ksweep, latents, sample_id):
    """Write the four reusable dopamine benchmark panels as SVG files."""
    order = [m for m in DOPA_MODELS if m in latents]
    FIG_dopamine_structural_gain(
        results_long,
        sample_id=sample_id,
        model_order=order,
        output_file=FIG_DIR / "sma_dopamine_delta_auroc.svg",
    )
    FIG_dopamine_latent_graph_smoothness(
        results_long,
        sample_id=sample_id,
        model_order=order,
        output_file=FIG_DIR / "sma_dopamine_morans_I.svg",
    )
    FIG_dopamine_intact_striatal_enrichment(
        results_long,
        sample_id=sample_id,
        model_order=order,
        output_file=FIG_DIR / "sma_dopamine_enrichment.svg",
    )
    FIG_dopamine_k_sensitivity(
        ksweep,
        sample_id=sample_id,
        model_order=order,
        output_file=FIG_DIR / "sma_dopamine_ksweep.svg",
    )
    FIG_dopamine_composite(
        results_long,
        sample_id=sample_id,
        model_order=order,
        output_file=FIG_DIR / "sma_dopamine_composite.svg",
    )


# ── outputs ──────────────────────────────────────────────────────────────────
def save_figure(fig, filename):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] wrote {path}")


def barplot(df, value_col, title, filename):
    # df is already aggregated to one value per (model, metric); no CI needed.
    g = sns.catplot(
        data=df, x="model", y=value_col, col="metric", col_wrap=4,
        kind="bar", palette="Set2", height=3.2, aspect=0.8, sharey=False, ci=None,
    )
    g.set_titles("{col_name}")
    g.set(xlabel="", ylabel="")
    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelbottom=True, labelrotation=30)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.axhline(0.0, color="#999999", lw=0.8, ls="--")
        for patch in ax.patches:
            patch.set_edgecolor("#cccccc")
            patch.set_linewidth(1)
    g.fig.suptitle(title, y=1.02)
    g.fig.tight_layout()
    save_figure(g.fig, filename)


def ensure_obs_labels_are_str(adata, label_keys):
    for label_key in label_keys:
        if label_key in adata.obs.columns:
            adata.obs[label_key] = adata.obs[label_key].astype(str)


def _stack_modalities_for_ingest(joint_emb, st_emb, sm_emb):
    joint_emb = np.asarray(joint_emb, dtype=float)
    st_emb = np.asarray(st_emb, dtype=float)
    sm_emb = np.asarray(sm_emb, dtype=float)
    if not (joint_emb.shape[0] == st_emb.shape[0] == sm_emb.shape[0]):
        warnings.warn(
            "Cannot ingest-project modality embeddings: multimodal/ST/SM row "
            f"counts are {joint_emb.shape[0]}/{st_emb.shape[0]}/{sm_emb.shape[0]}."
        )
        return None

    joint_dim = joint_emb.shape[1]
    st_dim = st_emb.shape[1]
    sm_dim = sm_emb.shape[1]

    if st_dim == joint_dim and sm_dim == joint_dim:
        return np.vstack([st_emb, sm_emb])

    if joint_dim == st_dim + sm_dim:
        st_query = np.hstack([st_emb, np.zeros((st_emb.shape[0], sm_dim))])
        sm_query = np.hstack([np.zeros((sm_emb.shape[0], st_dim)), sm_emb])
        return np.vstack([st_query, sm_query])

    warnings.warn(
        "Cannot ingest-project modality embeddings: multimodal/ST/SM latent "
        f"dims are {joint_dim}/{st_dim}/{sm_dim}."
    )
    return None


def _set_fixed_knn_distances_for_ingest(adata_ref, neighbors_key, rep_key):
    """Make ingest's pynndescent init robust to sparse zero-distance drops."""
    n_obs = adata_ref.n_obs
    n_neighbors = int(adata_ref.uns[neighbors_key]["params"]["n_neighbors"])
    if n_obs <= 1:
        return

    k_graph = min(n_neighbors, n_obs)
    k_search = min(k_graph + 1, n_obs)
    rep = np.asarray(adata_ref.obsm[rep_key], dtype=float)
    dist, idx = NearestNeighbors(n_neighbors=k_search).fit(rep).kneighbors(rep)

    graph_width = k_graph - 1
    graph_idx = np.empty((n_obs, graph_width), dtype=int)
    graph_dist = np.empty((n_obs, graph_width), dtype=float)
    for i in range(n_obs):
        keep = idx[i] != i
        row_idx = idx[i, keep][:graph_width]
        row_dist = dist[i, keep][:graph_width]
        if row_idx.size != graph_width:
            raise ValueError(
                f"Could not build fixed-width ingest KNN graph for row {i}: "
                f"expected {graph_width} neighbors, got {row_idx.size}."
            )
        graph_idx[i] = row_idx
        graph_dist[i] = row_dist

    rows = np.repeat(np.arange(n_obs), graph_width)
    data = graph_dist.ravel() + 1e-12
    distances_key = adata_ref.uns[neighbors_key]["distances_key"]
    adata_ref.obsp[distances_key] = sp.csr_matrix(
        (data, (rows, graph_idx.ravel())),
        shape=(n_obs, n_obs),
    )


def plot_umap_grid(adata, lineup, lineup_st, lineup_sm, sample_id, filename):
    """Compute and save one UMAP grid: model rows by embedding-family columns."""
    import scanpy as sc

    cluster_specs = [
        ("Multimodal", "multimodal", lineup, MM_LABEL_KEY),
        ("RNA / ST", "st", lineup_st, RNA_LABEL_KEY),
        ("SM / MSI", "sm", lineup_sm, MSI_LABEL_KEY),
    ]
    missing = [label_key for _, _, _, label_key in cluster_specs
               if label_key not in adata.obs.columns]
    if missing:
        warnings.warn(
            f"Missing obs labels for UMAP plotting: {missing}; skipping plot."
        )
        return

    model_order = []
    for _, _, embeddings, _ in cluster_specs:
        model_order.extend([name for name in embeddings.keys() if name not in model_order])
    if not model_order:
        print("[INFO] no embeddings found; skipping UMAP plot")
        return

    color_keys = [MM_LABEL_KEY, RNA_LABEL_KEY, MSI_LABEL_KEY]
    plot_adata = sc.AnnData(
        X=np.zeros((adata.n_obs, 1), dtype=np.float32),
        obs=adata.obs[color_keys].copy(),
    )
    for color_key in color_keys:
        plot_adata.obs[color_key] = plot_adata.obs[color_key].astype(str)

    fig, axes = plt.subplots(
        nrows=len(model_order),
        ncols=len(cluster_specs) + 1,
        figsize=(16.0, 3.2 * len(model_order)),
        squeeze=False,
    )

    for row, name in enumerate(model_order):
        ref_rep_key = None
        ref_neighbors_key = None
        ref_umap_key = None
        joint_emb = None

        for col, (column_title, rep_family, embeddings, color_key) in enumerate(cluster_specs):
            ax = axes[row, col]
            emb = embeddings.get(name)
            if emb is None:
                ax.axis("off")
                continue

            emb = np.asarray(emb, dtype=float)
            if emb.ndim != 2 or emb.shape[0] != adata.n_obs:
                warnings.warn(
                    f"Skipping {rep_family}/{name}: expected ({adata.n_obs}, n_latent), "
                    f"got {emb.shape}."
                )
                ax.axis("off")
                continue

            rep_key = f"X_{rep_family}_{name}"
            neighbors_key = f"{name}_{rep_family}_neighbors"
            umap_key = f"{name}_{rep_family}_umap"
            plot_adata.obsm[rep_key] = emb
            sc.pp.neighbors(
                plot_adata,
                use_rep=rep_key,
                n_neighbors=15,
                key_added=neighbors_key,
            )
            sc.tl.umap(
                plot_adata,
                min_dist=1,
                spread=1,
                neighbors_key=neighbors_key,
            )
            plot_adata.obsm[umap_key] = plot_adata.obsm["X_umap"].copy()
            sc.pl.embedding(
                plot_adata,
                basis=umap_key,
                color=color_key,
                ax=ax,
                show=False,
                size=60,
                title=f"{name} {column_title}: {color_key}",
            )

            if rep_family == "multimodal":
                ref_rep_key = rep_key
                ref_neighbors_key = neighbors_key
                ref_umap_key = umap_key
                joint_emb = emb

        ax = axes[row, len(cluster_specs)]
        st_emb = lineup_st.get(name)
        sm_emb = lineup_sm.get(name)
        if joint_emb is None or st_emb is None or sm_emb is None:
            ax.axis("off")
            continue

        query_rep = _stack_modalities_for_ingest(joint_emb, st_emb, sm_emb)
        if query_rep is None:
            ax.axis("off")
            continue

        query_obs_names = (
            [f"{obs_name}_rna" for obs_name in adata.obs_names.astype(str)]
            + [f"{obs_name}_msi" for obs_name in adata.obs_names.astype(str)]
        )
        query_adata = sc.AnnData(
            X=np.zeros((query_rep.shape[0], 1), dtype=np.float32),
            obs=pd.DataFrame(
                {
                    "modality": np.repeat(["RNA / ST", "MSI / SM"], adata.n_obs),
                },
                index=query_obs_names,
            ),
        )
        query_adata.obs["modality"] = query_adata.obs["modality"].astype(str)
        query_adata.obsm[ref_rep_key] = query_rep
        plot_adata.obsm["X_umap"] = plot_adata.obsm[ref_umap_key].copy()
        _set_fixed_knn_distances_for_ingest(plot_adata, ref_neighbors_key, ref_rep_key)
        sc.tl.ingest(
            query_adata,
            plot_adata,
            embedding_method="umap",
            neighbors_key=ref_neighbors_key,
        )
        ingest_umap_key = f"{name}_modalities_on_multimodal_umap"
        query_adata.obsm[ingest_umap_key] = query_adata.obsm["X_umap"].copy()
        sc.pl.embedding(
            query_adata,
            basis=ingest_umap_key,
            color="modality",
            ax=ax,
            show=False,
            size=30,
            title=f"{name} RNA+MSI on multimodal UMAP",
        )

    fig.suptitle(f"SMA {sample_id}: latent UMAPs", y=1.0)
    fig.tight_layout()
    save_figure(fig, filename)


#%%
def main():
    args = parse_args(notebook=is_notebook())
    sample_id = args.sample_id
    print(f"[INFO] Benchmarking SMA sample {sample_id}")

    joint, x_multigrate, per_modality = load_inputs(sample_id)
    ensure_obs_labels_are_str(joint, (MM_LABEL_KEY, RNA_LABEL_KEY, MSI_LABEL_KEY))
    n_spots = joint.n_obs
    rna_labels = joint.obs[RNA_LABEL_KEY].to_numpy()
    msi_labels = joint.obs[MSI_LABEL_KEY].to_numpy()
    coords = np.asarray(joint.obsm["spatial"], dtype=float)
    print(f"[INFO] {n_spots} spots; "
          f"RNA_clusters={len(np.unique(rna_labels))}, MSI_clusters={len(np.unique(msi_labels))}")

    # PCA floor: TruncatedSVD(30) on the ST block + on the SM block of `normalized`.
    st_mask = joint.var["type"].values == "ST"
    sm_mask = joint.var["type"].values == "SM"
    norm = joint.layers["normalized"]
    st_pca = _to_pca(norm[:, st_mask], 30)
    msi_pca = _to_pca(norm[:, sm_mask], 30)
    pca_floor = np.concatenate([st_pca, msi_pca], axis=1)

    # lineup of joint embeddings (standardized so distances/ridge are comparable)
    lineup = {
        "teacher": _standardize(joint.obsm["teacher_X_emb"]),
        "student": _standardize(joint.obsm["student_X_emb"]),
    }
    if "nonspatial_X_emb" in joint.obsm:
        lineup["nonspatial"] = _standardize(joint.obsm["nonspatial_X_emb"])
    if x_multigrate is not None:
        lineup["multigrate"] = _standardize(x_multigrate)
    lineup["pca"] = _standardize(pca_floor)

    # Per-modality latents are available for SpatialMETA in joint_adata.obsm and for
    # Multigrate in multigrate_modality_latents.npz; include PCA floors for context.
    lineup_st = {
        name: _standardize(mods["st"]) for name, mods in per_modality.items()
    }
    lineup_sm = {
        name: _standardize(mods["sm"]) for name, mods in per_modality.items()
    }
    lineup_st["pca"] = _standardize(st_pca)
    lineup_sm["pca"] = _standardize(msi_pca)

    # Match SpatialMETA training UMAPs as closely as possible: build neighbours on
    # the raw latent arrays stored/exported by each model, not metric-standardized copies.
    umap_lineup = {
        "teacher": np.asarray(joint.obsm["teacher_X_emb"], dtype=float),
        "student": np.asarray(joint.obsm["student_X_emb"], dtype=float),
    }
    if "nonspatial_X_emb" in joint.obsm:
        umap_lineup["nonspatial"] = np.asarray(joint.obsm["nonspatial_X_emb"], dtype=float)
    if x_multigrate is not None:
        umap_lineup["multigrate"] = np.asarray(x_multigrate, dtype=float)
    umap_lineup["pca"] = np.asarray(pca_floor, dtype=float)
    umap_lineup_st = {
        name: np.asarray(mods["st"], dtype=float) for name, mods in per_modality.items()
    }
    umap_lineup_sm = {
        name: np.asarray(mods["sm"], dtype=float) for name, mods in per_modality.items()
    }
    umap_lineup_st["pca"] = np.asarray(st_pca, dtype=float)
    umap_lineup_sm["pca"] = np.asarray(msi_pca, dtype=float)

    # UMAPs: one row per model, with multimodal / RNA-ST / SM-MSI columns.
    plot_umap_grid(
        joint,
        umap_lineup,
        umap_lineup_st,
        umap_lineup_sm,
        sample_id,
        filename="sma_benchmark_umap_grid.pdf",
    )

    # ── Metric A: bio-conservation ──────────────────────────────────────────
    labels_by_key = {RNA_LABEL_KEY: rna_labels, MSI_LABEL_KEY: msi_labels}
    bio_rows = []
    for name, emb in lineup.items():
        for metric, value in bio_conservation(emb, labels_by_key):
            bio_rows.append({"model": name, "metric": metric, "value": value})
    bio_df = pd.DataFrame(bio_rows)
    print("\n[bio-conservation]")
    print(bio_df.pivot(index="model", columns="metric", values="value").round(3).to_string())

    # ── Metric B: spatial structure ─────────────────────────────────────────
    scorer = SpatialScorer(coords, rna_labels, msi_labels, msi_pca)
    perm = SSM_RNG.permutation(n_spots)
    spatial_df = pd.concat(
        [
            scorer.score_lineup(lineup, coords, coords, tag="real"),
            scorer.score_lineup(lineup, coords[perm], coords[perm], tag="coord_permuted"),
        ],
        ignore_index=True,
    )
    spatial_summary = (
        spatial_df.groupby(["setting", "model", "metric"])["value"].mean().reset_index()
    )
    print("\n[spatial structure] real (mean over folds):")
    print(spatial_summary[spatial_summary.setting == "real"]
          .pivot(index="model", columns="metric", values="value").round(3).to_string())
    print("\n[spatial structure] coord-permuted control (~chance):")
    print(spatial_summary[spatial_summary.setting == "coord_permuted"]
          .pivot(index="model", columns="metric", values="value").round(3).to_string())

    # ── Metric C: cross-modal alignment ─────────────────────────────────────
    cross_rows = []
    for name, mods in per_modality.items():
        st, sm = mods["st"], mods["sm"]
        if st.shape[1] != sm.shape[1]:
            warnings.warn(f"[{name}] RNA/MSI latent dims differ "
                          f"({st.shape[1]} vs {sm.shape[1]}); skipping FOSCTTM.")
            continue
        cross_rows.append({"model": name, "metric": "1-FOSCTTM", "value": 1.0 - foscttm(st, sm)})
        cross_rows.append({"model": name, "metric": "iLISI",
                           "value": ilisi_modality_mixing(st, sm, args.lisi_perplexity)})
    cross_df = pd.DataFrame(cross_rows)
    print("\n[cross-modal alignment]:")
    if not cross_df.empty:
        print(cross_df.pivot(index="model", columns="metric", values="value").round(3).to_string())
    else:
        print("  (no per-modality latents found; see warning above)")

    # ── figures + tidy CSV ──────────────────────────────────────────────────
    joint_metrics_df = pd.concat([bio_df, spatial_summary[spatial_summary.setting == "real"]
                                  .drop(columns="setting")[["model", "metric", "value"]]],
                                 ignore_index=True)
    barplot(joint_metrics_df, "value",
            f"SMA {sample_id}: joint-embedding metrics",
            "sma_benchmark_joint_metrics_barplot.pdf")
    if not cross_df.empty:
        barplot(cross_df, "value",
                f"SMA {sample_id}: cross-modal alignment",
                "sma_benchmark_crossmodal_barplot.pdf")

    all_metrics = pd.concat([
        joint_metrics_df.assign(family="joint"),
        cross_df.assign(family="crossmodal") if not cross_df.empty else cross_df,
    ], ignore_index=True)
    out_dir = SPATIALJEPA_OUT / "multigrate_mouse_sma" / (sample_id + RUN_TAG)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "benchmark_metrics.csv"
    all_metrics.to_csv(out_csv, index=False)
    print(f"\n[INFO] wrote {out_csv}")

    # ── Metric D: dopamine representation across models ──────────────────────
    if DOPA_FEATURE in joint.var_names and {"lesion", "region"} <= set(joint.obs.columns):
        dopa_out = dopamine_representation_analysis(joint, sample_id)
        if dopa_out is not None:
            results_long, comparison, decod, ksweep = dopa_out
            for df, fname in (
                (results_long, "dopamine_metrics.csv"),
                (comparison, "dopamine_model_comparison.csv"),
                (decod, "dopamine_decodability.csv"),
                (ksweep, "dopamine_ksweep.csv"),
            ):
                path = out_dir / fname
                meta = {"sample_id": sample_id, "n_boot": DOPA_N_BOOT,
                        "dopamine_feature": DOPA_FEATURE}
                if "k" not in df.columns:  # ksweep already carries a per-row k
                    meta["k"] = DOPA_K
                df.assign(**meta).to_csv(path, index=False)
                print(f"[INFO] wrote {path}")
    else:
        warnings.warn(
            f"Skipping dopamine analysis for '{sample_id}': needs var['{DOPA_FEATURE}'] "
            f"and obs['lesion']/obs['region'] (only V11L12-109_B1 has all three)."
        )


if __name__ == "__main__":
    main()

# %%

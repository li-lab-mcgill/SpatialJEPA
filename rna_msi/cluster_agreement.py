#%% Held-out cluster agreement and spatial smoothness: a simpler companion to Fig. 5d-f.
#
# For every evaluation unit of the benchmark (a training section in-sample, the unseen half
# of a training section, another animal's section), each representation is clustered
# (KMeans on the standardized latent, K in KS) and compared by ARI and NMI with
#   teacher       the spatial teacher of the same run and seed (A_teacher; also the
#                 early-stopped A_early_teacher), clustered the same way, and
#   anatomy       region x lesion labels (intact/lesioned x striatum/other; K = 4).
# Spatial smoothness, without any reference: mean per-dimension Moran's I of the latent on
# the symmetric 6-NN tissue graph, and cluster coherence (fraction of a spot's tissue
# neighbours in its cluster, K = 10; chance = sum_k p_k^2).
#
# Representations (RNA+MSI input, joint latent; Fig. 5d-f condition): teachers with the
# scored section's own graph and graph-free students of A / A_early / A_wd, the ridge
# re-distillation (A_ridge, 10__), the identity teacher E and its student B, stock
# SpatialMETA G, untrained input_pca10 / smoothed_input_pca10 (graph-averaged input),
# spatial_xy (KMeans on coordinates) and, as a noise floor, the teacher of another seed
# (A_teacher_seed / A_early_teacher_seed).
#
# Writes <runs-dir>/evaluation/cluster_agreement_{long,summary}.csv; with --target external
# (see the section below) cluster_agreement_external_{long,summary}.csv and bridge_fidelity.csv.
# Env: $PY_MSI

from __future__ import annotations
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402

import argparse
import importlib.util
import itertools
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
load_dotenv(dotenv_path=ENV_FILE_PATH)

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(os.environ["SPATIALJEPA_RNA_MSI"])
sys.path.insert(0, str(SCRIPT_DIR))
import heldout_animals_common as atb

KS = (5, 10, 20)
K_COHERENCE = 10
REFERENCE_TEACHERS = ("A_teacher", "A_early_teacher")
# config directory -> representations read from it (joint latent)
RUN_REPS = {"A": ("A_teacher", "A", "A_ridge"), "A_early": ("A_early_teacher", "A_early"),
            "A_wd": ("A_wd_teacher", "A_wd"), "B": ("E", "B"), "G": ("G",)}


def parse_args():
    parser = argparse.ArgumentParser(description="Cluster agreement / spatial smoothness on held-out tissue.",
                                     allow_abbrev=False)
    parser.add_argument("--runs-dir", type=Path, default=atb.RUNS_DIR)
    parser.add_argument("--designs", nargs="+", default=["pairwise", "within_section"])
    parser.add_argument("--target", choices=("striatum", "external"), default="striatum",
                        help="benchmark sections, or the P22 / Multiome manuscript targets (pairwise models)")
    parser.add_argument("--prepare", action="store_true", help="external: encode the targets (GPU)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#%% metrics

def _standardize(Z):
    Z = np.asarray(Z, np.float64)
    return (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-12)


def kmeans_labels(Z, k: int) -> np.ndarray:
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(_standardize(Z))


def row_standardized_knn(coords, k: int = 6):
    import scipy.sparse as sp

    A = atb.spatial_knn_graph(coords, k=k).astype(float)
    return sp.diags(1.0 / np.asarray(A.sum(1)).ravel()) @ A


def mean_morans_i(Z, W) -> float:
    """Mean over latent dimensions of Moran's I (row-standardized W, so S0 = n)."""
    Z = _standardize(Z)
    return float(np.mean(np.sum(Z * (W @ Z), 0) / np.sum(Z * Z, 0)))


def coherence(labels, W) -> tuple[float, float]:
    """Mean fraction of tissue neighbours sharing a spot's label, and its chance level."""
    onehot = np.eye(labels.max() + 1)[labels]
    same = float(np.mean(np.sum(onehot * (W @ onehot), 1)))
    p = np.bincount(labels) / len(labels)
    return same, float(np.sum(p ** 2))


#%% scoring one evaluation unit

def score_unit(common: dict, by_seed: dict, W, label_refs: dict, ref_labels: dict | None = None,
               no_smoothness=("spatial_xy",), n_init: int = 10) -> list[dict]:
    """ARI/NMI and smoothness rows for one unit.

    ``by_seed``: {seed: {model: Z}} on the unit's spots. ``label_refs``: {name: labels},
    compared at K = their number of classes. ``ref_labels``: {(seed, ref, K): labels} for the
    teacher references; by default the KMeans labels of REFERENCE_TEACHERS in ``by_seed``
    (supplied from elsewhere when the teacher cannot be run on these cells). ``W``: row-
    standardized tissue graph, or None to skip smoothness.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ari, normalized_mutual_info_score as nmi

    label_k = {name: len(np.unique(v)) for name, v in label_refs.items()}
    ks = set(KS) | {K_COHERENCE} | set(label_k.values())
    labels, rows = {}, []
    for seed, lat in by_seed.items():
        for model, Z in lat.items():
            Zs = _standardize(Z)
            for k in ks:
                labels[(seed, model, k)] = KMeans(n_clusters=k, random_state=0, n_init=n_init).fit_predict(Zs)
            if W is None:
                continue
            same, chance = coherence(labels[(seed, model, K_COHERENCE)], W)
            base = dict(common, seed=seed, model=model, reference="none")
            rows += [dict(base, metric="morans_i", value=np.nan if model in no_smoothness else mean_morans_i(Z, W)),
                     dict(base, metric="coherence", value=same),
                     dict(base, metric="coherence_chance", value=chance)]
    if ref_labels is None:
        ref_labels = {key: v for key, v in labels.items() if key[1] in REFERENCE_TEACHERS}
    refs = sorted({r for (_, r, _) in ref_labels})
    for seed, lat in by_seed.items():
        for model in lat:
            for name, truth in label_refs.items():
                for metric, fn in (("ari", ari), ("nmi", nmi)):
                    rows.append(dict(common, seed=seed, model=model, reference=name, metric=metric,
                                     value=fn(truth, labels[(seed, model, label_k[name])])))
            for ref in refs:
                if (seed, ref, KS[0]) not in ref_labels:
                    continue
                for metric, fn in (("ari", ari), ("nmi", nmi)):
                    rows.append(dict(common, seed=seed, model=model, reference=ref, metric=metric,
                                     value=np.mean([fn(ref_labels[(seed, ref, k)], labels[(seed, model, k)])
                                                    for k in KS])))
        # noise floor: the same teacher config trained with another seed
        for ref in refs:
            for other in by_seed:
                if other == seed or (other, ref, KS[0]) not in ref_labels or (seed, ref, KS[0]) not in ref_labels:
                    continue
                for metric, fn in (("ari", ari), ("nmi", nmi)):
                    rows.append(dict(common, seed=seed, model=f"{ref}_seed", reference=ref, metric=metric,
                                     value=np.mean([fn(ref_labels[(seed, ref, k)], ref_labels[(other, ref, k)])
                                                    for k in KS])))
    return rows


#%% per fold

def fold_units(fold: dict, ev8) -> list[dict]:
    src = fold["source_joint"]
    sm_cols = src.var["type"].eq("SM").to_numpy()
    units = []
    for section in fold["meta"]["source_sections"]:
        rows = src.obs[atb.SECTION_KEY].astype(str).eq(section).to_numpy()
        units.append(dict(section=section, emb_key="+".join(fold["meta"]["source_sections"]), inputs="full",
                          rows=rows, obs=src.obs[rows], X=np.hstack([ev8.rna_inputs(src), atb._dense(src.X)[:, sm_cols]])[rows],
                          coords=np.asarray(src.obsm["spatial"])[rows]))
    for section, joint in fold["target_joint"].items():
        msi = fold["target_msi"][section]
        units.append(dict(section=section, emb_key=section, inputs="paired", rows=None, obs=joint.obs,
                          X=np.hstack([ev8.rna_inputs(joint), atb._dense(msi.X)]),
                          coords=np.asarray(msi.obsm["spatial"])))
    source_bases = {atb.base_section(s) for s in fold["meta"]["source_sections"]}
    for i, unit in enumerate(units):
        unit["relation"] = ("in_sample" if i < len(fold["meta"]["source_sections"])
                            else "same_section" if atb.base_section(unit["section"]) in source_bases
                            else "other_animal")
    return units


def baseline_latents(units, ev9) -> dict:
    """Source-fitted 10-D PCA of the paired input, raw and graph-averaged; coordinates."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    src = [u for u in units if u["relation"] == "in_sample"]
    fits = {}
    for name, X in (("input_pca10", np.vstack([u["X"] for u in src])),
                    ("smoothed_input_pca10", np.vstack([ev9.gcn_smooth(u["X"], u["coords"]) for u in src]))):
        scaler = StandardScaler().fit(X)
        fits[name] = (scaler, PCA(n_components=10, random_state=0).fit(scaler.transform(X)))
    out = {}
    for u in units:
        for name in fits:
            X = u["X"] if name == "input_pca10" else ev9.gcn_smooth(u["X"], u["coords"])
            scaler, pca = fits[name]
            out[(name, u["section"])] = pca.transform(scaler.transform(X))
        out[("spatial_xy", u["section"])] = u["coords"]
    return out


def run_latents(run_dir: Path, reps, units) -> dict:
    files = [run_dir / f for f in ("embeddings.npz", "teacher_insample.npz", "heldout_paired.npz",
                                   "diagnostic_latents.npz")]
    emb = {}
    for f in files:
        if f.exists():
            emb.update(dict(np.load(f)))
    out = {}
    for rep in reps:
        for u in units:
            Z = emb.get(f"{rep}|{u['emb_key']}|{u['inputs']}|joint")
            if Z is not None:
                out[(rep, u["section"])] = Z if u["rows"] is None else Z[u["rows"]]
    return out


def evaluate_fold(runs_dir: Path, unit_name: str, ev8, ev9) -> list[dict]:
    fold = atb.load_fold("striatum", [*unit_name.split("+")])
    units = fold_units(fold, ev8)
    design = "within_section" if "@" in unit_name else ("pairwise" if "+" not in unit_name else "loao")
    base_lat = baseline_latents(units, ev9)
    by_seed = {}
    for seed in (0, 1, 2):
        lat = dict(base_lat)
        for cfg, reps in RUN_REPS.items():
            run_dir = runs_dir / "striatum" / f"train-{unit_name}" / cfg / f"seed{seed}"
            if (run_dir / "DONE").exists():
                lat.update(run_latents(run_dir, reps, units))
        by_seed[seed] = lat
    rows = []
    for u in units:
        anat = (u["obs"]["lesion"].astype(str) + "|" + u["obs"]["region"].astype(str)).to_numpy()
        common = dict(design=design, fold=unit_name, section=u["section"], relation=u["relation"])
        unit_lat = {seed: {m: Z for (m, sec), Z in lat.items() if sec == u["section"]} for seed, lat in by_seed.items()}
        rows += score_unit(common, unit_lat, row_standardized_knn(u["coords"]), {"anatomy": anat})
    print(f"[cluster] {unit_name}: {len(units)} units", flush=True)
    return rows


#%% external RNA-only targets: P22 spatial RNA-ATAC and the 10x Multiome
#
# The whole-section (pairwise) benchmark models are encoded on both manuscript targets with
# the MSI block zeroed. Latents compared: the RNA-expert latent (st) for teachers, paired
# students, E, B and G (it only ever sees RNA), the joint latent for the RNA-only students
# (*_rna). Spatial teachers run on P22 with its own 6-NN graph; they cannot run on the
# dissociated Multiome.
#
# P22 references: the teacher of the same run (as above) and the Zhang et al. (2023)
# RNA_clusters / ATAC_clusters (spatially coherent anatomical clusters); smoothness on the
# P22 tissue graph.
# Multiome references, through a bridge independent of every evaluated model: MultiVI
# trained on P22 RNA+ATAC (scripts/baseline_multivi.py) embeds P22
# spots (X_multivi) and Multiome nuclei (X_multivi_zero_shot) in one frame. Each nucleus
# takes the majority label of its BRIDGE_K nearest P22 spots (RNA/ATAC clusters, and the
# P22 teacher's KMeans clusters for that run and seed) and the mean coordinates of its
# BRIDGE_COORD_K nearest spots (pseudo-coordinates for Moran's I / coherence). Non-spatial
# references: CellTypist (manuscript recipe, multiome_celltypist_labels.py) and the 10x
# graph clusters. Bridge fidelity is measured leave-one-out on P22.

EXTERNAL_TARGETS = ("p22", "multiome")
EXTERNAL_MODELS = {   # config directory -> checkpoint -> representation
    "A": {"teacher": "A_teacher", "student": "A", "student_rna": "A_rna"},
    "A_early": {"teacher": "A_early_teacher", "student": "A_early", "student_rna": "A_early_rna"},
    "A_wd": {"teacher": "A_wd_teacher", "student": "A_wd", "student_rna": "A_wd_rna"},
    "B": {"teacher": "E", "student": "B", "student_rna": "B_rna"},
    "G": {"model": "G"},
}
SPATIAL_TEACHERS = ("A_teacher", "A_early_teacher", "A_wd_teacher")
BRIDGE_K, BRIDGE_COORD_K = 15, 5
EXTERNAL_N_INIT = 4


def aligned_dir() -> Path:
    return Path(os.environ["SPATIALJEPA_DATA"]) / "aligned_data"


def external_dir() -> Path:
    path = atb.SPATIALJEPA_OUT / "spatialjepa_benchmark" / "external_targets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_h5(path: Path, multivi_key: str | None = None):
    """obs names, var names and the SCT_counts layer of an aligned h5ad, without X."""
    import anndata as ad
    import h5py

    with h5py.File(path, "r") as h5:
        counts = ad.experimental.read_elem(h5["layers/SCT_counts"])
    a = ad.read_h5ad(path, backed="r")
    out = dict(obs=a.obs.copy(), obs_names=pd.Index(a.obs_names), var_names=pd.Index(a.var_names),
               counts=counts.tocsr(), coords=np.asarray(a.obsm["spatial"], float) if "spatial" in a.obsm else None)
    a.file.close()
    return out


def _multivi(path: Path, key: str) -> pd.DataFrame:
    import h5py

    with h5py.File(path, "r") as h5:
        names = h5["obs"][h5["obs"].attrs["_index"]][:]
        Z = h5["obsm"][key][:]
    return pd.DataFrame(Z, index=[n.decode() if isinstance(n, bytes) else n for n in names])


def load_external(name: str) -> dict:
    if name == "p22":
        t = _read_h5(aligned_dir() / "source_rna_aligned_SCT.h5ad")
        t["labels"] = {c: t["obs"][c].astype(str).to_numpy() for c in ("RNA_clusters", "ATAC_clusters")}
        return t
    t = _read_h5(aligned_dir() / "target_rna_aligned_SCT.h5ad")
    keep = _multivi(aligned_dir() / "multivi_mdata_zero_shot.h5mu", "X_multivi_zero_shot").index
    rows = t["obs_names"].get_indexer(keep)
    if (rows < 0).any():
        raise KeyError("Multiome nuclei in the MultiVI bridge are missing from the RNA file")
    celltypist = pd.read_csv(external_dir() / "multiome_celltypist.csv", index_col=0).iloc[:, 0]
    t.update(obs=t["obs"].iloc[rows], obs_names=t["obs_names"][rows], counts=t["counts"][rows])
    t["labels"] = {"celltypist": celltypist.reindex(t["obs_names"]).astype(str).to_numpy(),
                   "graphclust": t["obs"]["arc_gex_graphclust_Cluster"].astype(str).to_numpy()}
    return t


def external_joint(fold_var: pd.DataFrame, target: dict):
    """Target RNA on a fold's model features: SCT_counts by gene symbol, MSI block zero."""
    from anndata import AnnData

    token = {g.upper(): i for i, g in reversed(list(enumerate(target["var_names"])))}
    st = fold_var["type"].eq("ST").to_numpy()
    idx = np.array([token.get(n.split(":", 1)[-1].upper(), -1) for n in fold_var.index[st]])
    X = np.zeros((len(target["obs_names"]), len(fold_var)), np.float32)
    cols = np.flatnonzero(st)
    X[:, cols[idx >= 0]] = target["counts"][:, idx[idx >= 0]].toarray()
    joint = AnnData(X=X, obs=pd.DataFrame(index=target["obs_names"].astype(str)), var=fold_var.copy())
    if target["coords"] is not None:
        joint.obsm["spatial"] = target["coords"]
    return joint, int((idx >= 0).sum())


def pairwise_folds(runs_dir: Path) -> list[str]:
    return sorted(p.parent.parent.parent.name.removeprefix("train-")
                  for p in runs_dir.glob("striatum/train-*/A/seed0/DONE")
                  if "@" not in p.parent.parent.parent.name and "+" not in p.parent.parent.parent.name)


def prepare_external(args) -> None:
    import anndata as ad

    targets = {name: load_external(name) for name in EXTERNAL_TARGETS}
    for unit in pairwise_folds(args.runs_dir):
        fold_var = ad.read_h5ad(atb.fold_data_dir("striatum", [unit]) / "source_joint.h5ad", backed="r").var.copy()
        joints = {name: external_joint(fold_var, t) for name, t in targets.items()}
        for name, (_, n) in joints.items():
            print(f"[external] {unit} -> {name}: {n}/{int(fold_var['type'].eq('ST').sum())} genes", flush=True)
        for cfg, ckpts in EXTERNAL_MODELS.items():
            for seed in (0, 1, 2):
                run_dir = args.runs_dir / "striatum" / f"train-{unit}" / cfg / f"seed{seed}"
                out = run_dir / "external_latents.npz"
                if not (run_dir / "DONE").exists() or (out.exists() and not args.overwrite):
                    continue
                arrays = {}
                for ckpt, rep in ckpts.items():
                    path = run_dir / f"{ckpt}.pt"
                    for name, (joint, _) in joints.items():
                        if rep in SPATIAL_TEACHERS and joint.obsm.get("spatial") is None:
                            continue
                        for latent, Z in atb.encode_checkpoint(path, joint, device=args.device).items():
                            arrays[f"{rep}|{name}|masked|{latent}"] = Z
                np.savez_compressed(out, **arrays)
                print(f"[external] {unit} {cfg} seed{seed}: {len(arrays)} latents", flush=True)


def external_rep_latent(rep: str) -> str:
    return "joint" if rep.endswith("_rna") else "st"


class Bridge:
    """MultiVI kNN from Multiome nuclei to P22 spots (label vote, pseudo-coordinates)."""

    def __init__(self, p22: dict, multiome: dict):
        from sklearn.neighbors import NearestNeighbors

        src = _multivi(aligned_dir() / "multivi_mdata.h5mu", "X_multivi").reindex(p22["obs_names"])
        tgt = _multivi(aligned_dir() / "multivi_mdata_zero_shot.h5mu", "X_multivi_zero_shot").reindex(multiome["obs_names"])
        self.p22_coords = p22["coords"]
        nn = NearestNeighbors(n_neighbors=BRIDGE_K + 1).fit(src.to_numpy())
        self.idx = nn.kneighbors(tgt.to_numpy(), n_neighbors=BRIDGE_K, return_distance=False)
        self.loo = nn.kneighbors(src.to_numpy(), return_distance=False)[:, 1:]   # P22 spots, self excluded

    @staticmethod
    def vote(labels: np.ndarray, idx: np.ndarray) -> np.ndarray:
        codes, inv = np.unique(labels, return_inverse=True)
        counts = np.zeros((len(idx), len(codes)))
        np.add.at(counts, (np.repeat(np.arange(len(idx)), idx.shape[1]), inv[idx].ravel()), 1)
        return codes[counts.argmax(1)]

    def labels(self, p22_labels, loo=False):
        return self.vote(np.asarray(p22_labels), self.loo if loo else self.idx)

    def coords(self, loo=False):
        idx = (self.loo if loo else self.idx)[:, :BRIDGE_COORD_K]
        return self.p22_coords[idx].mean(1)


def external_latents(runs_dir: Path, unit: str, target: str) -> dict:
    by_seed = {}
    for seed in (0, 1, 2):
        lat = {}
        for cfg, ckpts in EXTERNAL_MODELS.items():
            f = runs_dir / "striatum" / f"train-{unit}" / cfg / f"seed{seed}" / "external_latents.npz"
            if not f.exists():
                continue
            emb = np.load(f)
            for rep in ckpts.values():
                key = f"{rep}|{target}|masked|{external_rep_latent(rep)}"
                if key in emb:
                    lat[rep] = emb[key]
        by_seed[seed] = lat
    return by_seed


def external_baselines(unit: str, target: dict, ev8, ev9) -> dict:
    """Source-fitted and target-fitted RNA PCA (10-D), graph-smoothed PCA and coordinates on
    P22, and the RNA-ATAC comparators stored with the aligned data (MultiGATE students)."""
    import anndata as ad
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fold = atb.load_fold("striatum", [unit])
    joint, _ = external_joint(fold["source_joint"].var, target)
    X_src, X_t = ev8.rna_inputs(fold["source_joint"]), ev8.rna_inputs(joint)
    out = {}
    scaler = StandardScaler().fit(X_src)
    out["rna_pca10"] = PCA(10, random_state=0).fit(scaler.transform(X_src)).transform(scaler.transform(X_t))
    out["rna_pca10_target"] = PCA(10, random_state=0).fit_transform(StandardScaler().fit_transform(X_t))
    if target["coords"] is not None:
        out["smoothed_rna_pca10_target"] = PCA(10, random_state=0).fit_transform(
            StandardScaler().fit_transform(ev9.gcn_smooth(X_t, target["coords"])))
        out["spatial_xy"] = target["coords"]
    p22 = target["coords"] is not None
    lat = ad.read_h5ad(aligned_dir() / ("source_rna_aligned_with_latents.h5ad" if p22
                                        else "target_rna_aligned_with_latents.h5ad"), backed="r")
    rows = pd.Index(lat.obs_names).get_indexer(target["obs_names"])
    keys = {"MultiGATE_teacher": "MultiGATE_spatialjepa_teacher", "MultiGATE": "MultiGATE_spatialjepa_student",
            "MultiGATE_nonspatial": "MultiGATE_nonspatial"}
    for key, name in keys.items():
        if key in lat.obsm and (rows >= 0).all():
            out[name] = np.asarray(lat.obsm[key])[rows]
    lat.file.close()
    if not p22:
        out["MultiVI_bridge"] = _multivi(aligned_dir() / "multivi_mdata_zero_shot.h5mu",
                                         "X_multivi_zero_shot").reindex(target["obs_names"]).to_numpy()
    return out


def evaluate_external(args, ev8, ev9) -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score as ari

    p22, multiome = load_external("p22"), load_external("multiome")
    bridge = Bridge(p22, multiome)
    W_p22 = row_standardized_knn(p22["coords"])
    W_mo = row_standardized_knn(bridge.coords())
    mo_labels = dict(multiome["labels"], **{f"bridged_{k}": bridge.labels(v) for k, v in p22["labels"].items()})
    rows, fidelity = [], []
    for name, v in p22["labels"].items():   # leave-one-out bridge fidelity on P22
        fidelity.append(dict(what=f"{name} (leave-one-out vote)", ari=ari(v, bridge.labels(v, loo=True))))
    W_loo = row_standardized_knn(bridge.coords(loo=True))
    for unit in pairwise_folds(args.runs_dir):
        base = external_baselines(unit, p22, ev8, ev9)
        lat = external_latents(args.runs_dir, unit, "p22")
        by_seed = {seed: dict(base, **l) for seed, l in lat.items()}
        common = dict(design="external", fold=unit, section="P22", relation="p22")
        rows += score_unit(common, by_seed, W_p22, p22["labels"], n_init=EXTERNAL_N_INIT)
        # the P22 teacher's clusters, carried to the Multiome nuclei by the bridge
        ref_labels, mo_by_seed = {}, {}
        base_mo = external_baselines(unit, multiome, ev8, ev9)
        mo_lat = external_latents(args.runs_dir, unit, "multiome")
        for seed, l in lat.items():
            for ref in REFERENCE_TEACHERS:
                if ref not in l:
                    continue
                for k in KS:
                    km = KMeans(n_clusters=k, random_state=0, n_init=EXTERNAL_N_INIT).fit_predict(_standardize(l[ref]))
                    ref_labels[(seed, ref, k)] = bridge.labels(km)
                    fidelity.append(dict(what=f"{ref} K={k} (leave-one-out vote)", fold=unit, seed=seed,
                                         ari=ari(km, bridge.labels(km, loo=True))))
            for model in ("A_teacher", "A_early_teacher", "A", "A_early", "G"):
                if model in l:
                    fidelity.append(dict(what=f"{model} Moran's I, true vs pseudo coordinates", fold=unit, seed=seed,
                                         true=mean_morans_i(l[model], W_p22), pseudo=mean_morans_i(l[model], W_loo)))
            mo_by_seed[seed] = dict(base_mo, **mo_lat.get(seed, {}))
        common = dict(design="external", fold=unit, section="Multiome", relation="multiome")
        rows += score_unit(common, mo_by_seed, W_mo, mo_labels, ref_labels=ref_labels, n_init=EXTERNAL_N_INIT)
        print(f"[external] {unit}: P22 + Multiome scored", flush=True)
    out = args.runs_dir / "evaluation"
    df = pd.DataFrame(rows)
    df.to_csv(out / "cluster_agreement_external_long.csv", index=False)
    summary = (df.groupby(["relation", "reference", "metric", "model", "fold"])["value"].mean()
               .groupby(["relation", "reference", "metric", "model"]).agg(["mean", "min", "max", "count"]))
    summary.reset_index().to_csv(out / "cluster_agreement_external_summary.csv", index=False)
    pd.DataFrame(fidelity).to_csv(out / "bridge_fidelity.csv", index=False)
    print(f"wrote {out / 'cluster_agreement_external_summary.csv'}")


def main():
    args = parse_args()
    ev8 = _load_module("eval8", "heldout_animals_evaluate.py")
    ev9 = _load_module("eval9", "heldout_animals_dopamine_structure.py")
    if args.target == "external":
        if args.prepare:
            prepare_external(args)
        else:
            evaluate_external(args, ev8, ev9)
        return
    folds = sorted({p.parent.parent.parent.name.removeprefix("train-")
                    for p in args.runs_dir.glob("striatum/train-*/A/seed*/DONE")})
    rows = []
    for unit_name in folds:
        design = "within_section" if "@" in unit_name else ("pairwise" if "+" not in unit_name else "loao")
        if design in args.designs:
            rows += evaluate_fold(args.runs_dir, unit_name, ev8, ev9)
    df = pd.DataFrame(rows)
    out = args.runs_dir / "evaluation"
    df.to_csv(out / "cluster_agreement_long.csv", index=False)
    per_section = df.groupby(["design", "relation", "reference", "metric", "model", "fold", "section"])["value"].mean()
    summary = per_section.groupby(["design", "relation", "reference", "metric", "model"]).agg(["mean", "min", "max", "count"])
    summary.reset_index().to_csv(out / "cluster_agreement_summary.csv", index=False)
    print(f"wrote {out / 'cluster_agreement_summary.csv'}")


if __name__ == "__main__":
    main()

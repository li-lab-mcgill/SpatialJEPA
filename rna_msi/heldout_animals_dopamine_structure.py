try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Cross-animal SpatialJEPA ablation benchmark: the manuscript's Fig. 5 dopamine metrics.
#
# Scores every benchmark representation with the three latent-structure metrics used for
# SpatialMETA in the manuscript (Fig. 5d-f; dopamine_embedding_benchmark.py), using that
# script's own functions:
#   structural gain        ΔAUROC = AUROC(latent-kNN-smoothed dopamine) - AUROC(raw),
#                          intact vs lesioned, region spots only
#   latent-graph smoothness Moran's I of dopamine on the latent kNN graph, per lesion x
#                          region group (headline: intact region, as in the manuscript)
#   intact-region specificity log-odds that the top-10% smoothed-dopamine spots are
#                          intact-region spots (full section graph)
# with k=15 on the per-section standardized 10-D latent and dopamine total-count
# normalized to 1e4 over all MSI peaks (the manuscript's `normalized` layer).
#
# Settings:
#   in_sample_full    source section, paired RNA+MSI input (the manuscript's setting)
#   in_sample_masked  source section, MSI input zeroed (dopamine is not an input)
#   transfer          held-out animals' sections, MSI input zeroed (graph-free models)
#   transfer_paired   held-out animals' sections with their *measured* MSI as input: the
#                     Fig. 5d-f condition on a section the model never trained on. Graph
#                     teachers (`<config>_teacher`) build the held-out section's own graph
#                     at inference; in the RNA-only settings they are absent.
# Extra metric `student_teacher_r`: mean per-dimension correlation between a run's student
# and teacher latents on the same spots (the student is trained to match z-scored teacher
# dimensions), i.e. how well distillation holds on spots the student never saw.
#
# Two stages, because dopamine_embedding_benchmark.py and spatialmeta live in different envs:
#   --stage prepare   (spatialMETA env) teacher in-sample latents, held-out paired
#                     latents (heldout_paired.npz) + dopamine vectors
#   --stage evaluate  (nichecompass_liana env) metrics, contrasts, figures
# Outputs: <runs-dir>/evaluation/dopamine_structure_*.csv and figures/.

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
load_dotenv(dotenv_path=ENV_FILE_PATH)

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(os.environ["SPATIALJEPA_RNA_MSI"])
sys.path.insert(0, str(SCRIPT_DIR))
import heldout_animals_common as atb

warnings.filterwarnings("ignore")

STRUCTURE_CONTRASTS = {
    "A_teacher-G (manuscript: teacher vs non-spatial)": {"A_teacher": 1, "G": -1},
    "A_teacher-E (spatial vs identity teacher)": {"A_teacher": 1, "E": -1},
    "C_teacher-F (spatial vs identity teacher, no cross)": {"C_teacher": 1, "F": -1},
    "A_teacher-A_induced_teacher (mini-batch fix, teacher)": {"A_teacher": 1, "A_induced_teacher": -1},
    "A30_teacher-A_teacher (k=30 vs k=6, teacher)": {"A30_teacher": 1, "A_teacher": -1},
    "A-rna_pca10": {"A": 1, "rna_pca10": -1},
    "A-input_pca10": {"A": 1, "input_pca10": -1},
    "A_teacher-A (teacher vs student)": {"A_teacher": 1, "A": -1},
    "A_teacher-spatial_xy": {"A_teacher": 1, "spatial_xy": -1},
    "A_teacher-smoothed_input_pca10": {"A_teacher": 1, "smoothed_input_pca10": -1},
    "A_early_teacher-A_teacher (early-stopped vs full-schedule teacher)": {"A_early_teacher": 1, "A_teacher": -1},
    "A_early-A (student of early-stopped vs full-schedule teacher)": {"A_early": 1, "A": -1},
    "A_wd_teacher-A_teacher (weight-decayed vs default teacher)": {"A_wd_teacher": 1, "A_teacher": -1},
    "A_wd-A (student of weight-decayed vs default teacher)": {"A_wd": 1, "A": -1},
}
FIGURE_MODELS = (
    "A_teacher", "A", "A_induced_teacher", "A_induced", "A30_teacher", "A30",
    "B", "E", "C_teacher", "C", "C30_teacher", "C30", "D", "F", "G",
    "input_pca10", "smoothed_input_pca10", "rna_pca10", "spatial_xy",
    "A_adabn", "A_ridge", "A_es", "A_teacher_nograph", "A_teacher_exprgraph", "A_exprsmooth",
    "A_oracle_ridge", "A_early_teacher", "A_early", "A_wd_teacher", "A_wd",
)
SETTINGS = ("in_sample_full", "in_sample_masked", "transfer_paired", "transfer")
HEADLINE = (
    ("structural_gain", "Dopamine structural gain (ΔAUROC)"),
    ("moran_intact_region", "Latent-graph smoothness (Moran's I, intact region)"),
    ("specificity_log_odds", "Intact-region specificity (log-odds)"),
    # bounded companion: log-odds saturates near 21 when every top-10% spot is intact
    # region (eps=1e-9 in intact_enrichment), which dominates means across sections
    ("specificity_frac", "Fraction of top-10% dopamine spots in intact region"),
)
GROUPS = ("intact_region", "lesioned_region", "intact_outside", "lesioned_outside")


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(
        description="Manuscript dopamine latent-structure metrics for the benchmark runs.",
        allow_abbrev=False,
    )
    parser.add_argument("--stage", choices=("prepare", "evaluate"), required=not notebook,
                        default="evaluate")
    parser.add_argument("--runs-dir", type=Path, default=atb.RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--regions", nargs="+", default=["striatum", "sn"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-figures", action="store_true")
    if notebook:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finished_runs(runs_dir: Path, regions) -> list[dict]:
    runs = []
    for done in sorted(runs_dir.glob("*/train-*/*/seed*/DONE")):
        with open(done.parent / "run.json") as handle:
            meta = json.load(handle)
        if meta["region"] in regions:
            runs.append(dict(meta, run_dir=done.parent))
    return runs


#%% stage 1: teacher latents + dopamine vectors (spatialMETA env)

def stage_prepare(args) -> None:
    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    runs = finished_runs(args.runs_dir, args.regions)
    folds = sorted({(r["region"], tuple(r["train_animals"])) for r in runs})
    for region, train_animals in folds:
        path = atb.fold_data_dir(region, train_animals) / "dopamine_tic1e4.csv"
        if path.exists():
            continue
        with open(path.parent / "fold.json") as handle:
            meta = json.load(handle)
        frames = []
        for section in meta["source_sections"] + meta["target_sections"]:
            series = atb.manuscript_dopamine(section)
            frames.append(pd.DataFrame({"obs_name": series.index, "section": section,
                                        "dopamine": series.to_numpy()}))
        pd.concat(frames).to_csv(path, index=False)
        print(f"[prepare] {path}")

    fold_cache = {}
    for run in runs:
        out = run["run_dir"] / "teacher_insample.npz"
        if out.exists() or not (run["run_dir"] / "teacher.pt").exists():
            continue
        key = (run["region"], tuple(run["train_animals"]))
        if key not in fold_cache:
            fold_cache[key] = atb.load_fold(run["region"], list(run["train_animals"]))
        fold = fold_cache[key]
        latents = atb.encode_teacher_insample(run["run_dir"], fold["source_joint"], device=device)
        if latents is None:   # identity teacher: already in embeddings.npz as E / F
            continue
        source_key = "+".join(fold["meta"]["source_sections"])
        arrays = {
            f"{run['config']}_teacher|{source_key}|{inputs}|{latent}": Z
            for inputs, by_latent in latents.items() for latent, Z in by_latent.items()
        }
        np.savez_compressed(out, **arrays)
        print(f"[prepare] {run['run_id']}: teacher in-sample latents")

    for run in runs:
        out = run["run_dir"] / "heldout_paired.npz"
        if out.exists():
            continue
        key = (run["region"], tuple(run["train_animals"]))
        if key not in fold_cache:
            fold_cache[key] = atb.load_fold(run["region"], list(run["train_animals"]))
        fold = fold_cache[key]
        # RNA-only students never saw MSI input, so paired input is out of distribution
        jobs = {rep: run["run_dir"] / f"{model_key}.pt"
                for rep, model_key in run["representations"].items() if not rep.endswith("_rna")}
        if run["config_spec"].get("graph") not in (None, "identity"):
            jobs[f"{run['config']}_teacher"] = run["run_dir"] / "teacher.pt"
        arrays = {}
        for target_id in fold["meta"]["target_sections"]:
            joint = atb.paired_target_joint(fold, target_id)
            for rep, ckpt in jobs.items():
                for latent, Z in atb.encode_checkpoint(ckpt, joint, device=device).items():
                    if not np.isfinite(Z).all():
                        raise FloatingPointError(f"non-finite {rep} latents on {target_id}")
                    arrays[f"{rep}|{target_id}|paired|{latent}"] = Z
        np.savez_compressed(out, **arrays)
        print(f"[prepare] {run['run_id']}: held-out paired latents ({', '.join(jobs)})")


#%% stage 2: metrics (nichecompass_liana env, imports dopamine_embedding_benchmark)

def spatial_knn_overlap(b6, Z, coords, k) -> float:
    """Mean fraction of a spot's k latent neighbours that are among its k spatial ones."""
    lat = b6._knn_idx(Z, k)
    spa = b6._knn_idx(np.asarray(coords, float), k)
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(lat, spa)]))


def structure_metrics(b6, Z, dopa, obs, region, coords) -> dict:
    """The manuscript's three dopamine metrics for one latent on one section.

    ``latent_spatial_overlap`` says how much the latent kNN graph is just the tissue
    kNN graph (1 for the ``spatial_xy`` reference).
    """
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(np.asarray(Z, float))
    k = b6.DOPA_K
    intact = obs["lesion"].astype(str).to_numpy() == "intact"
    in_region = obs["region"].astype(str).eq(atb.REGION_LABEL[region]).to_numpy()
    out = {}
    s_idx = np.flatnonzero(in_region)
    d_auc, a_raw, a_sm = b6.delta_auroc(Z[s_idx], dopa[s_idx], intact[s_idx].astype(int), k)
    out.update(structural_gain=d_auc, auroc_raw=a_raw, auroc_smoothed=a_sm)
    masks = dict(zip(GROUPS, (intact & in_region, ~intact & in_region,
                              intact & ~in_region, ~intact & ~in_region)))
    for group, mask in masks.items():
        out[f"moran_{group}"] = (b6.morans_I_knn(dopa[mask], b6._knn_idx(Z[mask], k))
                                 if mask.sum() > k else np.nan)
    enrichment = b6.intact_enrichment(Z, dopa, intact & in_region, k)
    out["specificity_log_odds"] = enrichment["log_odds"]
    out["specificity_frac"] = enrichment["frac_intact_striatal"]
    out["latent_spatial_overlap"] = spatial_knn_overlap(b6, Z, coords, k)
    return out


def evaluate_fold(b6, region, train_animals, runs, ev8):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    fold = atb.load_fold(region, list(train_animals))
    source_key = "+".join(fold["meta"]["source_sections"])
    dopa_df = pd.read_csv(atb.fold_data_dir(region, train_animals) / "dopamine_tic1e4.csv")
    dopa_df["obs_name"] = dopa_df["obs_name"].astype(str)
    if any("@" in unit for unit in train_animals):
        design = "within_section"   # trained on half a section
    else:
        design = "pairwise" if len(train_animals) == 1 else "loao"
    base = dict(region=region, design=design, fold=atb.fold_name(train_animals),
                train_animals="+".join(train_animals))

    # evaluation units: each source section in-sample, each target section held out
    units = []
    src = fold["source_joint"]
    sm_cols = src.var["type"].eq("SM").to_numpy()
    for section in fold["meta"]["source_sections"]:
        rows = src.obs[atb.SECTION_KEY].astype(str).eq(section).to_numpy()
        units.append(dict(section=section, kind="in_sample", emb_key=source_key, rows=rows,
                          obs=src.obs[rows], X_rna=ev8.rna_inputs(src)[rows],
                          X_msi=atb._dense(src.X)[rows][:, sm_cols],
                          coords=np.asarray(src.obsm["spatial"])[rows]))
    for section, joint in fold["target_joint"].items():
        # target coordinates come from the evaluation-only MSI file, never the model input
        units.append(dict(section=section, kind="transfer", emb_key=section, rows=None,
                          obs=joint.obs, X_rna=ev8.rna_inputs(joint),
                          X_msi=atb._dense(fold["target_msi"][section].X),
                          coords=np.asarray(fold["target_msi"][section].obsm["spatial"])))
    source_bases = {atb.base_section(s) for s in fold["meta"]["source_sections"]}
    for unit in units:
        if unit["kind"] == "in_sample":
            unit["relation"] = "in_sample"
        elif atb.base_section(unit["section"]) in source_bases:
            unit["relation"] = "same_section"   # unseen half of a training section
        else:
            unit["relation"] = "other_animal"
        d = dopa_df[dopa_df["section"].eq(unit["section"])].set_index("obs_name")["dopamine"]
        unit["dopa"] = d.reindex(unit["obs"].index.astype(str)).to_numpy(float)
        assert np.isfinite(unit["dopa"]).all(), f"dopamine missing for {unit['section']}"
        unit["animal"] = str(unit["obs"]["animal"].iloc[0])

    rows = []

    def add(unit, setting, model, latent, seed, config, Z):
        for metric, value in structure_metrics(b6, Z, unit["dopa"], unit["obs"], region,
                                               unit["coords"]).items():
            rows.append(dict(base, section=unit["section"], animal=unit["animal"],
                             relation=unit["relation"], setting=setting,
                             model=model, latent=latent, seed=seed, config=config,
                             metric=metric, value=value))

    # linear baseline latents (source-fitted 10-D PCA): RNA only (encoder-equivalent
    # log1p input) and paired RNA+MSI (the model's full input), each standardized
    X_src = ev8.rna_inputs(src)
    scaler = StandardScaler().fit(X_src)
    pca = PCA(n_components=10, random_state=0).fit(scaler.transform(X_src))
    XP_src = np.hstack([X_src, atb._dense(src.X)[:, sm_cols]])
    scaler_p = StandardScaler().fit(XP_src)
    pca_p = PCA(n_components=10, random_state=0).fit(scaler_p.transform(XP_src))
    # graph-averaged paired input (per section graph), no learning: what inference-time
    # message passing alone does for a teacher on any section
    src_coords = np.asarray(src.obsm["spatial"])
    XS_src = np.vstack([gcn_smooth(XP_src[u["rows"]], src_coords[u["rows"]])
                        for u in units if u["kind"] == "in_sample"])
    scaler_s = StandardScaler().fit(XS_src)
    pca_s = PCA(n_components=10, random_state=0).fit(scaler_s.transform(XS_src))
    for unit in units:
        Z = pca.transform(scaler.transform(unit["X_rna"]))
        Zp = pca_p.transform(scaler_p.transform(np.hstack([unit["X_rna"], unit["X_msi"]])))
        if unit["kind"] == "in_sample":
            settings, paired = ("in_sample_full", "in_sample_masked"), "in_sample_full"
        else:
            settings, paired = ("transfer", "transfer_paired"), "transfer_paired"
        for setting in settings:
            add(unit, setting, "rna_pca10", "rna", -1, "baseline", Z)
            # reference: the latent is the tissue itself (pure spatial smoothing)
            add(unit, setting, "spatial_xy", "rna", -1, "reference", unit["coords"])
        add(unit, paired, "input_pca10", "rna", -1, "baseline", Zp)
        XS = gcn_smooth(np.hstack([unit["X_rna"], unit["X_msi"]]), unit["coords"])
        add(unit, paired, "smoothed_input_pca10", "rna", -1, "reference",
            pca_s.transform(scaler_s.transform(XS)))

    for run in runs:
        emb = dict(np.load(run["run_dir"] / "embeddings.npz"))
        teacher_file = run["run_dir"] / "teacher_insample.npz"
        reps = list(run["representations"])
        if teacher_file.exists():
            emb.update(dict(np.load(teacher_file)))
            reps.append(f"{run['config']}_teacher")
        paired_file = run["run_dir"] / "heldout_paired.npz"
        if paired_file.exists():
            emb.update(dict(np.load(paired_file)))
        # post-training adjustments from student_teacher_diagnostics.py
        diag_file = run["run_dir"] / "diagnostic_latents.npz"
        if diag_file.exists():
            diag = dict(np.load(diag_file))
            emb.update(diag)
            reps += [r for r in dict.fromkeys(k.split("|", 1)[0] for k in diag) if r not in reps]
        seen = {}   # (setting, section, rep, latent) -> Z, for student-teacher agreement
        for rep in reps:
            for latent in ("joint", "st"):
                for unit in units:
                    if unit["kind"] == "in_sample":
                        jobs = [("in_sample_masked", "masked")]
                        if not rep.endswith("_rna"):  # RNA-only students never saw MSI input
                            jobs.append(("in_sample_full", "full"))
                        for setting, inputs in jobs:
                            key = f"{rep}|{unit['emb_key']}|{inputs}|{latent}"
                            if key not in emb:
                                continue
                            Z = emb[key][unit["rows"]]
                            add(unit, setting, rep, latent, run["seed"], run["config"], Z)
                            seen[(setting, unit["section"], rep, latent)] = Z
                    else:
                        for setting, inputs in (("transfer", "masked"), ("transfer_paired", "paired")):
                            key = f"{rep}|{unit['emb_key']}|{inputs}|{latent}"
                            if key in emb:
                                add(unit, setting, rep, latent, run["seed"], run["config"], emb[key])
                                seen[(setting, unit["section"], rep, latent)] = emb[key]
        teacher_rep = (f"{run['config']}_teacher" if teacher_file.exists()
                       else atb.IDENTITY_TEACHER_NAMES.get(run["config"]))
        for (setting, section, rep, latent), Z in seen.items():
            T = seen.get((setting, section, teacher_rep, latent))
            if rep != run["config"] or T is None:
                continue
            unit = next(u for u in units if u["section"] == section)
            rows.append(dict(base, section=section, animal=unit["animal"],
                             relation=unit["relation"], setting=setting,
                             model=rep, latent=latent, seed=run["seed"], config=run["config"],
                             metric="student_teacher_r", value=mean_dim_corr(Z, T)))
        print(f"[{region} {base['fold']}] {run['config']} seed{run['seed']} done")
    return rows


def gcn_smooth(X: np.ndarray, coords: np.ndarray, k: int = 6) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2 X on the symmetric spatial kNN graph: the averaging a
    SpatialJEPA teacher's graph layer applies to its input (k=6, as in training)."""
    import scipy.sparse as sp
    from sklearn.neighbors import kneighbors_graph

    A = kneighbors_graph(np.asarray(coords, float), n_neighbors=k, include_self=False)
    A = A.maximum(A.T) + sp.identity(A.shape[0], format="csr")
    d = 1.0 / np.sqrt(np.asarray(A.sum(1)).ravel())
    return sp.diags(d) @ A @ sp.diags(d) @ X


def mean_dim_corr(S: np.ndarray, T: np.ndarray) -> float:
    """Mean over latent dimensions of corr(student_d, teacher_d) on the same spots."""
    S = (S - S.mean(0)) / S.std(0).clip(1e-12)
    T = (T - T.mean(0)) / T.std(0).clip(1e-12)
    return float(np.mean((S * T).mean(0)))


UNIT_KEYS = ["region", "design", "fold", "train_animals", "section", "animal", "relation", "setting",
             "latent", "metric"]


def structure_contrasts(df: pd.DataFrame, contrasts: dict):
    jepa = df[df["seed"] >= 0]
    base = df[df["seed"] < 0].drop(columns=["latent"])
    seeds = sorted(jepa["seed"].unique())
    latents = sorted(jepa["latent"].unique())
    expanded = [base.assign(seed=s, latent=l) for s in seeds for l in latents]
    long = pd.concat([jepa] + expanded, ignore_index=True)
    wide = long.pivot_table(index=UNIT_KEYS + ["seed"], columns="model", values="value", aggfunc="first")
    parts = []
    for name, weights in contrasts.items():
        if not set(weights) <= set(wide.columns):
            continue
        value = sum(coef * wide[model] for model, coef in weights.items())
        parts.append(value.rename("value").reset_index().assign(contrast=name))
    per_seed = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    by_unit = (per_seed.groupby(UNIT_KEYS + ["contrast"])["value"]
               .agg(mean="mean", sd="std", n_seeds="count").reset_index())
    summary = (by_unit.groupby(["region", "design", "relation", "setting", "latent", "metric", "contrast"])["mean"]
               .agg(mean_over_units="mean", min="min", max="max", n_units="count",
                    n_positive=lambda v: int((v > 0).sum()))
               .reset_index())
    return by_unit, summary


def fig_structure(sm: pd.DataFrame, region: str, setting: str, ev8, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = sm[sm["region"].eq(region) & sm["setting"].eq(setting) & sm["design"].eq("pairwise")
           & sm["latent"].isin(["joint", "rna"])]
    models = [m for m in FIGURE_MODELS if m in set(d["model"])]
    if not models:
        return
    fig, axes = plt.subplots(len(HEADLINE), 1, figsize=(0.55 * len(models) + 2.8, 2.7 * len(HEADLINE)),
                             sharex=True)
    for ax, (metric, title) in zip(axes, HEADLINE):
        sub = d[d["metric"].eq(metric)]
        for x, model in enumerate(models):
            pts = sub[sub["model"].eq(model)]
            for row in pts.itertuples():
                si = atb.ANIMALS.index(atb.parse_train_unit(row.train_animals.split("+")[0])[0])
                ti = atb.ANIMALS.index(row.animal)
                ax.errorbar(x + (si - 1) * 0.16, row.mean, yerr=0 if np.isnan(row.sd) else row.sd,
                            fmt=ev8.MARKERS[ti], ms=4.5, color=ev8.SERIES[si], mec="white", mew=0.5,
                            elinewidth=0.7)
            if len(pts):
                ax.hlines(pts["mean"].mean(), x - 0.3, x + 0.3, color=ev8.INK, linewidth=1.5, zorder=4)
        ax.axhline(0, color=ev8.INK_2, linewidth=0.7)
        ax.set_title(title, fontsize=9, color=ev8.INK, loc="left")
        ev8._style(ax)
    axes[-1].set_xticks(range(len(models)), models, rotation=40, ha="right", fontsize=8)
    handles = [plt.Line2D([], [], color=ev8.SERIES[i], marker="o", ls="", ms=6, label=f"trained on {a}")
               for i, a in enumerate(atb.ANIMALS)]
    handles += [plt.Line2D([], [], color=ev8.INK_2, marker=ev8.MARKERS[i], ls="", ms=6,
                           label=f"{'scored' if setting.startswith('transfer') else 'section'}: {a}")
                for i, a in enumerate(atb.ANIMALS)]
    handles += [plt.Line2D([], [], color=ev8.INK, lw=1.5, label="mean")]
    axes[0].legend(handles=handles, fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle(f"{region}: manuscript dopamine metrics, {setting.replace('_', ' ')} (joint latent)",
                 fontsize=10, color=ev8.INK)
    fig.tight_layout()
    ev8._save(fig, out / f"{region}_dopamine_structure_{setting}")


SUMMARY_MODELS = ("A_teacher", "A", "E", "B", "G", "input_pca10", "smoothed_input_pca10",
                  "rna_pca10", "spatial_xy")
SUMMARY_SETTINGS = {
    # (setting, relation, legend label)
    "animals": (
        ("in_sample_full", "in_sample", "in-sample, RNA+MSI input (Fig. 5d-f)"),
        ("transfer_paired", "other_animal", "held-out animal, RNA+MSI input"),
        ("transfer", "other_animal", "held-out animal, RNA-only input"),
    ),
    "within_section": (
        ("in_sample_full", "in_sample", "in-sample half, RNA+MSI input"),
        ("transfer_paired", "same_section", "held-out half of the same section, RNA+MSI"),
        ("transfer_paired", "other_animal", "other animals' sections, RNA+MSI"),
    ),
}
SUMMARY_METRICS = (
    ("structural_gain", "Structural gain (ΔAUROC)"),
    ("moran_intact_region", "Moran's I, intact striatum"),
    ("specificity_frac", "Top-10% dopamine spots in intact striatum"),
    ("latent_spatial_overlap", "Latent–tissue neighbour overlap"),
    ("student_teacher_r", "Student–teacher agreement (r)"),
)


def fig_heldout_summary(sm: pd.DataFrame, region: str, design: str, ev8, out: Path):
    """Fig. 5d-f metrics in-sample vs on held-out sections, per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = sm[sm["region"].eq(region) & sm["design"].eq(design) & sm["latent"].isin(["joint", "rna"])]
    models = [m for m in SUMMARY_MODELS if m in set(d["model"])]
    metrics = [(m, t) for m, t in SUMMARY_METRICS if m in set(d["metric"])]
    settings = SUMMARY_SETTINGS["within_section" if design == "within_section" else "animals"]
    if not models or not metrics:
        return
    fig, axes = plt.subplots(len(metrics), 1, figsize=(0.95 * len(models) + 3.2, 2.2 * len(metrics)),
                             sharex=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes[:, 0], metrics):
        sub = d[d["metric"].eq(metric)]
        for x, model in enumerate(models):
            for j, (setting, relation, _) in enumerate(settings):
                vals = sub[sub["model"].eq(model) & sub["setting"].eq(setting)
                           & sub["relation"].eq(relation)]["mean"].to_numpy()
                if not len(vals):
                    continue
                cx = x + (j - 1) * 0.27
                ax.scatter(cx + rng.uniform(-0.05, 0.05, len(vals)), vals, s=15, color=ev8.SERIES[j],
                           marker=ev8.MARKERS[j], edgecolor="white", linewidth=0.5, zorder=3)
                ax.hlines(vals.mean(), cx - 0.11, cx + 0.11, color=ev8.INK, linewidth=1.4, zorder=4)
        ax.axhline(0, color=ev8.INK_2, linewidth=0.7)
        ax.set_title(title, fontsize=9, color=ev8.INK, loc="left")
        ev8._style(ax)
    axes[-1, 0].set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=8)
    handles = [plt.Line2D([], [], color=ev8.SERIES[j], marker=ev8.MARKERS[j], ls="", ms=6, label=label)
               for j, (_, _, label) in enumerate(settings)]
    handles += [plt.Line2D([], [], color=ev8.INK, lw=1.4, label="mean")]
    axes[0, 0].legend(handles=handles, fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    train = {"pairwise": "one animal", "loao": "two animals",
             "within_section": "half a section"}.get(design, design)
    fig.suptitle(f"{region}: Fig. 5d-f metrics, trained on {train} (joint latent; points = scored "
                 "section x training set, seed-mean)", fontsize=9.5, color=ev8.INK)
    fig.tight_layout()
    ev8._save(fig, out / f"{region}_heldout_summary_{design}")


def stage_evaluate(args) -> None:
    b6 = _load_module("bench6", "dopamine_embedding_benchmark.py")
    ev8 = _load_module("eval8", "heldout_animals_evaluate.py")
    out_dir = args.out_dir or (args.runs_dir / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = finished_runs(args.runs_dir, args.regions)
    # only runs that stage 1 has processed (runs may finish between the two stages)
    ready = [r for r in runs
             if (r["run_dir"] / "heldout_paired.npz").exists()
             and (atb.fold_data_dir(r["region"], r["train_animals"]) / "dopamine_tic1e4.csv").exists()]
    if len(ready) < len(runs):
        print(f"[structure] skipping {len(runs) - len(ready)} finished run(s) not yet prepared "
              "(run --stage prepare again)")
    by_fold = {}
    for run in ready:
        by_fold.setdefault((run["region"], tuple(run["train_animals"])), []).append(run)
    rows = []
    for (region, train_animals), fold_runs in sorted(by_fold.items()):
        rows += evaluate_fold(b6, region, train_animals, fold_runs, ev8)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dopamine_structure_long.csv", index=False)
    sm = (df.groupby(UNIT_KEYS + ["model"], dropna=False)["value"]
          .agg(mean="mean", sd="std", n_seeds="count").reset_index())
    sm.to_csv(out_dir / "dopamine_structure_seedmean.csv", index=False)
    contrasts = dict(ev8.CONTRASTS, **STRUCTURE_CONTRASTS)
    by_unit, summary = structure_contrasts(df, contrasts)
    by_unit.to_csv(out_dir / "dopamine_structure_contrasts_by_unit.csv", index=False)
    summary.to_csv(out_dir / "dopamine_structure_contrasts_summary.csv", index=False)

    cols = [m for m, _ in HEADLINE] + ["latent_spatial_overlap", "student_teacher_r"]
    for setting in ("in_sample_full", "transfer_paired", "transfer"):
        d = sm[sm["setting"].eq(setting) & sm["latent"].isin(["joint", "rna"]) & sm["metric"].isin(cols)]
        if d.empty:
            continue
        table = (d.groupby(["region", "design", "relation", "model", "metric"])["mean"].mean()
                 .unstack("metric").reindex(columns=cols))
        order = [m for m in FIGURE_MODELS if m in table.index.get_level_values("model")]
        print(f"\n[{setting}] mean over {'source sections' if setting.startswith('in_') else 'held-out sections'}"
              " (joint latent):")
        print(table.reindex(order, level="model").round(3).to_string())

    if not args.no_figures:
        for region in sorted(set(df["region"])):
            for setting in SETTINGS:
                fig_structure(sm, region, setting, ev8, out_dir / "figures")
            for design in sorted(set(sm["design"])):
                fig_heldout_summary(sm, region, design, ev8, out_dir / "figures")
    print(f"[structure] wrote {out_dir}")


def main():
    args = parse_args(notebook=is_notebook())
    if args.stage == "prepare":
        stage_prepare(args)
    else:
        stage_evaluate(args)


if __name__ == "__main__":
    main()

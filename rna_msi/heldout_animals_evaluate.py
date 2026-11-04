try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Cross-animal SpatialJEPA ablation benchmark: PLS readouts, metrics, contrasts, figures.
#
# Reads every finished run written by heldout_animals_train.py (DONE marker), fits a
# source-only PLS decoder (K=4, as in the manuscript) from each representation to the
# source MSI, predicts the held-out animals' MSI from RNA-only embeddings, and scores the
# predictions against the measured (never model-visible) MSI.
#
# Readout protocols:
#   P0  decoder fitted on source embeddings encoded WITH measured MSI, applied to targets
#       encoded with MSI zeroed (manuscript Supp. Fig. S8)
#   P1  decoder fitted and applied on embeddings encoded with MSI zeroed (primary)
#   P2  as P1, for the student trained with zeroed MSI input (`<config>_rna`)
#   RNA linear RNA->MSI baselines (ridge, PLS, PCA-10 + PLS) fitted on the source
# Latents: joint (q_mu) and the RNA-expert latent st (st.q_mu, MSI-independent).
#
# Outputs (<runs-dir>/evaluation/): metrics_long.csv, metrics_seedmean.csv,
# transfer_matrix.csv, contrasts_by_pair.csv, contrasts_summary.csv,
# target_quality.csv, init_hash_check.csv, dopamine_predictions.csv.gz, figures/.
#
# Env: $PY_MSI (CPU only)

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
load_dotenv(dotenv_path=ENV_FILE_PATH)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(os.environ["SPATIALJEPA_RNA_MSI"])
sys.path.insert(0, str(SCRIPT_DIR))
import heldout_animals_common as atb

warnings.filterwarnings("ignore")

# linear combinations of model scores; models missing from a run set are skipped
CONTRASTS = {
    "A-B (spatial | cross)": {"A": 1, "B": -1},
    "C-D (spatial | no cross)": {"C": 1, "D": -1},
    "A-C (cross | spatial)": {"A": 1, "C": -1},
    "B-D (cross | no spatial)": {"B": 1, "D": -1},
    "(A-B)-(C-D) interaction": {"A": 1, "B": -1, "C": -1, "D": 1},
    "A30-B (spatial k=30 | cross)": {"A30": 1, "B": -1},
    "C30-D (spatial k=30 | no cross)": {"C30": 1, "D": -1},
    "(A30-B)-(C30-D) interaction": {"A30": 1, "B": -1, "C30": -1, "D": 1},
    "A30-A (k=30 vs k=6)": {"A30": 1, "A": -1},
    "B-E (distillation)": {"B": 1, "E": -1},
    "D-F (distillation, no cross)": {"D": 1, "F": -1},
    "A-E (full pipeline vs direct)": {"A": 1, "E": -1},
    "A-A_perm (tissue adjacency)": {"A": 1, "A_perm": -1},
    "A-A_induced (mini-batch fix)": {"A": 1, "A_induced": -1},
    "A-G (vs stock SpatialMETA)": {"A": 1, "G": -1},
    "A-ridge_rna": {"A": 1, "ridge_rna": -1},
    "A-pca10_pls": {"A": 1, "pca10_pls": -1},
}
FIGURE_CONTRASTS = (
    "A-B (spatial | cross)",
    "C-D (spatial | no cross)",
    "A-C (cross | spatial)",
    "(A-B)-(C-D) interaction",
    "A30-B (spatial k=30 | cross)",
    "A30-A (k=30 vs k=6)",
    "B-E (distillation)",
    "A-E (full pipeline vs direct)",
    "A-A_perm (tissue adjacency)",
    "A-A_induced (mini-batch fix)",
    "A-ridge_rna",
)
MODEL_ORDER = (
    "A", "A30", "B", "C", "C30", "D", "E", "F", "A_perm", "A_induced", "G",
    "ridge_rna", "pls_rna", "pca10_pls",
)
BASELINES = ("ridge_rna", "pls_rna", "pca10_pls")
KEYS = ["region", "design", "fold", "train_animals", "target_animal", "target_section", "split",
        "protocol", "latent", "metabolite", "metric"]

# reference palette (dataviz skill): categorical slots 1-3 validated all-pairs (light);
# blue<->red diverging with a gray midpoint; ink tokens for text
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#2a78d6", "#f0efec", "#e34948"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(
        description="Evaluate cross-animal SpatialJEPA ablation runs.",
        allow_abbrev=False,
    )
    parser.add_argument("--runs-dir", type=Path, default=atb.RUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--regions", nargs="+", default=["striatum", "sn"])
    parser.add_argument("--n-components", type=int, default=4)
    parser.add_argument("--primary-metric", default="spearman_region")
    parser.add_argument("--primary-protocol", default="P1")
    parser.add_argument("--no-figures", action="store_true")
    if notebook:
        return parser.parse_known_args()[0]
    return parser.parse_args()


#%% discovery and evaluation units

def discover_runs(runs_dir: Path, regions) -> pd.DataFrame:
    rows = []
    for done in sorted(runs_dir.glob("*/train-*/*/seed*/DONE")):
        with open(done.parent / "run.json") as handle:
            meta = json.load(handle)
        if meta["region"] not in regions:
            continue
        rows.append(dict(
            run_dir=done.parent,
            region=meta["region"],
            train_animals=tuple(meta["train_animals"]),
            config=meta["config"],
            seed=int(meta["seed"]),
            init_hash=meta.get("init_hash"),
            kind=atb.CONFIGS[meta["config"]]["kind"],
            representations=meta["representations"],
        ))
    return pd.DataFrame(rows)


def evaluation_units(fold: dict, region: str, keep: np.ndarray) -> list[dict]:
    """In-sample source section(s) and held-out target sections, with measured MSI."""
    units = []
    source_key = "+".join(fold["meta"]["source_sections"])
    src = fold["source_msi"]
    for section in fold["meta"]["source_sections"]:
        rows = src.obs[atb.SECTION_KEY].astype(str).eq(section).to_numpy()
        units.append(dict(section=section, split="in_sample", emb_key=source_key, rows=rows, msi=src[rows]))
    for section, msi in fold["target_msi"].items():
        units.append(dict(section=section, split="transfer", emb_key=section, rows=None, msi=msi))
    for unit in units:
        msi = unit["msi"]
        unit["obs"] = msi.obs
        unit["animal"] = str(msi.obs["animal"].iloc[0])
        unit["coords"] = np.asarray(msi.obsm["spatial"])
        unit["W"] = atb.spatial_knn_graph(unit["coords"], k=6)
        unit["measured"] = atb.measured_eval_space(msi, keep)
        unit["tic"] = atb._dense(msi.layers["tic"])[:, keep]
    return units


def score_unit(pred: np.ndarray, unit: dict, features, eval_idx: dict, region: str) -> list[dict]:
    """Rows of (metabolite, metric, value) for one prediction matrix on one unit."""
    rows = []
    meas = unit["measured"]
    for name, j in eval_idx.items():
        for metric, value in atb.metabolite_metrics(pred[:, j], meas[:, j], unit["obs"], unit["W"], region).items():
            rows.append(dict(metabolite=name, metric=metric, value=value))
    in_region = unit["obs"]["region"].astype(str).eq(atb.REGION_LABEL[region]).to_numpy()
    rows.append(dict(metabolite="ALL_median", metric="spearman_section",
                     value=float(np.nanmedian(atb.columnwise_spearman(pred, meas)))))
    rows.append(dict(metabolite="ALL_median", metric="spearman_region",
                     value=float(np.nanmedian(atb.columnwise_spearman(pred[in_region], meas[in_region])))))
    return rows


def protocol_jobs(rep: str):
    """(model name, protocol, latent, source input) evaluated for one representation."""
    if rep.endswith("_rna"):
        base = rep[: -len("_rna")]
        return [(base, "P2", latent, "masked") for latent in ("joint", "st")]
    return [(rep, "P0", "joint", "full"), (rep, "P1", "joint", "masked"), (rep, "P1", "st", "masked")]


#%% baselines

def rna_inputs(joint) -> np.ndarray:
    """Encoder-equivalent RNA input: log1p(SCT_counts) of the model ST features."""
    st = joint.var["type"].eq("ST").to_numpy()
    return np.log1p(atb._dense(joint.X)[:, st].astype(np.float64))


def spatial_block_groups(obs: pd.DataFrame, coords: np.ndarray, n_blocks: int = 10) -> np.ndarray:
    from sklearn.cluster import KMeans

    groups = np.empty(len(obs), dtype=object)
    for section in obs[atb.SECTION_KEY].astype(str).unique():
        m = obs[atb.SECTION_KEY].astype(str).eq(section).to_numpy()
        labels = KMeans(n_clusters=n_blocks, random_state=0, n_init=10).fit_predict(coords[m])
        groups[m] = [f"{section}:{b}" for b in labels]
    return groups


def fit_baselines(fold: dict, readout_template, n_components: int) -> dict:
    """Linear RNA->MSI baselines fitted on the source; returns name -> predict(X_rna)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    src = fold["source_joint"]
    X = rna_inputs(src)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Y = readout_template.scaled_targets(atb._dense(fold["source_msi"].X))

    groups = spatial_block_groups(src.obs, np.asarray(src.obsm["spatial"]))
    cv = list(GroupKFold(n_splits=5).split(Xs, Y, groups))
    ridge = RidgeCV(alphas=np.logspace(-1, 5, 13), cv=cv).fit(Xs, Y)
    pls = PLSRegression(n_components=n_components).fit(Xs, Y)
    pca = PCA(n_components=10, random_state=0).fit(Xs)
    pca_pls = PLSRegression(n_components=n_components).fit(pca.transform(Xs), Y)
    return {
        "ridge_rna": lambda Xn: ridge.predict(scaler.transform(Xn)),
        "pls_rna": lambda Xn: pls.predict(scaler.transform(Xn)),
        "pca10_pls": lambda Xn: pca_pls.predict(pca.transform(scaler.transform(Xn))),
        "_ridge_alpha": float(ridge.alpha_),
    }


#%% per-fold evaluation

def evaluate_fold(region: str, train_animals: tuple, fold_runs: pd.DataFrame, args):
    fold = atb.load_fold(region, list(train_animals))
    Y_src = atb._dense(fold["source_msi"].X)
    template = atb.PLSReadout(args.n_components).fit_targets(Y_src)
    keep = template.keep_
    features = fold["source_msi"].var_names[keep]
    eval_idx = {
        f"msi:{name}": int(np.flatnonzero(features == f"msi:{name}")[0])
        for name in atb.NEUROTRANSMITTERS
        if f"msi:{name}" in set(features)
    }
    dopa_j = eval_idx.get(atb.PRIMARY_METABOLITE)
    units = evaluation_units(fold, region, keep)
    design = "pairwise" if len(train_animals) == 1 else "loao"
    base = dict(region=region, design=design, fold=atb.fold_name(train_animals),
                train_animals="+".join(train_animals))

    records, quality, dopa = [], [], []

    def add(rows, unit, model, protocol, latent, seed, config, pred):
        for row in rows:
            records.append(dict(base, target_animal=unit["animal"], target_section=unit["section"],
                                split=unit["split"], model=model, protocol=protocol, latent=latent,
                                seed=seed, config=config, **row))
        if dopa_j is not None:
            dopa.append(pd.DataFrame(dict(
                obs_name=unit["obs"].index.astype(str), x=unit["coords"][:, 0], y=unit["coords"][:, 1],
                region=region, fold=base["fold"], target_section=unit["section"], split=unit["split"],
                model=model, protocol=protocol, latent=latent, seed=seed,
                predicted=pred[:, dopa_j], measured=unit["measured"][:, dopa_j],
            )))

    for unit in units:
        in_region = unit["obs"]["region"].astype(str).eq(atb.REGION_LABEL[region]).to_numpy()
        for name, j in eval_idx.items():
            q = atb.target_quality(unit["measured"][:, j], unit["tic"][:, j], unit["obs"], unit["W"], region)
            quality.append(dict(base, target_animal=unit["animal"], target_section=unit["section"],
                                split=unit["split"], metabolite=name, **q))

    # linear RNA -> MSI baselines (seed-free)
    baselines = fit_baselines(fold, template, args.n_components)
    X_src = rna_inputs(fold["source_joint"])
    for name in BASELINES:
        for unit in units:
            X = X_src[unit["rows"]] if unit["split"] == "in_sample" else rna_inputs(fold["target_joint"][unit["section"]])
            pred = np.asarray(baselines[name](X), dtype=np.float32)
            add(score_unit(pred, unit, features, eval_idx, region), unit, name, "RNA", "rna", -1, "baseline", pred)
    print(f"[{region} {base['fold']}] baselines done (ridge alpha={baselines['_ridge_alpha']:.3g})")

    for run in fold_runs.itertuples():
        emb = np.load(run.run_dir / "embeddings.npz")
        for rep in run.representations:
            for model, protocol, latent, src_input in protocol_jobs(rep):
                fit_key = f"{rep}|{units[0]['emb_key']}|{src_input}|{latent}"
                Z_fit = emb[fit_key]
                readout = atb.PLSReadout(args.n_components).fit(Z_fit, Y_src)
                for unit in units:
                    if unit["split"] == "in_sample":
                        Z = Z_fit[unit["rows"]]
                    else:
                        Z = emb[f"{rep}|{unit['emb_key']}|masked|{latent}"]
                    pred = readout.predict(Z)
                    add(score_unit(pred, unit, features, eval_idx, region), unit,
                        model, protocol, latent, run.seed, run.config, pred)
        print(f"[{region} {base['fold']}] {run.config} seed{run.seed} done")

    return records, quality, dopa


#%% aggregation

def seed_mean(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(KEYS + ["model"], dropna=False)["value"]
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )


def contrasts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed-matched contrasts per directed pair, then summaries across pairs."""
    df = df[df["split"].eq("transfer")]
    jepa = df[df["seed"] >= 0]
    base = df[df["seed"] < 0]
    combos = jepa[["protocol", "latent"]].drop_duplicates().itertuples(index=False)
    seeds = sorted(jepa["seed"].unique())
    expanded = [base.assign(protocol=p, latent=l, seed=s) for p, l in combos for s in seeds]
    long = pd.concat([jepa] + expanded, ignore_index=True)
    wide = long.pivot_table(index=KEYS + ["seed"], columns="model", values="value", aggfunc="first")

    parts = []
    for name, weights in CONTRASTS.items():
        if not set(weights) <= set(wide.columns):
            continue
        value = sum(coef * wide[model] for model, coef in weights.items())
        parts.append(value.rename("value").reset_index().assign(contrast=name))
    if not parts:
        return pd.DataFrame(), pd.DataFrame()
    per_seed = pd.concat(parts, ignore_index=True).dropna(subset=["value"])
    by_pair = (
        per_seed.groupby(KEYS + ["contrast"])["value"]
        .agg(mean="mean", sd="std", n_seeds="count")
        .reset_index()
    )
    summary = (
        by_pair.groupby(["region", "design", "protocol", "latent", "metabolite", "metric", "contrast"])["mean"]
        .agg(mean_over_pairs="mean", min="min", max="max", n_pairs="count",
             n_positive=lambda v: int((v > 0).sum()))
        .reset_index()
    )
    return by_pair, summary


def init_hash_check(runs: pd.DataFrame) -> pd.DataFrame:
    jepa = runs[runs["kind"].eq("jepa")]
    out = (
        jepa.groupby(["region", "train_animals", "seed"])
        .agg(n_configs=("config", "nunique"), n_hashes=("init_hash", "nunique"),
             configs=("config", lambda v: ",".join(sorted(v))))
        .reset_index()
    )
    out["train_animals"] = out["train_animals"].map("+".join)
    out["identical_init"] = out["n_hashes"].eq(1)
    return out


#%% figures

def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelsize=8, width=0.6)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {path.with_suffix('.png')}")


def _primary(sm: pd.DataFrame, args, metabolite=atb.PRIMARY_METABOLITE, latent="joint") -> pd.DataFrame:
    """Seed-mean primary metric: JEPA models under the primary protocol + RNA baselines."""
    m = sm["metabolite"].eq(metabolite) & sm["metric"].eq(args.primary_metric)
    jepa = m & sm["protocol"].eq(args.primary_protocol) & sm["latent"].eq(latent)
    return sm[jepa | (m & sm["protocol"].eq("RNA"))]


def fig_transfer_matrices(sm, region, args, out: Path):
    d = _primary(sm, args)
    d = d[d["region"].eq(region) & d["design"].eq("pairwise")]
    models = [m for m in MODEL_ORDER if m in set(d["model"])]
    if not models:
        return
    ncols = min(4, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols + 1.0, 2.4 * nrows), squeeze=False)
    for k, (ax, model) in enumerate(zip(axes.ravel(), models)):
        sub = d[d["model"].eq(model)]
        mat = np.full((3, 3), np.nan)
        for row in sub.itertuples():
            i = atb.ANIMALS.index(row.train_animals)
            j = atb.ANIMALS.index(row.target_animal)
            mat[i, j] = row.mean
        shown = mat.copy()
        np.fill_diagonal(shown, np.nan)
        # in-sample diagonal: neutral cell, italic gray value (not a transfer)
        ax.imshow(np.where(np.eye(3, dtype=bool), 1.0, np.nan), cmap=LinearSegmentedColormap.from_list(
            "diag", [GRID, GRID]), vmin=0, vmax=1)
        ax.imshow(shown, cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(3):
            for j in range(3):
                if np.isnan(mat[i, j]):
                    continue
                diag = i == j
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color=INK_2 if diag else INK, style="italic" if diag else "normal")
        ax.set_xticks(range(3), atb.ANIMALS, fontsize=7)
        ax.set_yticks(range(3), atb.ANIMALS if k % ncols == 0 else [], fontsize=7)
        ax.set_title(model, fontsize=9, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.subplots_adjust(wspace=0.08, hspace=0.35)
    axes[0, 0].set_ylabel("trained on", fontsize=8, color=INK_2)
    axes[-1, 0].set_xlabel("evaluated on", fontsize=8, color=INK_2)
    sm_ = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-1, 1))
    cbar = fig.colorbar(sm_, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label(f"Dopamine {args.primary_metric} ({args.primary_protocol}; diagonal = in-sample)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    _save(fig, out / f"{region}_transfer_matrices")


def fig_contrasts(by_pair, region, args, out: Path):
    if by_pair.empty:
        return
    m = (by_pair["region"].eq(region) & by_pair["design"].eq("pairwise")
         & by_pair["metabolite"].eq(atb.PRIMARY_METABOLITE) & by_pair["metric"].eq(args.primary_metric)
         & by_pair["protocol"].eq(args.primary_protocol) & by_pair["latent"].eq("joint"))
    d = by_pair[m]
    names = [c for c in FIGURE_CONTRASTS if c in set(d["contrast"])]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(0.75 * len(names) + 2.2, 3.6))
    for x, name in enumerate(names):
        sub = d[d["contrast"].eq(name)]
        for k, row in enumerate(sub.itertuples()):
            si = atb.ANIMALS.index(row.train_animals)
            ti = atb.ANIMALS.index(row.target_animal)
            offset = (si - 1) * 0.18 + (ti - 1) * 0.05
            ax.errorbar(x + offset, row.mean, yerr=0 if np.isnan(row.sd) else row.sd, fmt=MARKERS[ti],
                        ms=5.5, color=SERIES[si], mec="white", mew=0.6, elinewidth=0.8, capsize=0)
        ax.hlines(sub["mean"].mean(), x - 0.32, x + 0.32, color=INK, linewidth=1.6)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"Δ dopamine {args.primary_metric}", fontsize=9, color=INK)
    ax.set_title(f"{region}: paired contrasts per directed transfer ({args.primary_protocol}, joint latent)",
                 fontsize=9, color=INK)
    _style(ax)
    handles = [plt.Line2D([], [], color=SERIES[i], marker="o", ls="", ms=6, label=f"trained on {a}")
               for i, a in enumerate(atb.ANIMALS)]
    handles += [plt.Line2D([], [], color=INK_2, marker=MARKERS[i], ls="", ms=6, label=f"evaluated on {a}")
                for i, a in enumerate(atb.ANIMALS)]
    handles += [plt.Line2D([], [], color=INK, lw=1.6, label="mean over pairs")]
    ax.legend(handles=handles, fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, out / f"{region}_contrasts")


def fig_protocols(sm, region, args, out: Path):
    m = (sm["region"].eq(region) & sm["design"].eq("pairwise") & sm["split"].eq("transfer")
         & sm["metabolite"].eq(atb.PRIMARY_METABOLITE) & sm["metric"].eq(args.primary_metric)
         & sm["latent"].eq("joint") & sm["protocol"].isin(["P0", "P1", "P2"]))
    d = sm[m]
    models = [mm for mm in ("A", "A30", "B", "C", "C30", "D") if mm in set(d["model"])]
    if not models:
        return
    protocols = ("P0", "P1", "P2")
    labels = {"P0": "P0 manuscript (decoder fit with MSI)", "P1": "P1 zero-MSI decoder fit",
              "P2": "P2 RNA-only student"}
    fig, ax = plt.subplots(figsize=(1.1 * len(models) + 2.6, 3.4))
    for x, model in enumerate(models):
        for p, protocol in enumerate(protocols):
            vals = d[d["model"].eq(model) & d["protocol"].eq(protocol)]["mean"].to_numpy()
            if not len(vals):
                continue
            xs = np.full(len(vals), x + (p - 1) * 0.26)
            ax.scatter(xs, vals, s=22, color=SERIES[p], marker=MARKERS[p], edgecolor="white",
                       linewidth=0.6, zorder=3)
            ax.hlines(vals.mean(), xs[0] - 0.1, xs[0] + 0.1, color=INK, linewidth=1.4, zorder=4)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(models)), models, fontsize=9)
    ax.set_ylabel(f"dopamine {args.primary_metric}", fontsize=9, color=INK)
    ax.set_title(f"{region}: readout protocol (points = directed transfers, bar = mean)", fontsize=9, color=INK)
    _style(ax)
    handles = [plt.Line2D([], [], color=SERIES[p], marker=MARKERS[p], ls="", ms=6, label=labels[pr])
               for p, pr in enumerate(protocols)]
    ax.legend(handles=handles, fontsize=7, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, out / f"{region}_protocols")


def fig_spatial_maps(dopa: pd.DataFrame, region, args, out: Path):
    from scipy.stats import rankdata

    d = dopa[dopa["region"].eq(region) & dopa["split"].eq("transfer") & dopa["fold"].str.count(r"\+").eq(0)]
    d = d[(d["protocol"].eq(args.primary_protocol) & d["latent"].eq("joint")) | d["protocol"].eq("RNA")]
    columns = [m for m in ("A", "B", "C", "D", "E", "ridge_rna") if m in set(d["model"])]
    pairs = d[["fold", "target_section"]].drop_duplicates().sort_values(["fold", "target_section"])
    if pairs.empty or not columns:
        return
    ncols = len(columns) + 1
    fig, axes = plt.subplots(len(pairs), ncols, figsize=(1.7 * ncols, 1.75 * len(pairs)), squeeze=False)
    for r, (fold, section) in enumerate(pairs.itertuples(index=False)):
        sub = d[d["fold"].eq(fold) & d["target_section"].eq(section)]
        ref = sub[sub["model"].eq(columns[0])].groupby("obs_name")[["x", "y", "measured"]].first()
        panels = [("measured", ref["measured"])]
        for model in columns:
            pred = sub[sub["model"].eq(model)].groupby("obs_name")["predicted"].mean()
            panels.append((model, pred.reindex(ref.index)))
        animal = next(a for (a, reg), s in atb.SECTIONS.items() if s == section)
        for c, (label, values) in enumerate(panels):
            ax = axes[r, c]
            pct = rankdata(values.to_numpy()) / len(values)
            ax.scatter(ref["x"], ref["y"], c=pct, cmap=SEQUENTIAL, vmin=0, vmax=1, s=1.6, linewidths=0,
                       rasterized=True)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(label, fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(f"{fold.replace('train-', '')} → {animal}", fontsize=7, color=INK_2)
    sm_ = plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(0, 1))
    cbar = fig.colorbar(sm_, ax=axes, shrink=0.4, pad=0.01)
    cbar.set_label("dopamine percentile within section", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    _save(fig, out / f"{region}_dopamine_spatial_maps")


def fig_metabolite_heatmap(sm, region, args, out: Path):
    m = (sm["region"].eq(region) & sm["split"].eq("transfer") & sm["metric"].eq(args.primary_metric)
         & ((sm["protocol"].eq(args.primary_protocol) & sm["latent"].eq("joint")) | sm["protocol"].eq("RNA")))
    d = sm[m].groupby(["model", "metabolite"])["mean"].mean().unstack()
    if d.empty:
        return
    models = [mm for mm in MODEL_ORDER if mm in d.index]
    mets = [f"msi:{n}" for n in atb.NEUROTRANSMITTERS if f"msi:{n}" in d.columns] + (
        ["ALL_median"] if "ALL_median" in d.columns else [])
    d = d.loc[models, mets]
    fig, ax = plt.subplots(figsize=(0.62 * len(mets) + 1.6, 0.34 * len(models) + 1.2))
    ax.imshow(d.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(mets)), [c.replace("msi:", "") for c in mets], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(models)), models, fontsize=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{region}: mean {args.primary_metric} over directed transfers", fontsize=9, color=INK)
    _save(fig, out / f"{region}_metabolite_heatmap")


def fig_target_quality(quality: pd.DataFrame, out: Path):
    d = quality[quality["metabolite"].eq(atb.PRIMARY_METABOLITE)].drop_duplicates(["target_section"])
    if d.empty:
        return
    d = d.sort_values("target_section")
    metrics = [("moran_measured", "Moran's I (measured)"),
               ("lesion_auroc_measured_region", "intact vs lesioned AUROC"),
               ("frac_nonzero_region", "fraction non-zero (region)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.6))
    for ax, (col, label) in zip(axes, metrics):
        ax.scatter(range(len(d)), d[col], s=28, color=SERIES[0], zorder=3)
        for x, v in enumerate(d[col]):
            ax.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=7, color=INK_2)
        ax.set_xticks(range(len(d)), d["target_section"], rotation=40, ha="right", fontsize=7)
        ax.set_title(label, fontsize=8, color=INK)
        _style(ax)
    fig.suptitle("Measured dopamine signal per section (is it evaluable?)", fontsize=9, color=INK)
    _save(fig, out / "target_quality_dopamine")


#%% main

def main():
    args = parse_args(notebook=is_notebook())
    out_dir = args.out_dir or (args.runs_dir / "evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.runs_dir, args.regions)
    if runs.empty:
        print(f"No finished runs under {args.runs_dir}")
        return
    print(f"[eval] {len(runs)} finished runs in {runs.groupby(['region', 'train_animals']).ngroups} folds")

    records, quality, dopa = [], [], []
    for (region, train_animals), fold_runs in runs.groupby(["region", "train_animals"]):
        r, q, d = evaluate_fold(region, train_animals, fold_runs, args)
        records += r
        quality += q
        dopa += d

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metrics_long.csv", index=False)
    sm = seed_mean(df)
    sm.to_csv(out_dir / "metrics_seedmean.csv", index=False)
    quality_df = pd.DataFrame(quality)
    quality_df.to_csv(out_dir / "target_quality.csv", index=False)
    dopa_df = pd.concat(dopa, ignore_index=True) if dopa else pd.DataFrame()
    if not dopa_df.empty:
        dopa_df.to_csv(out_dir / "dopamine_predictions.csv.gz", index=False)

    hashes = init_hash_check(runs)
    hashes.to_csv(out_dir / "init_hash_check.csv", index=False)
    if not hashes["identical_init"].all():
        print("[WARN] configs with the same fold/seed do not share initial weights:")
        print(hashes[~hashes["identical_init"]].to_string(index=False))

    tm = _primary(sm, args)
    tm = tm[tm["design"].eq("pairwise")][["region", "model", "protocol", "latent", "train_animals",
                                           "target_animal", "split", "mean", "sd", "n_seeds"]]
    tm.to_csv(out_dir / "transfer_matrix.csv", index=False)

    by_pair, summary = contrasts(df)
    by_pair.to_csv(out_dir / "contrasts_by_pair.csv", index=False)
    summary.to_csv(out_dir / "contrasts_summary.csv", index=False)

    if not summary.empty:
        primary = summary[summary["metabolite"].eq(atb.PRIMARY_METABOLITE)
                          & summary["metric"].eq(args.primary_metric)
                          & summary["protocol"].eq(args.primary_protocol)
                          & summary["latent"].eq("joint")]
        print(f"\n[primary] dopamine {args.primary_metric}, {args.primary_protocol}, joint latent "
              "(mean over directed transfers; n_positive / n_pairs):")
        print(primary[["region", "design", "contrast", "mean_over_pairs", "n_positive", "n_pairs"]]
              .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    if not args.no_figures:
        fig_dir = out_dir / "figures"
        for region in sorted(set(df["region"])):
            fig_transfer_matrices(sm, region, args, fig_dir)
            fig_contrasts(by_pair, region, args, fig_dir)
            fig_protocols(sm, region, args, fig_dir)
            if not dopa_df.empty:
                fig_spatial_maps(dopa_df, region, args, fig_dir)
            fig_metabolite_heatmap(sm, region, args, fig_dir)
        fig_target_quality(quality_df[quality_df["split"].eq("transfer")], fig_dir)
    print(f"[eval] wrote {out_dir}")


if __name__ == "__main__":
    main()

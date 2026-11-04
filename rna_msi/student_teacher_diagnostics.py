try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Student vs teacher on held-out tissue: how different, why, and graph-free fixes.
#
# For each finished SpatialJEPA run of the Fig. 5d-f held-out benchmark (configs with a
# graph teacher, e.g. A), compare the graph-free student's latent with the teacher's
# (teacher given the scored section's own graph) on
#   - the source section(s) (in-sample; the student was trained on these spots),
#   - held-out spatial blocks of the source (re-distillation validation only), and
#   - held-out sections (other animals, or the unseen half of the training section),
# all with RNA+MSI input (the Fig. 5d-f condition).
#
# Similarity metrics (candidate C vs teacher T, joint latent):
#   dim_r      mean per-dimension corr(C_d, T_d) (the student is trained per dimension)
#   linear_r2  out-of-fold R^2 of a linear map C -> T (spatial-block CV; rotation-free)
#   knn_overlap mean fraction of shared k=15 neighbours between the C and T kNN graphs
#
# Candidates (graph-free; none uses held-out coordinates, teacher latents or MSI labels
# except where marked):
#   <cfg>                 trained student, as in the benchmark
#   <cfg>_adabn           student with encoder BatchNorm statistics re-estimated on the
#                         scored section's own inputs (test-time, label-free)
#   <cfg>_ridge           ridge from each spot's own input to the source teacher latent,
#                         penalty chosen by spatial-block CV on the source
#   <cfg>_es              student re-distilled from the trained teacher with spatial-block
#                         validation and early stopping (source only)
#   <cfg>_teacher_nograph teacher weights with an identity graph
#   <cfg>_teacher_exprgraph teacher weights with a 6-NN graph in expression space (PCA of
#                         the spot inputs, PCA fitted on the source) instead of tissue
#                         coordinates -- applicable to dissociated cells too
#   <cfg>_exprsmooth      student latent averaged over that expression graph (GCN norm)
#   <cfg>_oracle_ridge    NOT deployable: ridge fitted on the scored section's own teacher
#                         latent (out-of-fold, spatial blocks) -- the ceiling for any
#                         per-spot (graph-free) function of the input
#
# Writes per run: diagnostic_latents.npz (keys `<rep>|<section>|<inputs>|<latent>`, read
# by 9__ for the Fig. 5d-f metrics), student_teacher_diagnostics.csv, redistill_es.json.
# Then `--aggregate` collects them into <runs-dir>/evaluation/.
#
# Env: $PY_MSI

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
load_dotenv(dotenv_path=ENV_FILE_PATH)

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(os.environ["SPATIALJEPA_RNA_MSI"])
sys.path.insert(0, str(SCRIPT_DIR))
import heldout_animals_common as atb

K_NN = 15
N_BLOCKS = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Student-teacher diagnostics on held-out tissue.",
                                     allow_abbrev=False)
    parser.add_argument("--runs-dir", type=Path, default=atb.RUNS_DIR)
    parser.add_argument("--configs", nargs="+", default=["A"])
    parser.add_argument("--designs", nargs="+", default=["pairwise", "within_section"])
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th run starting at i")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--es-max-epoch", type=int, default=1000)
    parser.add_argument("--es-patience", type=int, default=100)
    parser.add_argument("--base-only", action="store_true",
                        help="student vs teacher (+ oracle ridge) only; skip the graph-free adjustments")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def design_of(train_animals) -> str:
    if any("@" in u for u in train_animals):
        return "within_section"
    return "pairwise" if len(train_animals) == 1 else "loao"


#%% latents

def full_latents(model) -> dict:
    """Joint / RNA-expert / MSI-expert latents of all of ``model.X`` in one eval pass."""
    import torch

    model.eval()
    X = torch.as_tensor(atb._dense(model.X), dtype=torch.float32, device=model.device)
    with torch.no_grad():
        H = model.encode(X)
    return {"joint": H["q_mu"].cpu().numpy(), "st": H["st"]["q_mu"].cpu().numpy(),
            "sm": H["sm"]["q_mu"].cpu().numpy()}


def host(state_dict, meta, joint, device, **overrides):
    import spatialmeta as smt
    from spatialjepa.spatialmeta_backbone import rebuild_spatialjepa_model

    data = smt.util._classes.AnnDataJointSMST(joint.copy())
    return rebuild_spatialjepa_model(state_dict, dict(meta, batch_keys=None, **overrides), data, device=device)


def adabn_(model) -> None:
    """Re-estimate encoder BatchNorm statistics on ``model.X`` (full batch, label-free)."""
    import torch
    import torch.nn as nn

    bns = [m for n, m in model.named_modules() if isinstance(m, nn.BatchNorm1d) and n.startswith("encoder")]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None   # cumulative average: one full-batch pass gives the exact statistics
    model.eval()
    for bn in bns:
        bn.train()
    X = torch.as_tensor(atb._dense(model.X), dtype=torch.float32, device=model.device)
    with torch.no_grad():
        model.encode(X)
    model.eval()


def expression_coords(pca_scaler, joint) -> np.ndarray:
    """Source-fitted 20-D PCA of the encoder-equivalent input: stands in for coordinates."""
    scaler, pca = pca_scaler
    return pca.transform(scaler.transform(model_features(joint)))


def fit_expression_pca(source_joint, n_components: int = 20):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = model_features(source_joint)
    scaler = StandardScaler().fit(X)
    return scaler, PCA(n_components=n_components, random_state=0).fit(scaler.transform(X))


def with_coords(joint, coords):
    out = joint.copy()
    out.obsm["spatial"] = np.asarray(coords, float)
    return out


def model_features(joint) -> np.ndarray:
    """The encoder's input transform: log1p for ST, as-is for SM (model space)."""
    X = atb._dense(joint.X).astype(np.float64)
    st = joint.var["type"].eq("ST").to_numpy()
    X[:, st] = np.log1p(X[:, st])
    return X


def spatial_groups(coords: np.ndarray, sections=None, n_blocks: int = N_BLOCKS) -> np.ndarray:
    from sklearn.cluster import KMeans

    sections = np.zeros(len(coords), dtype=object) if sections is None else np.asarray(sections)
    groups = np.empty(len(coords), dtype=object)
    for s in np.unique(sections):
        m = sections == s
        labels = KMeans(n_clusters=n_blocks, random_state=0, n_init=10).fit_predict(coords[m])
        groups[m] = [f"{s}:{b}" for b in labels]
    return groups


#%% similarity

def knn_overlap(C, T, k=K_NN) -> float:
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    def idx(Z):
        Z = StandardScaler().fit_transform(Z)
        return NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z, return_distance=False)[:, 1:]

    a, b = idx(C), idx(T)
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(a, b)]))


def oof_linear_r2(C, T, groups) -> float:
    """Out-of-fold R^2 (pooled over dimensions) of a linear map C -> T."""
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import GroupKFold

    pred = np.zeros_like(T)
    for tr, te in GroupKFold(n_splits=5).split(C, T, groups):
        pred[te] = LinearRegression().fit(C[tr], T[tr]).predict(C[te])
    return float(1 - ((T - pred) ** 2).sum() / ((T - T.mean(0)) ** 2).sum())


def similarity(C, T, groups, ev9) -> dict:
    return {"dim_r": ev9.mean_dim_corr(C, T), "linear_r2": oof_linear_r2(C, T, groups),
            "knn_overlap": knn_overlap(C, T)}


#%% graph-free re-distillation

def ridge_pipeline(X, Y, groups):
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cv = list(GroupKFold(n_splits=5).split(X, Y, groups))
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13), cv=cv)).fit(X, Y)


def oracle_ridge(X, Y, groups) -> np.ndarray:
    """Out-of-fold ridge predictions (inner GCV penalty) of Y from the spot's own input."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros_like(Y)
    for tr, te in GroupKFold(n_splits=5).split(X, Y, groups):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
        pred[te] = model.fit(X[tr], Y[tr]).predict(X[te])
    return pred


def redistill_es(source_joint, teacher_H: dict, groups, seed: int, args, ev9):
    """Student re-distilled from fixed (eval-mode) teacher latents with early stopping.

    Same architecture, optimiser, schedule and loss as SpatialJEPA_trainer (Adam 1e-3,
    wd 1e-4, StepLR 200/0.1, z-scored MSE on q_mu 1.0 / st 0.5 / sm 0.5), but two of the
    ten spatial blocks are held out and the epoch with the best held-out agreement
    (dim_r on the joint latent) is kept.
    """
    import spatialmeta as smt
    import torch
    import torch.optim as optim
    from spatialjepa.spatialmeta_backbone import SpatialJEPA_trainer, spatialJEPA_model

    atb.seed_everything(seed)
    data = smt.util._classes.AnnDataJointSMST(source_joint.copy())
    model = spatialJEPA_model(data, graph_conv=True, graph="identity", device=args.device)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.1)
    targets = {path: torch.as_tensor(teacher_H[name], device=model.device)
               for name, path in (("joint", ("q_mu",)), ("st", ("st", "q_mu")), ("sm", ("sm", "q_mu")))}
    weights = {path: w for path, w in SpatialJEPA_trainer.DISTILL_TARGETS}

    blocks = np.unique(groups)
    rng = np.random.default_rng(seed)
    val_blocks = set(rng.choice(blocks, size=max(1, len(blocks) // 5), replace=False))
    val = np.flatnonzero([g in val_blocks for g in groups])
    train = np.setdiff1d(np.arange(len(groups)), val)
    X_all = torch.as_tensor(atb._dense(model.X), dtype=torch.float32, device=model.device)

    best = (-np.inf, 0, None)
    curve = []
    for epoch in range(1, args.es_max_epoch + 1):
        model.train()
        order = rng.permutation(train)
        batches = [order[i:i + atb.N_PER_BATCH] for i in range(0, len(order), atb.N_PER_BATCH)]
        if len(batches) > 1 and len(batches[-1]) < 2:   # BatchNorm needs >1 sample
            batches[-2] = np.concatenate(batches[-2:]); batches.pop()
        for idx in batches:
            model._current_batch_indices = torch.as_tensor(idx, dtype=torch.long)
            H = model.encode(X_all[idx])
            loss = sum(w * torch.nn.functional.mse_loss(
                SpatialJEPA_trainer.zscore(SpatialJEPA_trainer.get_nested(H, path)),
                SpatialJEPA_trainer.zscore(targets[path][idx]))
                for path, w in weights.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model._current_batch_indices = None
        sched.step()
        model.eval()
        with torch.no_grad():
            Z = model.encode(X_all)["q_mu"].cpu().numpy()
        r_val = ev9.mean_dim_corr(Z[val], teacher_H["joint"][val])
        r_train = ev9.mean_dim_corr(Z[train], teacher_H["joint"][train])
        curve.append((epoch, r_train, r_val))
        if r_val > best[0]:
            best = (r_val, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
        elif epoch - best[1] >= args.es_patience:
            break
    model.load_state_dict(best[2])
    model.eval()
    log = dict(best_epoch=best[1], best_val_r=best[0], stopped_epoch=curve[-1][0],
               final_train_r=curve[-1][1], final_val_r=curve[-1][2], curve=curve,
               n_train=int(len(train)), n_val=int(len(val)))
    return model, val, log


#%% per run

def diagnose_run(run: dict, args, ev9) -> None:
    import torch

    out_npz = run["run_dir"] / "diagnostic_latents.npz"
    out_csv = run["run_dir"] / "student_teacher_diagnostics.csv"
    if out_csv.exists() and out_npz.exists() and not args.overwrite:
        return
    t0 = time.time()
    cfg = run["config"]
    fold = atb.load_fold(run["region"], list(run["train_animals"]))
    src = fold["source_joint"]
    source_key = "+".join(fold["meta"]["source_sections"])
    src_sections = src.obs[atb.SECTION_KEY].astype(str).to_numpy()
    src_groups = spatial_groups(np.asarray(src.obsm["spatial"]), src_sections)

    teacher_ckpt = torch.load(run["run_dir"] / "teacher.pt", map_location=args.device)
    student_ckpt = torch.load(run["run_dir"] / "student.pt", map_location=args.device)
    graph_teacher = teacher_ckpt.get("graph", "spatial") != "identity"
    multi = len(fold["meta"]["source_sections"]) > 1
    t_meta = dict(teacher_ckpt, section_key=atb.SECTION_KEY if multi else None)

    # source (in-sample), RNA+MSI input
    T_src = full_latents(host(teacher_ckpt["state_dict"], t_meta, src, args.device))
    S_src = full_latents(host(student_ckpt["state_dict"], student_ckpt, src, args.device))
    X_src = model_features(src)

    arrays, rows = {}, []

    def record(unit, kind, cand, Z, T, groups, store=True, inputs="paired"):
        rows.append(dict(run_id=run["run_id"], region=run["region"], fold=atb.fold_name(run["train_animals"]),
                         train_animals="+".join(run["train_animals"]), design=design_of(run["train_animals"]),
                         config=cfg, seed=run["seed"], section=unit, relation=kind, candidate=cand,
                         **similarity(Z, T, groups, ev9)))
        if store:
            arrays[f"{cand}|{unit}|{inputs}|joint"] = Z.astype(np.float32)

    for section in fold["meta"]["source_sections"]:
        m = src_sections == section
        record(section, "in_sample", cfg, S_src["joint"][m], T_src["joint"][m], src_groups[m], store=False)

    adjust = graph_teacher and not args.base_only   # adjustments only make sense against a graph teacher
    expr_pca = fit_expression_pca(src)
    if adjust:
        ridge = ridge_pipeline(X_src, T_src["joint"], src_groups)
        es_model, val_idx, es_log = redistill_es(src, T_src, src_groups, run["seed"], args, ev9)
        es_src = full_latents(es_model)
        tng_src = full_latents(host(teacher_ckpt["state_dict"], t_meta, src, args.device, graph="identity"))
        e_src = expression_coords(expr_pca, src)
        teg_src = full_latents(host(teacher_ckpt["state_dict"], dict(t_meta, graph="spatial"),
                                    with_coords(src, e_src), args.device))
        esm_src = np.vstack([ev9.gcn_smooth(S_src["joint"][src_sections == s], e_src[src_sections == s])
                             for s in fold["meta"]["source_sections"]])
        for section in fold["meta"]["source_sections"]:
            m = src_sections == section
            unit_rows = np.flatnonzero(m)
            record(section, "in_sample", f"{cfg}_ridge", ridge.predict(X_src[m]), T_src["joint"][m],
                   src_groups[m], inputs="full")
            record(section, "in_sample", f"{cfg}_es", es_src["joint"][m], T_src["joint"][m], src_groups[m],
                   inputs="full")
            record(section, "in_sample", f"{cfg}_teacher_nograph", tng_src["joint"][m], T_src["joint"][m],
                   src_groups[m], inputs="full")
            record(section, "in_sample", f"{cfg}_teacher_exprgraph", teg_src["joint"][m], T_src["joint"][m],
                   src_groups[m], inputs="full")
            record(section, "in_sample", f"{cfg}_exprsmooth", esm_src[m], T_src["joint"][m],
                   src_groups[m], inputs="full")
        # held-out spatial blocks of the source (re-distillation validation spots)
        vb = np.zeros(len(src_sections), bool); vb[val_idx] = True
        rows.append(dict(run_id=run["run_id"], region=run["region"], fold=atb.fold_name(run["train_animals"]),
                         train_animals="+".join(run["train_animals"]), design=design_of(run["train_animals"]),
                         config=cfg, seed=run["seed"], section=source_key, relation="source_val_blocks",
                         candidate=f"{cfg}_es", dim_r=es_log["best_val_r"], linear_r2=np.nan, knn_overlap=np.nan))
        rows.append(dict(rows[-1], candidate=cfg, dim_r=ev9.mean_dim_corr(S_src["joint"][vb], T_src["joint"][vb])))
        with open(run["run_dir"] / "redistill_es.json", "w") as handle:
            json.dump(es_log, handle)

    source_bases = {atb.base_section(s) for s in fold["meta"]["source_sections"]}
    for target_id in fold["meta"]["target_sections"]:
        joint = atb.paired_target_joint(fold, target_id)
        kind = "same_section" if atb.base_section(target_id) in source_bases else "other_animal"
        groups = spatial_groups(np.asarray(joint.obsm["spatial"]))
        T = full_latents(host(teacher_ckpt["state_dict"], dict(teacher_ckpt, section_key=None), joint, args.device))
        student = host(student_ckpt["state_dict"], student_ckpt, joint, args.device)
        S = full_latents(student)
        record(target_id, kind, cfg, S["joint"], T["joint"], groups, store=False)
        X = model_features(joint)
        oracle = oracle_ridge(X, T["joint"], groups)
        record(target_id, kind, f"{cfg}_oracle_ridge", oracle, T["joint"], groups)
        if adjust:
            adabn_(student)
            record(target_id, kind, f"{cfg}_adabn", full_latents(student)["joint"], T["joint"], groups)
            record(target_id, kind, f"{cfg}_ridge", ridge.predict(X), T["joint"], groups)
            es_t = full_latents(host(es_model.state_dict(), es_model._spatialjepa_meta, joint, args.device))
            record(target_id, kind, f"{cfg}_es", es_t["joint"], T["joint"], groups)
            tng = full_latents(host(teacher_ckpt["state_dict"], dict(teacher_ckpt, section_key=None), joint,
                                    args.device, graph="identity"))
            record(target_id, kind, f"{cfg}_teacher_nograph", tng["joint"], T["joint"], groups)
            e_t = expression_coords(expr_pca, joint)
            teg = full_latents(host(teacher_ckpt["state_dict"], dict(teacher_ckpt, section_key=None, graph="spatial"),
                                    with_coords(joint, e_t), args.device))
            record(target_id, kind, f"{cfg}_teacher_exprgraph", teg["joint"], T["joint"], groups)
            record(target_id, kind, f"{cfg}_exprsmooth", ev9.gcn_smooth(S["joint"], e_t), T["joint"], groups)

    np.savez_compressed(out_npz, **arrays)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[diag] {run['run_id']}: {len(rows)} rows in {time.time() - t0:.0f}s")


def finished_runs(runs_dir: Path, configs, designs) -> list[dict]:
    runs = []
    for done in sorted(runs_dir.glob("*/train-*/*/seed*/DONE")):
        with open(done.parent / "run.json") as handle:
            meta = json.load(handle)
        if meta["config"] in configs and design_of(meta["train_animals"]) in designs:
            runs.append(dict(meta, run_dir=done.parent))
    return runs


def aggregate(args) -> None:
    frames = [pd.read_csv(p) for p in sorted(args.runs_dir.glob("*/train-*/*/seed*/student_teacher_diagnostics.csv"))]
    if not frames:
        print("no diagnostics found"); return
    df = pd.concat(frames, ignore_index=True)
    out = args.runs_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "student_teacher_diagnostics_long.csv", index=False)
    summary = (df.groupby(["design", "config", "relation", "candidate"])[["dim_r", "linear_r2", "knn_overlap"]]
               .agg(["mean", "std", "count"]))
    summary.to_csv(out / "student_teacher_diagnostics_summary.csv")
    pd.set_option("display.width", 220)
    print(df.groupby(["design", "config", "relation", "candidate"])[["dim_r", "linear_r2", "knn_overlap"]]
          .mean().round(3).to_string())
    es = []
    for p in sorted(args.runs_dir.glob("*/train-*/*/seed*/redistill_es.json")):
        log = json.load(open(p))
        es.append(dict(run=str(p.parent.relative_to(args.runs_dir)),
                       **{k: v for k, v in log.items() if k != "curve"}))
    if es:
        es = pd.DataFrame(es)
        es.to_csv(out / "redistill_es_summary.csv", index=False)
        print("\nre-distillation early stopping:")
        print(es[["best_epoch", "best_val_r", "final_train_r", "final_val_r"]].describe().round(3).to_string())


def main():
    args = parse_args()
    if args.aggregate:
        aggregate(args); return
    ev9 = _load_module("eval9", "heldout_animals_dopamine_structure.py")
    i, n = (int(x) for x in args.shard.split("/"))
    runs = finished_runs(args.runs_dir, args.configs, args.designs)[i::n]
    print(f"[diag] {len(runs)} run(s) in shard {i}/{n}")
    for run in runs:
        diagnose_run(run, args, ev9)


if __name__ == "__main__":
    main()

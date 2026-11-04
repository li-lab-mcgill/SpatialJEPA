try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
#%% Cross-animal SpatialJEPA ablation benchmark: train one (fold, config, seed).
#
# Trains SpatialJEPA variants on one mouse's FMP-10 section (or two, for the optional
# leave-one-animal-out mode) and encodes the other animals' sections from RNA alone.
# PLS readouts and metrics are computed afterwards by heldout_animals_evaluate.py.
#
# Configs (heldout_animals_common.CONFIGS):
#   A / A30   spatial-kNN teacher (k=6 / k=30), added ST->SM loss
#   B         identity-graph teacher, added ST->SM loss      (teacher = control E)
#   C / C30   spatial-kNN teacher (k=6 / k=30), no ST->SM loss
#   D         identity-graph teacher, no ST->SM loss         (teacher = control F)
#   A_perm    kNN graph with node labels permuted within section (degree-matched)
#   A_induced A with the legacy induced-subgraph mini-batches (manuscript behaviour)
#   A_early   A with the teacher early-stopped: a first pass trains the teacher alone on
#             8 of 10 spatial blocks and scores the other 2 after every epoch; the run
#             then trains teacher + students on every spot for the best epoch count, and
#             the students keep distilling from the frozen teacher for the rest of the
#             --max-epoch schedule (same student budget as A)
#   A_wd      A with teacher AdamW weight decay 1.0 (instead of 1e-6), full schedule
#   G         stock SpatialMETA ConditionalVAESTSM (FC encoders, no teacher)
#
# Every SpatialJEPA run distils two students from the same teacher batches: `student`
# (paired RNA+MSI input, as in the manuscript) and `student_rna` (MSI input zeroed,
# distilling the joint and RNA-expert latents only).
#
# Outputs (${SPATIALJEPA_OUT}/spatialjepa_benchmark/animal_transfer/<region>/train-<animals>/
# <config>/seed<k>/, or under $ANIMAL_TRANSFER_RUNS_DIR): checkpoints, losses.json/.png,
# embeddings.npz, run.json, DONE.
#
# Env: $PY_MSI
# Exit codes: 0 ok (or already DONE), 3 no usable GPU (node problem, not a run failure).

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE_PATH = os.environ["SPATIALJEPA_CONFIG"]
load_dotenv(dotenv_path=ENV_FILE_PATH)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(os.environ["SPATIALJEPA_RNA_MSI"])
sys.path.insert(0, str(SCRIPT_DIR))
import heldout_animals_common as atb

GPU_UNAVAILABLE_EXIT = 3


def is_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def parse_args(notebook: bool = False):
    parser = argparse.ArgumentParser(
        description="Train one cross-animal SpatialJEPA ablation run.",
        allow_abbrev=False,
    )
    parser.add_argument("--train-animals", nargs="+", default=["mPD1"],
                        help=f"training units: {', '.join(atb.ANIMALS)}, or one half of an "
                             "animal's section as e.g. mPD3@h0 (the other half is held out)")
    parser.add_argument("--region", choices=("striatum", "sn"), default="striatum")
    parser.add_argument("--config", choices=tuple(atb.CONFIGS), default="A")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-neighbors", type=int, default=None, help="override the config's teacher k")
    parser.add_argument("--neighborhood", choices=("exact", "induced"), default=None,
                        help="override the config's teacher mini-batch neighbourhood")
    # manuscript mouse schedule (main.tex Methods; train_rna_msi.py)
    parser.add_argument("--max-epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-step", type=int, default=200)
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--teacher-weight-decay", type=float, default=None,
                        help="override the teacher's AdamW weight decay (default 1e-6)")
    parser.add_argument("--teacher-dropout", type=float, default=None,
                        help="override the teacher encoders' dropout (default 0.1)")
    parser.add_argument("--early-stop", action="store_true", help="early-stop the teacher (as A_early)")
    parser.add_argument("--es-val-blocks", type=int, default=2, help="held-out blocks (of 10) per section")
    parser.add_argument("--es-patience", type=int, default=100)
    parser.add_argument("--select-only", action="store_true",
                        help="run the early-stopping pass, write early_stopping.json and exit")
    parser.add_argument("--no-rna-student", action="store_true")
    parser.add_argument("--panel-top", type=int, default=atb.N_TARGET_TOP)
    parser.add_argument("--no-target-panel", action="store_true")
    parser.add_argument("--prepare-only", action="store_true",
                        help="build the fold cache (CPU) and exit")
    parser.add_argument("--device", default="cuda:0")
    if notebook:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def gpu_sanity_check(device: str) -> str:
    import torch

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        x = torch.ones(1024, device=device)
        assert float((x * 2).sum().item()) == 2048.0
        return torch.cuda.get_device_name(torch.device(device))
    except Exception as exc:
        print(f"[GPU] unusable on {socket.gethostname()}: {exc}", file=sys.stderr)
        sys.exit(GPU_UNAVAILABLE_EXIT)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(SCRIPT_DIR), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def plot_losses(loss_dict: dict, path: Path) -> None:
    keys = [k for k, v in loss_dict.items() if len(v)]
    ncols = 4
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, key in zip(axes, keys):
        ax.plot(loss_dict[key])
        ax.set_title(key.replace("epoch_", "").replace("_list", ""), fontsize=9)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


#%% early-stopping pass for the teacher

# ConditionalVAESTSM_STtoSM.fit(mode=...) weights; KL at its post-warm-up value
SINGLE_MODE_WEIGHTS = dict(reconstruction_loss_st=5, reconstruction_loss_sm=1, reconstruction_loss_st_corr=5,
                           reconstruction_loss_sm_corr=1, kldiv_loss=0.5)
MULTI_MODE_WEIGHTS = dict(reconstruction_loss_st=8, reconstruction_loss_sm=2, reconstruction_loss_st_corr=8,
                          reconstruction_loss_sm_corr=2, kldiv_loss=1, mmd_loss=10)
MODE_WEIGHTS = {"single": SINGLE_MODE_WEIGHTS, "multi": MULTI_MODE_WEIGHTS}


def select_teacher_epochs(fold, cfg, args, multi_section: bool) -> dict:
    """Pick the teacher's epoch count on held-out spatial blocks of the source.

    ``args.es_val_blocks`` of 10 KMeans blocks per source section are held out. The
    teacher alone (no students) trains on the other spots, with a graph over those spots
    only, under the run's schedule. After every epoch its weights go into a twin hosted
    on the held-out spots (their own graph), scored with the training objective at the
    final KL weight (eval mode, fixed reparameterisation noise). Stops after
    ``args.es_patience`` epochs without improvement. ``cfg["mode"]`` / ``cfg["batch_keys"]``
    (defaults "single" / None) select the multi-section objective (decoder batch
    conditioning + MMD) used by train_rna_msi.py for the human source.
    """
    import spatialmeta as smt
    import torch
    from spatialjepa.spatialmeta_backbone import _scheduled_adamw_epochs, spatialJEPA_model

    mode, batch_keys = cfg.get("mode", "single"), cfg.get("batch_keys")
    src = fold["source_joint"]
    sections = src.obs[atb.SECTION_KEY].astype(str).to_numpy()
    groups = atb.spatial_blocks(np.asarray(src.obsm["spatial"]), sections)
    val = atb.validation_blocks(groups, sections, args.seed, n_val=args.es_val_blocks)
    if (~val).sum() % atb.N_PER_BATCH == 1:   # BatchNorm cannot take a one-row batch
        val[np.flatnonzero(~val)[-1]] = True

    def build(adata):
        return spatialJEPA_model(
            smt.util._classes.AnnDataJointSMST(adata),
            graph_conv=True,
            graph=cfg["graph"],
            n_neighbors=cfg["n_neighbors"],
            graph_seed=args.seed,
            neighborhood=cfg["neighborhood"],
            section_key=atb.SECTION_KEY if multi_section else None,
            device=args.device,
            dropout_rate=cfg["dropout_rate"],
            batch_keys=batch_keys,
        )

    atb.seed_everything(args.seed)
    teacher = build(src[~val].copy())
    twin = build(src[val].copy())
    X_val = torch.as_tensor(atb._dense(twin.X), dtype=torch.float32, device=twin.device)
    batch_val = None if twin.batch_codes is None else torch.hstack(
        [torch.as_tensor(code, dtype=torch.long).unsqueeze(1) for code in twin.batch_codes]).to(twin.device)
    weights = dict(MODE_WEIGHTS[mode], reconstruction_loss_sm_cross=cfg["sm_cross_weight"])
    devices = [torch.device(args.device)] if torch.device(args.device).type == "cuda" else []
    curve, best = [], dict(val_loss=np.inf, epoch=0)

    def score(epoch: int) -> bool:
        twin.load_state_dict({k: v for k, v in teacher.state_dict().items() if not k.endswith("edge_index")},
                             strict=False)
        twin.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(0)
            _, _, L = twin.forward(X_val, batch_index=batch_val, reduction="sum")
        terms = {k: float(L[k].mean()) for k in weights}
        total = sum(w * terms[k] for k, w in weights.items())
        curve.append(dict(epoch=epoch, val_loss=total, **terms))
        if total < best["val_loss"]:
            best.update(val_loss=total, epoch=epoch)
        return epoch - best["epoch"] >= args.es_patience

    t0 = time.time()
    steps = len(teacher.as_dataloader(batch_size=atb.N_PER_BATCH, shuffle=True))
    with _scheduled_adamw_epochs(label="teacher_es", steps_per_epoch=steps, step_size=args.lr_step,
                                 gamma=args.lr_gamma):
        train_loss = teacher.fit(
            max_epoch=args.max_epoch,
            n_per_batch=atb.N_PER_BATCH,
            lr=args.lr,
            mode=mode,
            sm_cross_weight=cfg["sm_cross_weight"],
            weight_decay=cfg["teacher_weight_decay"],
            epoch_callback=score,
        )
    return dict(
        best_epoch=best["epoch"],
        best_val_loss=best["val_loss"],
        stopped_epoch=curve[-1]["epoch"],
        patience=args.es_patience,
        n_train=int((~val).sum()),
        n_val=int(val.sum()),
        val_blocks=sorted(set(groups[val])),
        seconds=time.time() - t0,
        curve=curve,
        train_total_loss=[float(x) for x in train_loss["epoch_total_loss_list"]],
    )


#%% train SpatialJEPA (teacher + students) or stock SpatialMETA

def train_jepa(source_joint, cfg, args, multi_section: bool, max_epoch: int, cap_kl_warmup: bool = True,
               student_epochs=None):
    import torch.optim as optim
    from spatialjepa.spatialmeta_backbone import (
        SpatialJEPA_trainer,
        StudentSpec,
        _scheduled_adamw_epochs,
        spatialJEPA_model,
    )

    section_key = atb.SECTION_KEY if multi_section else None
    # identical construction order for every config: with the same seed, teacher and
    # students start from the same weights (the graph does not change parameter shapes)
    teacher = spatialJEPA_model(
        source_joint,
        graph_conv=True,
        graph=cfg["graph"],
        n_neighbors=cfg["n_neighbors"],
        graph_seed=args.seed,
        neighborhood=cfg["neighborhood"],
        section_key=section_key,
        device=args.device,
        dropout_rate=cfg["dropout_rate"],
    )
    student = spatialJEPA_model(source_joint, graph_conv=True, graph="identity", device=args.device)
    models = {"teacher": teacher, "student": student}
    extra = []
    if not args.no_rna_student:
        student_rna = spatialJEPA_model(source_joint, graph_conv=True, graph="identity", device=args.device)
        models["student_rna"] = student_rna
        extra.append(
            StudentSpec(
                name="student_rna",
                model=student_rna,
                distill_targets=atb.RNA_ONLY_DISTILL_TARGETS,
                mask_sm_input=True,
            )
        )
    init_hash = atb.parameter_hash(*models.values())

    trainer = SpatialJEPA_trainer(teacher, student, n_per_batch=atb.N_PER_BATCH, extra_students=extra)
    trainer.student_lr_scheduler = optim.lr_scheduler.StepLR(
        trainer.student_optimizer, step_size=args.lr_step, gamma=args.lr_gamma
    )
    for spec in extra:
        spec.lr_scheduler = optim.lr_scheduler.StepLR(
            spec.optimizer, step_size=args.lr_step, gamma=args.lr_gamma
        )
    with _scheduled_adamw_epochs(
        label="teacher",
        steps_per_epoch=trainer.batches_per_epoch,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    ) as teacher_lr_states:
        loss_dict = trainer.fit_teacher(
            max_epoch=max_epoch,
            lr=args.lr,
            mode="single",
            sm_cross_weight=cfg["sm_cross_weight"],
            weight_decay=cfg["teacher_weight_decay"],
            cap_kl_warmup=cap_kl_warmup,
        )
    loss_dict["epoch_teacher_lr_list"] = teacher_lr_states[0].epoch_lr_list if teacher_lr_states else []
    # an early-stopped teacher stays frozen while the students finish their schedule
    # (train-mode targets, as in the batch scheme; teacher BatchNorm stats untouched)
    for _ in range((student_epochs or max_epoch) - max_epoch):
        trainer.distill_epoch_end()
    # per-batch teacher latents are not needed downstream
    trainer.batch_H.clear()
    trainer.epoch_H.clear()
    return models, loss_dict, init_hash


def train_stock(source_joint, args):
    import spatialmeta as smt
    from spatialjepa.spatialmeta_backbone import _scheduled_adamw_epochs

    model = smt.model.ConditionalVAESTSM(
        source_joint,
        device=args.device,
        reconstruction_method_sm="g",
        reconstruction_method_st="zinb",
        batch_keys=None,
    )
    init_hash = atb.parameter_hash(model)
    steps = len(model.as_dataloader(batch_size=atb.N_PER_BATCH, shuffle=True))
    with _scheduled_adamw_epochs(
        label="G", steps_per_epoch=steps, step_size=args.lr_step, gamma=args.lr_gamma
    ) as lr_states:
        loss_dict = model.fit(
            max_epoch=args.max_epoch, n_per_batch=atb.N_PER_BATCH, lr=args.lr, mode="single"
        )
    loss_dict["epoch_lr_list"] = lr_states[0].epoch_lr_list if lr_states else []
    return {"model": model}, loss_dict, init_hash


def save_models(models: dict, cfg: dict, out_dir: Path) -> None:
    import torch
    from spatialjepa.spatialmeta_backbone import save_spatialjepa_model

    for name, model in models.items():
        meta = getattr(model, "_spatialjepa_meta", None)
        if meta is not None:
            save_spatialjepa_model(model, out_dir / f"{name}.pt", full_graph=meta["graph"] != "identity")
        else:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_class": "spatialmeta.model.ConditionalVAESTSM",
                    "graph_conv": False,
                    "reconstruction_method_sm": model.reconstruction_method_sm,
                    "reconstruction_method_st": model.reconstruction_method_st,
                    "hidden_stacks": model.hidden_stacks,
                    "n_latent": model.n_latent,
                },
                out_dir / f"{name}.pt",
            )


def encode_all(models: dict, config: str, fold: dict, args) -> dict:
    """Latents of every graph-free representation on source and target sections.

    Source sections are encoded with real MSI ("full", protocol P0's decoder fit) and
    with MSI zeroed ("masked"); targets only ever with MSI zeroed.
    """
    arrays = {}
    source_id = "+".join(fold["meta"]["source_sections"])
    arrays[f"obs_names|{source_id}"] = fold["source_joint"].obs_names.to_numpy(dtype=str)
    for target_id, joint in fold["target_joint"].items():
        arrays[f"obs_names|{target_id}"] = joint.obs_names.to_numpy(dtype=str)

    reps = atb.representation_names(config, rna_student="student_rna" in models)
    for rep, model_key in reps.items():
        model = models[model_key]
        jobs = [(source_id, fold["source_joint"], "full"), (source_id, fold["source_joint"], "masked")]
        jobs += [(tid, joint, "masked") for tid, joint in fold["target_joint"].items()]
        for section_id, joint, inputs in jobs:
            latents = atb.encode_latents(model, joint, mask_sm=inputs == "masked", device=args.device)
            for latent, Z in latents.items():
                if not np.isfinite(Z).all():
                    raise FloatingPointError(f"non-finite {rep} latents on {section_id} ({inputs})")
                arrays[f"{rep}|{section_id}|{inputs}|{latent}"] = Z
    return arrays


#%% main

def main():
    args = parse_args(notebook=is_notebook())
    cfg = dict(atb.CONFIGS[args.config])
    if args.n_neighbors is not None:
        cfg["n_neighbors"] = args.n_neighbors
    if args.neighborhood is not None:
        cfg["neighborhood"] = args.neighborhood
    if cfg["kind"] == "jepa":
        cfg.setdefault("teacher_weight_decay", 1e-6)   # fit_core default
        cfg.setdefault("dropout_rate", 0.1)            # SpatialMETA default
        cfg.setdefault("early_stop", False)
        if args.teacher_weight_decay is not None:
            cfg["teacher_weight_decay"] = args.teacher_weight_decay
        if args.teacher_dropout is not None:
            cfg["dropout_rate"] = args.teacher_dropout
        cfg["early_stop"] = cfg["early_stop"] or args.early_stop
    train_animals = list(dict.fromkeys(args.train_animals))
    for unit in train_animals:
        atb.parse_train_unit(unit)   # validate early

    data_dir = atb.ensure_fold_cache(
        args.region,
        train_animals,
        use_target_panel=not args.no_target_panel,
        n_target_top=args.panel_top,
    )
    if args.prepare_only:
        print(f"[fold] cache ready: {data_dir}")
        return

    out_dir = atb.run_dir(args.region, train_animals, args.config, args.seed)
    if (out_dir / "DONE").exists():
        print(f"[skip] {out_dir} is already DONE")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = gpu_sanity_check(args.device)
    import spatialmeta as smt
    import torch

    fold = atb.load_fold(args.region, train_animals)
    source_joint = smt.util._classes.AnnDataJointSMST(fold["source_joint"])
    run_name = atb.run_id(args.region, train_animals, args.config, args.seed)
    print(f"[run] {run_name} on {socket.gethostname()} ({gpu_name}); "
          f"source {source_joint.shape}, targets {list(fold['target_joint'])}")

    multi_section = len(train_animals) > 1
    max_epoch, es_log = args.max_epoch, None
    if cfg.get("early_stop"):
        es_settings = dict(max_epoch=args.max_epoch, lr=args.lr, lr_step=args.lr_step, lr_gamma=args.lr_gamma,
                           patience=args.es_patience, val_blocks=args.es_val_blocks,
                           teacher_weight_decay=cfg["teacher_weight_decay"], dropout_rate=cfg["dropout_rate"])
        es_path = out_dir / "early_stopping.json"
        if es_path.exists():   # selection pass of an interrupted run with the same settings
            with open(es_path) as handle:
                es_log = json.load(handle)
            if es_log.get("settings", es_settings) != es_settings or es_log["patience"] != args.es_patience:
                es_log = None
        if es_log is None:
            es_log = dict(select_teacher_epochs(fold, cfg, args, multi_section), settings=es_settings)
            with open(es_path, "w") as handle:
                json.dump(es_log, handle, default=float)
        print(f"[early-stop] best epoch {es_log['best_epoch']} (stopped at {es_log['stopped_epoch']}; "
              f"{es_log['n_train']} train / {es_log['n_val']} val spots, {es_log['seconds'] / 60:.1f} min)")
        if args.select_only:
            return
        max_epoch = es_log["best_epoch"]

    atb.seed_everything(args.seed)
    t0 = time.time()
    if cfg["kind"] == "stock":
        models, loss_dict, init_hash = train_stock(source_joint, args)
    else:
        # an early-stopped refit replays the first max_epoch epochs of the full schedule
        models, loss_dict, init_hash = train_jepa(
            source_joint, cfg, args, multi_section, max_epoch=max_epoch, cap_kl_warmup=es_log is None,
            student_epochs=args.max_epoch,
        )
    train_seconds = time.time() - t0

    save_models(models, cfg, out_dir)
    with open(out_dir / "losses.json", "w") as handle:
        json.dump({k: [float(x) for x in v] for k, v in loss_dict.items()}, handle)
    plot_losses(loss_dict, out_dir / "losses.png")

    arrays = encode_all(models, args.config, fold, args)
    np.savez_compressed(out_dir / "embeddings.npz", **arrays)

    run_meta = dict(
        run_id=run_name,
        region=args.region,
        train_animals=train_animals,
        config=args.config,
        config_spec=cfg,
        seed=args.seed,
        representations=atb.representation_names(args.config, rna_student="student_rna" in models),
        init_hash=init_hash,
        max_epoch=max_epoch,
        student_epochs=args.max_epoch if cfg["kind"] == "jepa" else None,
        early_stopping=None if es_log is None else {k: v for k, v in es_log.items()
                                                    if k not in ("curve", "train_total_loss")},
        lr=args.lr,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        n_per_batch=atb.N_PER_BATCH,
        rna_student=not args.no_rna_student,
        fold=fold["meta"],
        train_seconds=train_seconds,
        seconds_per_epoch=train_seconds / max(max_epoch, 1),
        host=socket.gethostname(),
        gpu=gpu_name,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        git_sha=git_sha(),
        torch_version=torch.__version__,
    )
    with open(out_dir / "run.json", "w") as handle:
        json.dump(run_meta, handle, indent=2, default=str)
    (out_dir / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[done] {run_name}: {train_seconds / 60:.1f} min training -> {out_dir}")


if __name__ == "__main__":
    main()

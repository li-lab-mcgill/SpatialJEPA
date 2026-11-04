"""How well does the teacher itself generalise to unseen tissue?

Part 1 -- embedding consistency (stored latents, RNA+MSI input, joint latent). For each
scored spot set (a whole section, or one half of a section) collect every encoding of
those spots by a model family, from every training unit and seed, labelled by what the
model was trained on:
  in_sample        trained on these spots (pairwise: the whole section; split: this half)
  in_sample_full   (halves only) trained on the whole section, restricted to this half
  heldout_half     trained on the other half of the same section
  heldout_animal   trained on another animal's whole section
For every pair of encodings: symmetric out-of-fold linear R^2 (mean of both directions,
5-fold spatial-block CV) and k=15 kNN overlap, grouped by the pair's relations.

Part 2 -- reconstruction (decoders driven by the latent mean): NLL under the model's own
likelihoods and per-feature correlation, for RNA and MSI from the joint latent and MSI
from the RNA-expert latent (the ST->SM path), on the training spots and on held-out
halves / animals: the train/test gap of the teacher as a VAE.

Families: A_teacher (spatial 6-NN teacher, section graph at inference), E (identity
teacher = B's teacher), G (stock SpatialMETA), A (graph-free student, part 1 only), and
A_early_teacher / A_early (the same with the teacher early-stopped on held-out blocks) and
A_wd_teacher / A_wd (teacher with AdamW weight decay 1.0).

  python teacher_consistency.py [--part 1|2|both] [--device cuda:1]
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
import importlib.util
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SMA = Path(os.environ["SPATIALJEPA_RNA_MSI"])
sys.path.insert(0, str(SMA))
os.environ.setdefault("ANIMAL_TRANSFER_PANEL_TOP", "2000")
import heldout_animals_common as atb  # noqa: E402

BENCH = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/spatialjepa_benchmark")
ROOTS = {"pairwise": BENCH / "animal_transfer_p2000", "within_section": BENCH / "animal_transfer_p2000_split"}
OUT = ROOTS["pairwise"] / "evaluation"
ANIMALS = ("mPD1", "mPD3", "mPD4")
SEEDS = (0, 1, 2)
# family: (config directory, in-sample latent file)
FAMILIES = {"A_teacher": ("A", "teacher_insample.npz"), "E": ("B", "embeddings.npz"),
            "G": ("G", "embeddings.npz"), "A": ("A", "embeddings.npz"),
            "A_early_teacher": ("A_early", "teacher_insample.npz"), "A_early": ("A_early", "embeddings.npz"),
            "A_wd_teacher": ("A_wd", "teacher_insample.npz"), "A_wd": ("A_wd", "embeddings.npz")}
RECON_FAMILIES = {"A_teacher": ("A", "teacher.pt"), "E": ("B", "teacher.pt"), "G": ("G", "model.pt"),
                  "A_early_teacher": ("A_early", "teacher.pt"), "A_wd_teacher": ("A_wd", "teacher.pt")}


def load(name, f):
    spec = importlib.util.spec_from_file_location(name, SMA / f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


d10 = load("d10", "student_teacher_diagnostics.py")


def section_of(animal):
    return atb.SECTIONS[(animal, "striatum")]


def fold_dir(design, unit):
    return ROOTS[design] / "striatum" / f"train-{unit}"


_obs_cache = {}


def obs_names(design, unit, which):
    """Spot ids of a cached fold's source ('source') or a target section id."""
    key = (design, unit, which)
    if key not in _obs_cache:
        import anndata as ad

        data = fold_dir(design, unit) / "data"
        path = data / ("source_joint.h5ad" if which == "source" else f"target_{which}_joint.h5ad")
        a = ad.read_h5ad(path, backed="r")
        coords = np.asarray(a.obsm["spatial"], float) if "spatial" in a.obsm else None
        _obs_cache[key] = (pd.Index(a.obs_names), coords)
        a.file.close()
    return _obs_cache[key]


def latent(design, unit, family, seed, set_id, inputs):
    cfg, insample_file = FAMILIES[family]
    run = fold_dir(design, unit) / cfg / f"seed{seed}"
    path = run / (insample_file if inputs == "full" else "heldout_paired.npz")
    return np.load(path)[f"{family}|{set_id}|{inputs}|joint"].astype(np.float64)


#%% part 1: embedding consistency

def encodings_full(family, animal):
    """All encodings of one whole section (pairwise trainings)."""
    s = section_of(animal)
    ids, coords = obs_names("pairwise", animal, "source")
    encs = []
    for unit in ANIMALS:
        for seed in SEEDS:
            if unit == animal:
                Z, z_ids = latent("pairwise", unit, family, seed, s, "full"), ids
                rel = "in_sample"
            else:
                Z, z_ids = latent("pairwise", unit, family, seed, s, "paired"), obs_names("pairwise", unit, s)[0]
                rel = "heldout_animal"
            encs.append(dict(rel=rel, unit=unit, seed=seed, Z=align(Z, z_ids, ids)))
    return ids, coords, encs


def encodings_half(family, animal, half):
    """All encodings of one half section (split + pairwise trainings)."""
    s = section_of(animal)
    unit_here, unit_other = f"{animal}@h{half}", f"{animal}@h{1 - half}"
    half_id = f"{s}@h{half}"
    ids, coords = obs_names("within_section", unit_here, "source")
    full_ids = obs_names("pairwise", animal, "source")[0]
    encs = []
    for seed in SEEDS:
        encs.append(dict(rel="in_sample", unit=unit_here, seed=seed,
                         Z=latent("within_section", unit_here, family, seed, half_id, "full")))
        encs.append(dict(rel="heldout_half", unit=unit_other, seed=seed,
                         Z=align(latent("within_section", unit_other, family, seed, half_id, "paired"),
                                 obs_names("within_section", unit_other, half_id)[0], ids)))
        encs.append(dict(rel="in_sample_full", unit=animal, seed=seed,
                         Z=align(latent("pairwise", animal, family, seed, s, "full"), full_ids, ids)))
        for other in ANIMALS:
            if other != animal:
                encs.append(dict(rel="heldout_animal", unit=other, seed=seed,
                                 Z=align(latent("pairwise", other, family, seed, s, "paired"),
                                         obs_names("pairwise", other, s)[0], ids)))
    return ids, coords, encs


def align(Z, z_ids, ids):
    pos = pd.Index(z_ids).get_indexer(ids)
    if (pos < 0).any():
        raise KeyError(f"{(pos < 0).sum()} spots missing")
    return Z[pos]


def pair_rows(family, set_id, kind, coords, encs):
    groups = d10.spatial_groups(coords)
    rows = []
    for a, b in itertools.combinations(encs, 2):
        rels = sorted([a["rel"], b["rel"]])
        same_unit = a["unit"] == b["unit"]
        rows.append(dict(family=family, set_id=set_id, kind=kind, pair=" x ".join(rels),
                         same_unit=same_unit, unit_a=a["unit"], seed_a=a["seed"], unit_b=b["unit"], seed_b=b["seed"],
                         linear_r2=0.5 * (d10.oof_linear_r2(a["Z"], b["Z"], groups)
                                          + d10.oof_linear_r2(b["Z"], a["Z"], groups)),
                         knn_overlap=d10.knn_overlap(a["Z"], b["Z"])))
    return rows


def part1():
    rows = []
    for family in FAMILIES:
        for animal in ANIMALS:
            _, coords, encs = encodings_full(family, animal)
            rows += pair_rows(family, section_of(animal), "full", coords, encs)
            for half in (0, 1):
                _, coords, encs = encodings_half(family, animal, half)
                rows += pair_rows(family, f"{section_of(animal)}@h{half}", "half", coords, encs)
            print(f"[part1] {family} {animal}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "teacher_consistency_pairs.csv", index=False)
    # weight scored spot sets equally: mean within set first
    per_set = df.groupby(["family", "kind", "pair", "same_unit", "set_id"])[["linear_r2", "knn_overlap"]].mean()
    summary = per_set.groupby(["family", "kind", "pair", "same_unit"]).agg(["mean", "min", "max", "count"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "teacher_consistency_summary.csv", index=False)
    print(summary.round(3).to_string())


#%% part 2: reconstruction

def host_any(ckpt, joint, device):
    import spatialmeta as smt

    if ckpt.get("model_class") == "spatialmeta.model.ConditionalVAESTSM":
        data = smt.util._classes.AnnDataJointSMST(joint.copy())
        model = smt.model.ConditionalVAESTSM(data, device=device, reconstruction_method_sm="g",
                                             reconstruction_method_st="zinb")
        model.load_state_dict(ckpt["state_dict"])
        return model
    return d10.host(ckpt["state_dict"], dict(ckpt, section_key=None), joint, device)


def feature_corr(Y, P) -> float:
    """Median over features of Pearson r(decoded, observed) across spots (scale/bias-free)."""
    Yc, Pc = Y - Y.mean(0), P - P.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (Yc * Pc).sum(0) / np.sqrt((Yc ** 2).sum(0) * (Pc ** 2).sum(0))
    return float(np.nanmedian(r))


def reconstruction(model, sm_names) -> dict:
    """Decoder fit from the latent mean: mean NLL per spot x feature (the training
    objective) and median per-feature correlation, for RNA and MSI from the joint latent
    and MSI from the RNA-expert latent (ST->SM path)."""
    import torch
    from spatialmeta.util.loss import LossFunction

    model.eval()
    X = torch.as_tensor(atb._dense(model.X), dtype=torch.float32, device=model.device)
    st, sm = model._type == "ST", model._type == "SM"
    X_st, X_sm = X[:, st], X[:, sm]
    with torch.no_grad():
        H = model.encode(X)
        H["z"] = H["q_mu"]
        R = model.decode(H, X_st.sum(1))["latent"]
        px_cross = model.decoder_sm(H["st"]["q_mu"])
        msi_rna, msi_rna_scale = model.px_sm_scale_decoder(px_cross), model.px_sm_rate_decoder(px_cross).exp()
        rna_nll = LossFunction.zinb_reconstruction_loss(X_st, mu=R["px_rna_scale"], theta=R["px_rna_rate"].exp(),
                                                        gate_logits=R["px_rna_dropout"], reduction="none")
        msi_nll = LossFunction.gaussian_reconstruction_loss(X_sm, mean=R["px_sm_scale"],
                                                            variance=R["px_sm_rate"].exp(), reduction="none")
        cross_nll = LossFunction.gaussian_reconstruction_loss(X_sm, mean=msi_rna, variance=msi_rna_scale,
                                                              reduction="none")
        rna = (1 - torch.sigmoid(R["px_rna_dropout"])) * R["px_rna_scale"]
    np_ = lambda t: t.cpu().numpy().astype(np.float64)  # noqa: E731
    X_st, X_sm, rna, msi, msi_rna = map(np_, (X_st, X_sm, rna, R["px_sm_scale"], msi_rna))
    out = dict(rna_nll=float(rna_nll.mean()), msi_nll=float(msi_nll.mean()), msi_from_rna_nll=float(cross_nll.mean()),
               rna_corr=feature_corr(np.log1p(X_st), np.log1p(rna)), msi_corr=feature_corr(X_sm, msi),
               msi_from_rna_corr=feature_corr(X_sm, msi_rna))
    if "msi:Dopamine" in sm_names:
        j = list(sm_names).index("msi:Dopamine")
        out["dopamine_from_rna_r"] = atb.spearman(msi_rna[:, j], X_sm[:, j])
        out["dopamine_from_joint_r"] = atb.spearman(msi[:, j], X_sm[:, j])
    return out


def part2(device):
    import torch

    rows = []
    for design, root in ROOTS.items():
        atb.BENCH_DIR = atb.RUNS_DIR = root   # fold caches live under each benchmark root
        for done in sorted(root.glob("striatum/train-*/*/seed*/DONE")):
            run = done.parent
            meta = json.load(open(run / "run.json"))
            units = meta["train_animals"]
            if len(units) != 1:
                continue
            fams = [f for f, (cfg, _) in RECON_FAMILIES.items() if cfg == meta["config"]]
            if not fams:
                continue
            fold = atb.load_fold("striatum", units)
            sm_names = fold["source_joint"].var_names[fold["source_joint"].var["type"].eq("SM").to_numpy()]
            sets = [(fold["meta"]["source_sections"][0], "in_sample", fold["source_joint"])]
            for t in fold["meta"]["target_sections"]:
                rel = "heldout_half" if atb.base_section(t) == atb.base_section(sets[0][0]) else "heldout_animal"
                sets.append((t, rel, atb.paired_target_joint(fold, t)))
            for family in fams:
                ckpt = torch.load(run / RECON_FAMILIES[family][1], map_location=device)
                for set_id, rel, joint in sets:
                    model = host_any(ckpt, joint, device)
                    rows.append(dict(family=family, design=design, unit=units[0], seed=meta["seed"],
                                     set_id=set_id, relation=rel, **reconstruction(model, sm_names)))
                    del model
            print(f"[part2] {meta['run_id']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "teacher_reconstruction_long.csv", index=False)
    metrics = [c for c in df.columns if c.endswith(("_nll", "_corr", "_r"))]
    summary = df.groupby(["family", "design", "relation"])[metrics].mean()
    summary.reset_index().to_csv(OUT / "teacher_reconstruction_summary.csv", index=False)
    print(summary.round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["1", "2", "both"], default="both")
    p.add_argument("--device", default="cuda:1")
    a = p.parse_args()
    if a.part in ("1", "both"):
        part1()
    if a.part in ("2", "both"):
        part2(a.device)

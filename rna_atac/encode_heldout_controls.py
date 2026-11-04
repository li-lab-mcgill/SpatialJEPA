"""Encode P22 and the held-out 10x Multiome with every MultiGATE model of run 20260505_190037.

analyze_rna_atac.py only saved Multiome embeddings for the distilled student. The
held-out alignment test (heldout_multiome_alignment.py) also needs the graph-free
controls on the Multiome:
  nonspatial      model_stage1_nonspatial.pth, trained on P22 with the student's identity
                  graph and no teacher (seed 2022)
  teacher_nograph model_stage1_teacher.pth evaluated with the identity graph (the teacher's
                  weights without its spatial neighbourhood)
The data preparation repeats analyze_rna_atac.py (SCT gene / gene-peak filtering, spatial
graph rad 40 for the teacher, identity graph otherwise); only named function definitions
are loaded from that script, none of its top-level code runs. Before anything is written,
the student (P22 and Multiome), the spatial teacher and the nonspatial model on P22 must
reproduce the embeddings saved for the manuscript (metrics_bootstrap/metric_inputs).

Run in the nichecompass_liana env (MultiGATE, bedtools).
Output: $SPATIALJEPA_OUT/MultiGATE/rna_atac_analysis/20260505_190037/heldout_alignment/multigate_encodings.npz
"""
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402
import ast
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch
from dotenv import load_dotenv

load_dotenv(os.environ["SPATIALJEPA_CONFIG"])
BASE = Path(os.environ["SPATIALJEPA_HOME"])
sys.path.insert(0, os.environ["SPATIALJEPA_REPO"])
sys.path.insert(0, os.environ["SPATIALJEPA_RNA_ATAC"])
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]

import MultiGATE  # noqa: E402
from MultiGATE.model_MultiGATE import MGATE  # noqa: E402
from train_rna_atac import (  # noqa: E402
    build_graph_inputs,
    build_source_student_graph_tf,
    get_sct_genes_and_gp_filtering,
    prepare_target_for_spatial_graph_type,
)

DATA = BASE / "data" / "aligned_data"
ORIGINAL = Path(os.environ["SPATIALJEPA_RNA_ATAC"]) / "analyze_rna_atac.py"
RUN_DIR = Path(os.environ.get("SPATIALJEPA_RNA_ATAC_RUN_DIR") or BASE / "outputs/rna_atac_models/20260505_190037") / "artifacts" / "models"
SAVED = BASE / "outputs/MultiGATE/rna_atac_analysis/20260505_190037/metrics_bootstrap/metric_inputs"
OUT = BASE / "outputs/MultiGATE/rna_atac_analysis/20260505_190037/heldout_alignment"
GTF = BASE / "data" / "gene_annotations" / "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"
BP_WIDTH, GRAPH_TYPE, SKIP_GP_ATTENTION = 400, "ATAC", False  # run params (checked against saved embeddings)
LOADERS = {
    "hidden_dims_from_state_dict", "rho_mask_mode_from_state_dict", "linear_decoder_kwargs_from_state_dict",
    "rho_mask_mode_from_mgate", "linear_decoder_kwargs_from_mgate", "mgate_from_state_dict",
    "unpack_mgate_checkpoint_payload", "build_zero_shot_mgate", "_as_sparse_tensor", "run_inference",
}


def original_functions():
    """Load the co-embed checkpoint and inference helpers without running the pipeline."""
    tree = ast.parse(ORIGINAL.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in LOADERS]
    assert {n.name for n in body} == LOADERS
    ns = {"np": np, "torch": torch, "MGATE": MGATE}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(ORIGINAL), "exec"), ns)
    return ns


def prepare():
    """analyze_rna_atac.py lines 2013-2225 with the identity target graph."""
    source_rna = sc.read_h5ad(DATA / "source_rna_aligned_SCT.h5ad")
    source_atac = sc.read_h5ad(DATA / "source_atac_aligned.h5ad")
    target_rna = sc.read_h5ad(DATA / "target_rna_aligned_SCT.h5ad")
    target_atac = sc.read_h5ad(DATA / "target_atac_aligned.h5ad")
    assert target_rna.obs_names.equals(target_atac.obs_names)
    source_rna.obsm["spatial"] = source_rna.obsm["spatial"] * -1
    source_atac.obsm["spatial"] = source_atac.obsm["spatial"] * -1

    MultiGATE.Cal_gene_peak_Net_new(source_rna, source_atac, 150000, file=str(GTF))
    gp_net = source_atac.uns["gene_peak_Net"].copy()
    del source_atac.uns["gene_peak_Net"]
    source_rna, source_atac, target_rna, target_atac, gp_net = get_sct_genes_and_gp_filtering(
        source_rna=source_rna, source_atac=source_atac, target_rna=target_rna,
        target_atac=target_atac, gp_net=gp_net)
    source_rna = source_rna[:, source_rna.var["highly_variable"]].copy()
    source_atac = source_atac[:, source_atac.var["highly_variable"]].copy()
    target_rna = target_rna[:, target_rna.var["highly_variable"]].copy()
    target_atac = target_atac[:, target_atac.var["highly_variable"]].copy()
    source_rna.uns["gene_peak_Net"] = gp_net.copy()

    MultiGATE.Cal_Spatial_Net(source_rna, rad_cutoff=40)
    MultiGATE.Cal_Spatial_Net(source_atac, rad_cutoff=40)
    source_atac.uns["gene_peak_Net"] = gp_net.copy()
    target_rna, target_atac = prepare_target_for_spatial_graph_type(
        target_rna=target_rna, target_atac=target_atac, source_rna=source_rna,
        source_atac=source_atac, spatial_graph_type="identity", gtf_path=str(GTF))

    spatial_graph, source_gp, source_x1, source_x2 = build_graph_inputs(
        source_rna, source_atac, bp_width=BP_WIDTH, graph_type=GRAPH_TYPE)
    identity_graph = build_source_student_graph_tf(source_rna=source_rna, spatial_graph_type="identity")
    target_graph, target_gp, target_x1, target_x2 = build_graph_inputs(
        target_rna, target_atac, bp_width=BP_WIDTH, graph_type=GRAPH_TYPE)
    return dict(
        source=dict(obs=np.asarray(source_rna.obs_names), gp=source_gp, x1=source_x1, x2=source_x2,
                    graphs=dict(spatial=spatial_graph, identity=identity_graph)),
        target=dict(obs=np.asarray(target_rna.obs_names), gp=target_gp, x1=target_x1, x2=target_x2,
                    graphs=dict(identity=target_graph)),
    )


def load(fns, name, device):
    payload = torch.load(RUN_DIR / name, map_location=device, weights_only=False)
    state_dict, _ = fns["unpack_mgate_checkpoint_payload"](payload)
    return fns["mgate_from_state_dict"](state_dict, device, skip_gp_attention=SKIP_GP_ATTENTION)[0]


def saved_embedding(domain, view, key, obs):
    a = ad.read_h5ad(SAVED / f"{domain}__{view}.h5ad")
    assert np.array_equal(np.asarray(a.obs_names), obs), domain
    return np.asarray(a.obsm[key])


def main():
    torch.manual_seed(0)
    torch.set_num_threads(8)
    device = torch.device("cpu")  # co-embed inference ran on CPU (deterministic)
    fns = original_functions()
    data = prepare()
    models = {"student": load(fns, "model_stage1.pth", device),
              "teacher": load(fns, "model_stage1_teacher.pth", device),
              "nonspatial": load(fns, "model_stage1_nonspatial.pth", device)}
    plan = [("teacher", "source", "spatial", "teacher"),
            ("student", "source", "identity", "student"),
            ("student", "target", "identity", "student"),
            ("nonspatial", "source", "identity", "nonspatial"),
            ("nonspatial", "target", "identity", "nonspatial"),
            ("teacher", "source", "identity", "teacher_nograph"),
            ("teacher", "target", "identity", "teacher_nograph")]
    out = {"source_obs_names": data["source"]["obs"], "target_obs_names": data["target"]["obs"]}
    for model, domain, graph, label in plan:
        d = data[domain]
        rna, atac, _ = fns["run_inference"](models[model], d["graphs"][graph], d["gp"], d["x1"], d["x2"], device)
        out[f"{label}__{domain}__rna"], out[f"{label}__{domain}__atac"] = rna, atac
        print(f"{label:16s} {domain:6s} graph={graph:8s} {rna.shape}", flush=True)

    checks = [("teacher", "source", "teacher_source", "MultiGATE_full_teacher"),
              ("student", "source", "source", "MultiGATE_source_aligned"),
              ("student", "target", "target", "MultiGATE"),
              ("nonspatial", "source", "source_nonspatial", "MultiGATE_nonspatial")]
    for label, domain, saved_domain, key in checks:
        for view in ("rna", "atac"):
            ref = saved_embedding(saved_domain, view, key, data[domain]["obs"])
            diff = np.abs(out[f"{label}__{domain}__{view}"] - ref).max()
            print(f"check {label} {domain} {view}: max |new - saved| = {diff:.2e}", flush=True)
            assert diff < 1e-4, "re-encoding does not reproduce the manuscript embeddings"

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "multigate_encodings.npz", **out)
    print("wrote", OUT / "multigate_encodings.npz", flush=True)


if __name__ == "__main__":
    main()

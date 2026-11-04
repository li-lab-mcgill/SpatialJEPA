"""Smoke tests: settings, syntax of every script, and the graph inputs of both backbones on tiny synthetic
data (the teacher's tissue graph, the student's self-only graph, the degree-matched permuted control).

Run in each environment of envs/:  pytest -q tests/   (tests that need a package missing from the active
environment are skipped).
"""
import os
import pathlib
import py_compile

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted(str(f.relative_to(REPO)) for f in REPO.rglob("*.py") if ".git" not in f.parts)


def test_settings_are_exported():
    import spatialjepa.paths as paths

    assert paths.REPO == REPO
    assert (paths.RNA_ATAC / "train_rna_atac.py").exists()
    assert (paths.RNA_MSI / "train_rna_msi.py").exists()
    assert (paths.HUMAN_CAH / "cah_components.py").exists()
    for key in ("SPATIALJEPA_HOME", "SPATIALJEPA_DATA", "SPATIALJEPA_OUT", "SPATIALJEPA_CONFIG", "SPATIALJEPA_REPO",
                "SPATIALJEPA_RNA_ATAC", "SPATIALJEPA_RNA_MSI", "SPATIALJEPA_HUMAN_CAH", "SPATIALJEPA_SHARED",
                "PY_MSI", "PY_ANALYSIS", "PY_SCVI", "PY_MULTIGRATE", "PY_MOTIF"):
        assert key in os.environ, key


@pytest.mark.parametrize("path", SCRIPTS)
def test_script_compiles(path):
    py_compile.compile(str(REPO / path), doraise=True)


def _coords(n_per_section=40, sections=2, seed=0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 10, size=(n_per_section * sections, 2))
    labels = np.repeat(np.arange(sections), n_per_section)
    return coords, labels


def test_spatialmeta_backbone_graphs():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    pytest.importorskip("spatialmeta")
    from spatialjepa.spatialmeta_backbone import (build_identity_edge_index, build_permuted_spatial_edge_index,
                                                  build_spatial_edge_index)

    coords, labels = _coords()
    edges = build_spatial_edge_index(coords, n_neighbors=6, section_labels=labels)
    assert edges.shape[0] == 2 and edges.shape[1] >= 6 * len(coords)
    # teacher graph: symmetric, no self-loops, no edges between sections (block-diagonal)
    pairs = set(map(tuple, edges.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    assert all(i != j for i, j in pairs)
    assert all(labels[i] == labels[j] for i, j in pairs)
    # student graph: self-loops only
    ident = build_identity_edge_index(len(coords))
    assert torch.equal(ident[0], ident[1]) and ident.shape[1] == len(coords)
    # permuted control: same degree sequence (as a multiset), still within sections
    perm = build_permuted_spatial_edge_index(coords, n_neighbors=6, section_labels=labels, seed=1)
    deg = np.bincount(edges[0].numpy(), minlength=len(coords))
    pdeg = np.bincount(perm[0].numpy(), minlength=len(coords))
    assert sorted(deg) == sorted(pdeg)
    assert all(labels[i] == labels[j] for i, j in perm.t().tolist())


def test_multigate_backbone_builds():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from MultiGATE.model_MultiGATE import MGATE

    model = MGATE(hidden_dims1=[20, 8], hidden_dims2=[30, 8], spot_num=12)
    assert sum(p.numel() for p in model.parameters()) > 0

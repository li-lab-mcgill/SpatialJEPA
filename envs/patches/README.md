# Local edits of installed packages

The environments that produced the manuscript results contained two hand edits of installed packages.
`envs/install.sh` applies them to each new environment (`patch -p1`, relative to site-packages).

| Environment | Patch | Edit |
|---|---|---|
| `spatialjepa-multigrate` | `scvi-compat-literal.diff` | `scvi/_compat.py` re-exports `Literal` from `typing`. In scvi-tools 1.0.4 this module is a shim that raises `DeprecationWarning`, and multigrate 0.0.2 imports `Literal` from it. |
| `spatialjepa-analysis` | `nichecompass-omnipath-nan-targets.diff` | `nichecompass/utils/gene_programs.py`, `extract_gp_dict_from_omnipath_lr_interactions`: drops NaN entries from the target genes of each ligand–receptor pair (used by `shared/data_utils.py`). |

One more package differs from its released version: SpatialMETA, installed from commit 909844e. The environment
that ran had one local change in `spatialmeta/data/_dataloader.py`, which only downloads SpatialMETA's tutorial
datasets and is not used by the analyses; it is not reproduced.

"""Where data, outputs and the per-environment interpreters live.

Every script in this repository imports this module first, so these settings are the only place to edit
when running the code on another machine. Values are read, first match wins, from:

1. environment variables;
2. ``configs/spatialjepa.env`` (``KEY=VALUE`` lines; copy ``configs/spatialjepa.env.example``);
3. the defaults below.

Settings
--------
SPATIALJEPA_HOME
    Root holding ``data/`` (inputs) and ``outputs/`` (models and analysis results), laid out as in the
    Zenodo deposit (docs/DATA.md). Default: the parent directory of this repository.
SPATIALJEPA_MANUSCRIPT
    Stand-in for the manuscript project: the few figure scripts that wrote straight into it write their
    PNG/SVG under ``figures/`` here, and two of them read layout templates from there (docs/DATA.md).
    Default: ``$SPATIALJEPA_HOME/outputs/manuscript``.
SPATIALJEPA_BEDTOOLS_BIN
    Directory holding the bedtools binaries used by pybedtools. Default: the directory of ``bedtools`` on
    PATH, else that of PY_ANALYSIS.
PY_MSI, PY_ANALYSIS, PY_SCVI, SPATIALJEPA_RNA_ATAC_RUN_DIR
    Exported MLflow run of the RNA-ATAC model (tools/export_mlflow_run.py): run.json, metrics/, artifacts/.
    Default: ``$SPATIALJEPA_HOME/outputs/rna_atac_models/20260505_190037`` when that folder exists.
PY_MSI, PY_ANALYSIS, PY_SCVI, PY_MULTIGRATE, PY_MOTIF
    Python interpreters of the five environments in ``envs/`` (spatialjepa-msi, spatialjepa-analysis,
    spatialjepa-scvi, spatialjepa-multigrate, spatialjepa-motifs). Default: ``python`` on PATH.

Importing the module also exports the variables that the analysis scripts read from the environment
(``SPATIALJEPA_HOME``, ``SPATIALJEPA_DATA``, ``SPATIALJEPA_OUT``), so subprocesses inherit them.
"""
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs" / "spatialjepa.env"


def _read_config(path):
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            values[key] = os.path.expandvars(os.path.expanduser(value.strip().strip("'\"")))
    return values


_CONFIG = _read_config(CONFIG)


def setting(key, default=None):
    """Value of ``key`` from the environment, else configs/spatialjepa.env, else ``default``."""
    return os.environ.get(key) or _CONFIG.get(key) or default


HOME = Path(setting("SPATIALJEPA_HOME", REPO.parent)).expanduser()
DATA = HOME / "data"
OUT = HOME / "outputs"
MANUSCRIPT = Path(setting("SPATIALJEPA_MANUSCRIPT", OUT / "manuscript")).expanduser()

# code locations inside this repository
RNA_ATAC = REPO / "rna_atac"
RNA_MSI = REPO / "rna_msi"
HUMAN_CAH = REPO / "human_cah"
SHARED = REPO / "shared"

# interpreters of the environments in envs/ (see README)
PY_MSI = setting("PY_MSI", "python")
PY_ANALYSIS = setting("PY_ANALYSIS", "python")
PY_SCVI = setting("PY_SCVI", "python")
PY_MULTIGRATE = setting("PY_MULTIGRATE", "python")
PY_MOTIF = setting("PY_MOTIF", "python")
BEDTOOLS_BIN = setting("SPATIALJEPA_BEDTOOLS_BIN",
                       os.path.dirname(shutil.which("bedtools") or PY_ANALYSIS) or "")

RNA_ATAC_RUN_DIR = setting("SPATIALJEPA_RNA_ATAC_RUN_DIR")
if RNA_ATAC_RUN_DIR is None and (OUT / "rna_atac_models" / "20260505_190037").is_dir():
    RNA_ATAC_RUN_DIR = OUT / "rna_atac_models" / "20260505_190037"
if RNA_ATAC_RUN_DIR is not None:
    os.environ.setdefault("SPATIALJEPA_RNA_ATAC_RUN_DIR", str(RNA_ATAC_RUN_DIR))

# Export the settings (and any other KEY=VALUE of the config file, e.g. MLFLOW_TRACKING_URI) so that the
# analysis scripts, code they execute and their subprocesses all see the same values.
for _key, _value in {**_CONFIG,
                     "SPATIALJEPA_HOME": HOME, "SPATIALJEPA_DATA": DATA, "SPATIALJEPA_OUT": OUT,
                     "SPATIALJEPA_MANUSCRIPT": MANUSCRIPT, "SPATIALJEPA_CONFIG": CONFIG,
                     "SPATIALJEPA_REPO": REPO, "SPATIALJEPA_RNA_ATAC": RNA_ATAC, "SPATIALJEPA_RNA_MSI": RNA_MSI,
                     "SPATIALJEPA_HUMAN_CAH": HUMAN_CAH, "SPATIALJEPA_SHARED": SHARED,
                     "PY_MSI": PY_MSI, "PY_ANALYSIS": PY_ANALYSIS, "PY_SCVI": PY_SCVI,
                     "PY_MULTIGRATE": PY_MULTIGRATE, "PY_MOTIF": PY_MOTIF,
                     "SPATIALJEPA_BEDTOOLS_BIN": BEDTOOLS_BIN}.items():
    os.environ.setdefault(_key, str(_value))

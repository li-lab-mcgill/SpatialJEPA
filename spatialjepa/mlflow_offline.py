"""Read an MLflow run exported to a folder (tools/export_mlflow_run.py) without a tracking server.

The RNA-ATAC training (rna_atac/train_rna_atac.py) logs parameters, metrics and checkpoints to MLflow, and
rna_atac/analyze_rna_atac.py reads them back through an ``mlflow.tracking.MlflowClient``. ``OfflineClient``
answers the same calls from an exported run folder::

    <run_dir>/run.json                    experiment, run info, params, tags, latest metrics, child runs
    <run_dir>/metrics/<key>.json          metric history ([{"value", "step", "timestamp"}, ...])
    <run_dir>/artifacts/...               the run's artifact tree (models/, splits/, matrices/, umap/)

The analysis uses it when SPATIALJEPA_RNA_ATAC_RUN_DIR points at such a folder (spatialjepa/paths.py sets
it when the deposited run is present).
"""
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace


class OfflineClient:
    """The subset of ``mlflow.tracking.MlflowClient`` used by rna_atac/analyze_rna_atac.py."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        meta = json.loads((self.run_dir / "run.json").read_text())
        self._experiment = meta["experiment"]
        self._runs = {r["run_id"]: r for r in [meta["run"], *meta.get("children", [])]}
        self._parent_id = meta["run"]["run_id"]
        self.tracking_config = {"backend_type": "offline", "tracking_uri": f"file://{self.run_dir}",
                                "experiment_name": self._experiment["name"]}

    # ------------------------------------------------------------------ runs
    def _run(self, run_id):
        if run_id not in self._runs:
            raise KeyError(f"run {run_id} is not in the exported folder {self.run_dir}")
        r = self._runs[run_id]
        info = SimpleNamespace(run_id=r["run_id"], run_uuid=r["run_id"], run_name=r.get("run_name"),
                               experiment_id=self._experiment["experiment_id"], status=r.get("status"),
                               start_time=r.get("start_time"), end_time=r.get("end_time"),
                               artifact_uri=f"file://{self._artifact_root(run_id)}")
        data = SimpleNamespace(params=dict(r.get("params", {})), tags=dict(r.get("tags", {})),
                               metrics=dict(r.get("metrics", {})))
        return SimpleNamespace(info=info, data=data)

    def get_experiment_by_name(self, name):
        if name != self._experiment["name"]:
            return None
        return SimpleNamespace(experiment_id=self._experiment["experiment_id"], name=name)

    def get_run(self, run_id):
        return self._run(run_id)

    def search_runs(self, experiment_ids=None, filter_string="", **_):
        runs = [self._run(i) for i in self._runs]
        m = re.fullmatch(r"\s*attributes\.run_name\s*=\s*'([^']*)'\s*", filter_string or "")
        if m:
            return [r for r in runs if r.info.run_name == m.group(1)]
        m = re.fullmatch(r"\s*tags\.mlflow\.parentRunId\s*=\s*'([^']*)'\s*", filter_string or "")
        if m:
            return [r for r in runs if r.data.tags.get("mlflow.parentRunId") == m.group(1)]
        if filter_string:
            raise NotImplementedError(f"OfflineClient.search_runs: unsupported filter {filter_string!r}")
        return runs

    def get_metric_history(self, run_id, key):
        f = self._metric_dir(run_id) / f"{key}.json"
        points = json.loads(f.read_text()) if f.exists() else []
        return [SimpleNamespace(key=key, value=p["value"], step=p.get("step", 0), timestamp=p.get("timestamp"))
                for p in points]

    # ------------------------------------------------------------------ artifacts
    def _artifact_root(self, run_id):
        return self.run_dir / "artifacts" if run_id == self._parent_id else self.run_dir / "children" / run_id / "artifacts"

    def _metric_dir(self, run_id):
        return self.run_dir / "metrics" if run_id == self._parent_id else self.run_dir / "children" / run_id / "metrics"

    def list_artifacts(self, run_id, path=None):
        root = self._artifact_root(run_id)
        base = root / (path or "")
        if not base.is_dir():
            return []
        return [SimpleNamespace(path=str(p.relative_to(root)), is_dir=p.is_dir(),
                                file_size=None if p.is_dir() else p.stat().st_size)
                for p in sorted(base.iterdir())]

    def download_artifacts(self, run_id, path, dst_path=None):
        src = self._artifact_root(run_id) / path
        if not src.exists():
            raise FileNotFoundError(f"artifact {path!r} of run {run_id} not found under {src.parent}")
        dst = Path(dst_path or ".") / Path(path).name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return str(dst)

    def log_artifact(self, run_id, local_path, artifact_path=None):
        print(f"[offline] not logging {local_path} to run {run_id} (exported run is read-only)")


def default_run_dir():
    """The deposited RNA-ATAC run folder, if SPATIALJEPA_RNA_ATAC_RUN_DIR is set and exists."""
    d = os.environ.get("SPATIALJEPA_RNA_ATAC_RUN_DIR")
    return d if d and (Path(d) / "run.json").exists() else None

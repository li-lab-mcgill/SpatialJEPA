"""Export one MLflow run (and its child runs) to a folder that spatialjepa.mlflow_offline.OfflineClient reads.

Run once, with the tracking server that holds the RNA-ATAC training run reachable:

    MLFLOW_TRACKING_URI=http://127.0.0.1:5000 python tools/export_mlflow_run.py \\
        --experiment multigate_mouse_brain_live_zeroshot --run-name 20260505_190037 \\
        --out $SPATIALJEPA_HOME/outputs/rna_atac_models/20260505_190037 [--artifacts-from DIR]

Writes run.json (experiment, run info, params, tags, latest metrics, child runs), metrics/<key>.json (full
histories) and artifacts/ (the run's artifact tree; --artifacts-from copies an already-local artifact folder
instead of downloading it, e.g. <artifact root>/<experiment>/<run_id>/artifacts).
"""
import argparse
import json
import shutil
from pathlib import Path

from mlflow.tracking import MlflowClient


def run_record(client, run):
    keys = sorted(run.data.metrics)
    return dict(run_id=run.info.run_id, run_name=run.info.run_name, status=run.info.status,
                start_time=run.info.start_time, end_time=run.info.end_time,
                params=dict(run.data.params), tags=dict(run.data.tags), metrics=dict(run.data.metrics),
                _metric_keys=keys)


def dump_metrics(client, run_id, keys, out):
    out.mkdir(parents=True, exist_ok=True)
    for key in keys:
        hist = client.get_metric_history(run_id, key)
        (out / f"{key}.json").write_text(json.dumps(
            [dict(value=m.value, step=m.step, timestamp=m.timestamp) for m in hist]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--artifacts-from", type=Path, default=None)
    ap.add_argument("--with-child-artifacts", action="store_true")
    a = ap.parse_args()

    client = MlflowClient()
    exp = client.get_experiment_by_name(a.experiment)
    if exp is None:
        raise SystemExit(f"experiment {a.experiment!r} not found at the tracking URI")
    runs = client.search_runs([exp.experiment_id], filter_string=f"attributes.run_name = '{a.run_name}'")
    if len(runs) != 1:
        raise SystemExit(f"expected one run named {a.run_name!r}, found {len(runs)}")
    run = runs[0]
    children = client.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{run.info.run_id}'")

    a.out.mkdir(parents=True, exist_ok=True)
    rec = run_record(client, run)
    dump_metrics(client, run.info.run_id, rec.pop("_metric_keys"), a.out / "metrics")
    kids = []
    for child in children:
        c = run_record(client, child)
        dump_metrics(client, child.info.run_id, c.pop("_metric_keys"), a.out / "children" / child.info.run_id / "metrics")
        if a.with_child_artifacts:
            dst = a.out / "children" / child.info.run_id / "artifacts"; dst.mkdir(parents=True, exist_ok=True)
            client.download_artifacts(child.info.run_id, "", str(dst))
        kids.append(c)
    (a.out / "run.json").write_text(json.dumps(dict(
        experiment=dict(experiment_id=exp.experiment_id, name=exp.name), run=rec, children=kids), indent=1))

    if a.artifacts_from:
        shutil.copytree(a.artifacts_from, a.out / "artifacts", dirs_exist_ok=True)
    else:
        (a.out / "artifacts").mkdir(exist_ok=True)
        client.download_artifacts(run.info.run_id, "", str(a.out / "artifacts"))
    print(f"exported run {a.run_name} ({run.info.run_id}) with {len(rec['params'])} params, "
          f"{len(rec['metrics'])} metrics and {len(kids)} child runs to {a.out}")


if __name__ == "__main__":
    main()

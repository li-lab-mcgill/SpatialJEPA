"""Detachable worker for the scib/FOSCTTM barplot metrics and their bootstrap CIs.

The metrics only need the joint embeddings and the cluster label columns, not the
models or the raw counts, so the expensive part can be split off from
``analyze_rna_atac.py`` and run as a background process:

  1. In the live (interactive) session, right after the ``scib_metric_specs``
     list has been built::

         from bootstrap_integration_metrics import dump_metric_inputs
         dump_metric_inputs(scib_metric_specs, "/path/to/metric_inputs")

  2. From a shell, detached from the session (survives logout / server shutdown)::

         nohup python bootstrap_integration_metrics.py \
             --input-dir /path/to/metric_inputs \
             --output-csv /path/to/metrics_with_ci.csv \
             --n-bootstrap 20 --foscttm-n-bootstrap 1000 --n-jobs 4 > bootstrap.log 2>&1 &

  3. Back in ``analyze_rna_atac.py``, plot without recomputing::

         --metrics-csv /path/to/metrics_with_ci.csv

The CSV is appended after every domain, so a killed run keeps its finished
domains and ``--resume`` picks up where it stopped.

Knowing when it ends
--------------------
The worker writes ``DONE.json`` (state ``complete`` or ``failed``) as its last
act, and prints ``BOOTSTRAP COMPLETE`` to the log. Three ways to be told:

  * ``notify_when_done(work_dir)`` -- a daemon thread prints a message into this
    session (with a terminal bell) as soon as the job ends. Started automatically
    by ``launch_background_bootstrap``. Dies with the kernel.
  * A shell watcher, independent of this session -- survives a kernel restart::

        until [ -f WORK_DIR/DONE.json ]; do sleep 30; done; \
            printf '\\a'; echo "BOOTSTRAP DONE"; cat WORK_DIR/DONE.json

  * ``--notify-cmd`` on the worker (or ``notify_cmd=`` when launching), which
    runs any command when the job ends -- e.g. write a file, post to a webhook,
    or send mail. ``{state}``, ``{work_dir}`` and ``{csv}`` are substituted.
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
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData


def _bootstrap_sys_path():
    """Put the MultiGATE and BAKLAVA repos on sys.path, as the main scripts do."""
    base_dir = os.getenv("SPATIALJEPA_HOME")
    if base_dir is None:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=os.environ["SPATIALJEPA_CONFIG"])
        base_dir = os.getenv("SPATIALJEPA_HOME")
    if base_dir is None:
        raise EnvironmentError("SPATIALJEPA_HOME is not set and no .env provided it.")
    for path in (
        os.environ["SPATIALJEPA_REPO"],
        os.environ["SPATIALJEPA_RNA_ATAC"],
        os.environ["SPATIALJEPA_SHARED"],
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    return base_dir


_SCIB_NUMERIC_KEYS = (
    "silhouette_label",
    "ilisi",
    "clisi",
    "graph_connectivity",
    "bras",
    "bio_conservation",
    "batch_correction",
    "total",
)

SPECS_FILENAME = "specs.json"
JOB_FILENAME = "job.json"
DONE_FILENAME = "DONE.json"
INPUTS_DIRNAME = "metric_inputs"
CSV_FILENAME = "metrics_with_ci.csv"
LOG_FILENAME = "bootstrap.log"
CSV_COLUMNS = ["domain", "metric_name", "value", "ci_low", "ci_high", "n_replicates"]


# ── dumping (runs in the live session) ────────────────────────────────────


def _slim_adata(adata, embedding_key, label_column):
    """A minimal AnnData carrying only what the metrics read."""
    if embedding_key not in adata.obsm:
        raise KeyError("Missing obsm['{}'].".format(embedding_key))
    if label_column not in adata.obs.columns:
        raise KeyError("Missing obs['{}'].".format(label_column))
    slim = AnnData(
        X=np.zeros((adata.n_obs, 1), dtype=np.float32),
        obs=adata.obs[[label_column]].copy(),
    )
    slim.obsm[embedding_key] = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    return slim


def dump_metric_inputs(scib_metric_specs, out_dir):
    """Write the embeddings + labels needed to recompute every metric panel.

    ``scib_metric_specs`` is the list built in ``analyze_rna_atac.py``:
    ``(domain, rna_adata, atac_adata, domain_name, label_key, embedding_key)``.
    The FOSCTTM specs are a subset of these, so one dump covers both.
    """
    os.makedirs(out_dir, exist_ok=True)
    specs = []
    for domain, rna_adata, atac_adata, domain_name, label_key, embedding_key in scib_metric_specs:
        rna_slim = _slim_adata(rna_adata, embedding_key, label_key["rna"])
        atac_slim = _slim_adata(atac_adata, embedding_key, label_key["atac"])
        rna_path = "{}__rna.h5ad".format(domain)
        atac_path = "{}__atac.h5ad".format(domain)
        rna_slim.write_h5ad(os.path.join(out_dir, rna_path))
        atac_slim.write_h5ad(os.path.join(out_dir, atac_path))
        specs.append(
            {
                "domain": domain,
                "domain_name": domain_name,
                "label_key": dict(label_key),
                "embedding_key": embedding_key,
                "rna_file": rna_path,
                "atac_file": atac_path,
                # FOSCTTM needs paired rows; unpaired domains get scib metrics only
                "foscttm": bool(rna_adata.n_obs == atac_adata.n_obs),
            }
        )
    with open(os.path.join(out_dir, SPECS_FILENAME), "w") as handle:
        json.dump(specs, handle, indent=2)
    print("Wrote {} domain specs to {}".format(len(specs), out_dir))
    return out_dir


# ── background job control (runs in the live session) ─────────────────────


def job_paths(work_dir):
    work_dir = os.path.abspath(work_dir)
    return {
        "work_dir": work_dir,
        "input_dir": os.path.join(work_dir, INPUTS_DIRNAME),
        "output_csv": os.path.join(work_dir, CSV_FILENAME),
        "log_path": os.path.join(work_dir, LOG_FILENAME),
        "job_path": os.path.join(work_dir, JOB_FILENAME),
        "done_path": os.path.join(work_dir, DONE_FILENAME),
    }


def launch_background_bootstrap(
    scib_metric_specs,
    work_dir,
    n_bootstrap=20,
    bootstrap_frac=0.8,
    bootstrap_seed=0,
    n_jobs=4,
    foscttm_n_bootstrap=1000,
    resume=False,
    python_exe=None,
    notify=True,
    notify_cmd=None,
):
    """Dump the inputs and start the worker as a detached background process.

    Returns immediately -- the caller can keep running other cells. The child is
    put in its own session (``start_new_session=True``, i.e. what ``setsid`` does)
    so it does not die when this session's terminal or kernel goes away.
    """
    import subprocess

    paths = job_paths(work_dir)
    os.makedirs(paths["work_dir"], exist_ok=True)

    dump_metric_inputs(scib_metric_specs, paths["input_dir"])

    script_path = os.path.abspath(__file__)
    cmd = [
        python_exe or sys.executable,
        script_path,
        "--input-dir", paths["input_dir"],
        "--output-csv", paths["output_csv"],
        "--n-bootstrap", str(n_bootstrap),
        "--foscttm-n-bootstrap", str(foscttm_n_bootstrap),
        "--bootstrap-frac", str(bootstrap_frac),
        "--bootstrap-seed", str(bootstrap_seed),
        "--n-jobs", str(n_jobs),
    ]
    if resume:
        cmd.append("--resume")
    if notify_cmd:
        cmd.extend(["--notify-cmd", notify_cmd])

    log_handle = open(paths["log_path"], "a" if resume else "w")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach: survives logout / kernel shutdown
            cwd=os.path.dirname(script_path),
        )
    finally:
        log_handle.close()

    job = {
        "pid": proc.pid,
        "cmd": cmd,
        "n_domains": len(scib_metric_specs),
        "n_bootstrap": n_bootstrap,
        "foscttm_n_bootstrap": foscttm_n_bootstrap,
        "n_jobs": n_jobs,
        "started_at": pd.Timestamp.now().isoformat(),
        **{k: v for k, v in paths.items()},
    }
    with open(paths["job_path"], "w") as handle:
        json.dump(job, handle, indent=2)

    print(
        "Bootstrap running in background (pid {}), {} domains x {} replicates.\n"
        "  status : bootstrap_status({!r})\n"
        "  log    : tail -f {}\n"
        "  results: load_bootstrap_metrics({!r})".format(
            proc.pid, job["n_domains"], n_bootstrap, work_dir, paths["log_path"], work_dir
        )
    )
    if notify:
        notify_when_done(work_dir)
    return job


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def bootstrap_status(work_dir, log_lines=5):
    """Report progress of a background bootstrap without blocking."""
    paths = job_paths(work_dir)
    if not os.path.exists(paths["job_path"]):
        raise FileNotFoundError("No job.json in {}".format(paths["work_dir"]))
    with open(paths["job_path"]) as handle:
        job = json.load(handle)

    done_domains = []
    if os.path.exists(paths["output_csv"]):
        try:
            done_domains = sorted(pd.read_csv(paths["output_csv"])["domain"].unique())
        except Exception:  # mid-append read
            done_domains = []

    ## the sentinel is authoritative: it distinguishes a finished job from one
    ## that died partway. Fall back to pid liveness only while it is absent.
    done_payload = None
    if os.path.exists(paths["done_path"]):
        try:
            with open(paths["done_path"]) as handle:
                done_payload = json.load(handle)
        except Exception:  # being written right now
            done_payload = None

    alive = _pid_alive(job["pid"])
    if done_payload is not None:
        state = done_payload["state"]  # 'complete' or 'failed'
    elif alive:
        state = "running"
    else:
        state = "died"  # process gone without writing a sentinel (OOM-kill, etc.)
    elapsed = pd.Timestamp.now() - pd.Timestamp(job["started_at"])

    print(
        "state: {}  |  {}/{} domains  |  elapsed {}".format(
            state, len(done_domains), job["n_domains"], str(elapsed).split(".")[0]
        )
    )
    if done_payload is not None and done_payload.get("error"):
        print("error: {}".format(done_payload["error"]))
    if log_lines and os.path.exists(paths["log_path"]):
        with open(paths["log_path"]) as handle:
            tail = [l for l in handle.read().rstrip("\n").split("\n") if l.strip()]
        if tail:
            print("log tail:")
            for line in tail[-log_lines:]:
                print("  " + line)

    return {
        "state": state,
        "alive": alive,
        "done_domains": done_domains,
        "n_domains": job["n_domains"],
        "elapsed": elapsed,
        **paths,
    }


def load_bootstrap_metrics(work_dir, require_complete=True):
    """Read the worker's CSV, refusing partial results unless asked otherwise."""
    paths = job_paths(work_dir)
    if not os.path.exists(paths["output_csv"]):
        raise FileNotFoundError(
            "No results yet at {}. Check bootstrap_status({!r}).".format(
                paths["output_csv"], work_dir
            )
        )
    metrics_df = pd.read_csv(paths["output_csv"])

    expected = None
    if os.path.exists(paths["job_path"]):
        with open(paths["job_path"]) as handle:
            expected = json.load(handle).get("n_domains")
    n_done = metrics_df["domain"].nunique()
    if expected is not None and n_done < expected:
        message = "Bootstrap incomplete: {}/{} domains present in {}.".format(
            n_done, expected, paths["output_csv"]
        )
        if require_complete:
            raise RuntimeError(message + " Pass require_complete=False to plot anyway.")
        warnings.warn(message)
    return metrics_df


def notify_when_done(work_dir, poll_seconds=30, bell=True, callback=None):
    """Print a message in this session as soon as the background job ends.

    Non-blocking: a daemon thread does the polling, so you keep running cells.
    The message appears in the output of whichever cell is running when the job
    finishes (or at the next prompt), and covers failure as well as success.

    Note this watcher lives in *this* interpreter -- restarting the kernel kills
    the notification but not the job. For a watcher that outlives the session,
    use the shell one-liner in this module's docstring.
    """
    import threading
    import time

    paths = job_paths(work_dir)

    def _watch():
        while True:
            try:
                state = _peek_state(paths)  # silent check; no printing from the thread
            except Exception:
                state = None
            if state in ("complete", "failed", "died"):
                message = "\n*** metrics bootstrap {}: {} ***".format(state, paths["work_dir"])
                if state == "complete":
                    message += "\n    model_metrics_df = load_bootstrap_metrics({!r})".format(work_dir)
                else:
                    message += "\n    see {}".format(paths["log_path"])
                print(("\a" if bell else "") + message, flush=True)
                if callback is not None:
                    try:
                        callback(state, paths)
                    except Exception as exc:
                        print("notify callback failed: {}".format(exc), flush=True)
                return
            time.sleep(poll_seconds)

    thread = threading.Thread(target=_watch, name="metrics-bootstrap-watch", daemon=True)
    thread.start()
    print(
        "Watching {} (every {}s); you will get a message here when it ends.".format(
            paths["work_dir"], poll_seconds
        )
    )
    return thread


def _peek_state(paths):
    """Terminal state of a job, or None while it is still running."""
    if os.path.exists(paths["done_path"]):
        try:
            with open(paths["done_path"]) as handle:
                return json.load(handle)["state"]
        except Exception:
            return None
    if os.path.exists(paths["job_path"]):
        with open(paths["job_path"]) as handle:
            pid = json.load(handle)["pid"]
        if not _pid_alive(pid):
            return "died"
    return None


def wait_for_bootstrap(work_dir, poll_seconds=60, timeout_seconds=None):
    """Block until the background job finishes (only when you *want* to wait)."""
    import time

    paths = job_paths(work_dir)
    started = time.time()
    while True:
        if _peek_state(paths) is not None:  # silent poll; report once at the end
            return bootstrap_status(work_dir)
        if timeout_seconds is not None and (time.time() - started) > timeout_seconds:
            raise TimeoutError("Bootstrap still running after {}s.".format(timeout_seconds))
        time.sleep(poll_seconds)


# ── computing (runs in the detached worker) ───────────────────────────────


def percentile_ci(replicate_values, ci=(2.5, 97.5)):
    """Percentile interval of a set of bootstrap replicates."""
    replicate_values = np.asarray(
        [v for v in replicate_values if v is not None and np.isfinite(v)], dtype=float
    )
    if replicate_values.size < 2:
        return np.nan, np.nan
    return (
        float(np.percentile(replicate_values, ci[0])),
        float(np.percentile(replicate_values, ci[1])),
    )


def spread_ci(point_value, replicate_values, frac, z=1.96):
    """Error bar for a dataset-level metric, anchored at the full-data estimate.

    Subsample replicates are *biased* relative to the full data -- fewer cells
    shifts every kNN/Leiden-based metric -- so a percentile interval of the
    replicates can sit entirely to one side of the point estimate. Take only the
    replicate spread as the uncertainty and centre it on the point estimate.

    The spread is rescaled by sqrt(frac / (1 - frac)), the delete-d jackknife
    factor (Shao & Wu 1989) for subsamples of m = frac * n drawn without
    replacement. It replaces an earlier sqrt(frac) factor, which assumes near-
    independent subsamples (m << n or drawing with replacement): at frac = 0.8
    the subsamples share most cells, so their spread understates full-sample
    variability and sqrt(frac) gave intervals ~2.2x too narrow.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError("spread_ci needs 0 < frac < 1, got {}.".format(frac))
    replicate_values = np.asarray(
        [v for v in replicate_values if v is not None and np.isfinite(v)], dtype=float
    )
    if replicate_values.size < 2:
        return np.nan, np.nan
    scale = float(np.sqrt(frac / (1.0 - frac)))
    half_width = z * float(replicate_values.std(ddof=1)) * scale
    return float(point_value - half_width), float(point_value + half_width)


def bootstrap_mean_ci(per_cell_values, seed, n_bootstrap, ci=(2.5, 97.5)):
    """Nonparametric bootstrap of the mean of per-cell scores."""
    per_cell_values = np.asarray(per_cell_values, dtype=float)
    if n_bootstrap < 2 or per_cell_values.size < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = per_cell_values.size
    boot_means = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        boot_means[b] = per_cell_values[rng.integers(0, n, size=n)].mean()
    return percentile_ci(boot_means, ci=ci)


def subsample_adata(adata, rng, frac):
    n_sub = max(int(round(frac * adata.n_obs)), 2)
    n_sub = min(n_sub, adata.n_obs)
    idx = rng.choice(adata.n_obs, size=n_sub, replace=False)
    idx.sort()
    return adata[idx].copy()


def compute_domain_rows(
    spec, input_dir, n_bootstrap, frac, seed, n_jobs, ci=(2.5, 97.5), foscttm_n_bootstrap=1000
):
    """Point estimates + bootstrap CIs for one (domain, embedding) pair.

    ``n_bootstrap`` counts the scib subsample replicates (each re-runs the full
    benchmark); ``foscttm_n_bootstrap`` counts the cheap FOSCTTM resamples.
    """
    from joblib import Parallel, delayed

    from train_rna_atac import compute_scib_metrics_for_domain

    domain = spec["domain"]
    domain_name = spec["domain_name"]
    label_key = spec["label_key"]
    embedding_key = spec["embedding_key"]

    import scanpy as sc

    rna_adata = sc.read_h5ad(os.path.join(input_dir, spec["rna_file"]))
    atac_adata = sc.read_h5ad(os.path.join(input_dir, spec["atac_file"]))

    rows = []

    ## FOSCTTM: bootstrap the mean of the per-cell scores (resampling with
    ## replacement). Subsampling cells would shrink the pool of competing cells
    ## and bias the statistic itself. Uses its own replicate count: 2.5/97.5
    ## percentiles need many replicates (with 20 they fall near the extreme
    ## values), and resampling precomputed per-cell scores makes 1000 cheap.
    if spec["foscttm"]:
        from evals_utils import foscttm_moscot

        per_cell = np.asarray(
            foscttm_moscot(
                np.asarray(rna_adata.obsm[embedding_key]),
                np.asarray(atac_adata.obsm[embedding_key]),
            ),
            dtype=float,
        ).ravel()
        lo, hi = bootstrap_mean_ci(per_cell, seed=seed, n_bootstrap=foscttm_n_bootstrap, ci=ci)
        # reported as 1 - FOSCTTM (higher is better), which flips the interval
        rows.append(
            {
                "domain": domain,
                "metric_name": "foscttm",
                "value": 1.0 - float(per_cell.mean()),
                "ci_low": 1.0 - hi,
                "ci_high": 1.0 - lo,
                "n_replicates": foscttm_n_bootstrap if foscttm_n_bootstrap >= 2 else 0,
            }
        )

    ## scib metrics are dataset-level (kNN graph + Leiden) and cannot be
    ## recomputed from per-cell scores, so each replicate re-runs the benchmark on
    ## a subsample drawn *without* replacement -- duplicated cells would sit at
    ## distance zero and corrupt every neighbourhood-based metric.
    point_metrics = compute_scib_metrics_for_domain(
        rna_adata=rna_adata,
        atac_adata=atac_adata,
        domain_name=domain_name,
        label_key=label_key,
        scib_n_jobs=1,
        embedding_key=embedding_key,
    )

    def _replicate(replicate_seed):
        rng = np.random.default_rng(replicate_seed)
        try:
            return compute_scib_metrics_for_domain(
                rna_adata=subsample_adata(rna_adata, rng, frac),
                atac_adata=subsample_adata(atac_adata, rng, frac),
                domain_name=domain_name,
                label_key=label_key,
                scib_n_jobs=1,
                embedding_key=embedding_key,
            )
        except Exception as exc:  # a subsample can drop a label entirely
            warnings.warn(
                "scib replicate failed for domain '{}' (embedding '{}'): {}".format(
                    domain, embedding_key, exc
                )
            )
            return None

    replicate_metrics = []
    if n_bootstrap >= 2:
        replicate_metrics = Parallel(n_jobs=n_jobs)(
            delayed(_replicate)(seed + b) for b in range(n_bootstrap)
        )
        replicate_metrics = [m for m in replicate_metrics if m is not None]

    for metric_name in _SCIB_NUMERIC_KEYS:
        if metric_name not in point_metrics:
            continue
        values = [m[metric_name] for m in replicate_metrics if metric_name in m]
        lo, hi = spread_ci(float(point_metrics[metric_name]), values, frac=frac)
        rows.append(
            {
                "domain": domain,
                "metric_name": metric_name,
                "value": float(point_metrics[metric_name]),
                "ci_low": lo,
                "ci_high": hi,
                "n_replicates": len(values),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Directory written by dump_metric_inputs.")
    parser.add_argument("--output-csv", required=True, help="Where to write the metrics + CIs.")
    parser.add_argument(
        "--n-bootstrap", type=int, default=20,
        help="scib subsample replicates per domain (each re-runs the full benchmark).",
    )
    parser.add_argument(
        "--foscttm-n-bootstrap", type=int, default=1000,
        help="FOSCTTM bootstrap replicates (cheap: resamples precomputed per-cell scores).",
    )
    parser.add_argument("--bootstrap-frac", type=float, default=0.8)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel replicates per domain.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep domains already present in --output-csv and only compute the rest.",
    )
    parser.add_argument(
        "--notify-cmd",
        default=None,
        help=(
            "Shell command run once when the job ends, whether it succeeded or "
            "failed. {state}, {work_dir} and {csv} are substituted, e.g. "
            "'echo {state} | mail -s bootstrap me@example.com'."
        ),
    )
    args = parser.parse_args()

    _bootstrap_sys_path()

    work_dir = os.path.dirname(os.path.abspath(args.output_csv))
    try:
        rc = _run_all_domains(args, work_dir)
    except BaseException as exc:  # includes KeyboardInterrupt / SystemExit
        import traceback

        _write_done_sentinel(
            work_dir,
            state="failed",
            error="{}: {}".format(type(exc).__name__, exc),
            traceback_text=traceback.format_exc(),
            output_csv=args.output_csv,
        )
        _run_notify_cmd(args.notify_cmd, "failed", work_dir, args.output_csv)
        raise
    _run_notify_cmd(args.notify_cmd, "complete", work_dir, args.output_csv)
    return rc


def _write_done_sentinel(work_dir, state, output_csv, error=None, traceback_text=None):
    """Record the terminal state so watchers can tell 'finished' from 'crashed'."""
    n_done = 0
    if os.path.exists(output_csv):
        try:
            n_done = int(pd.read_csv(output_csv)["domain"].nunique())
        except Exception:
            n_done = 0
    payload = {
        "state": state,
        "finished_at": pd.Timestamp.now().isoformat(),
        "n_domains_done": n_done,
        "output_csv": output_csv,
        "error": error,
        "traceback": traceback_text,
    }
    with open(os.path.join(work_dir, DONE_FILENAME), "w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _run_notify_cmd(notify_cmd, state, work_dir, output_csv):
    if not notify_cmd:
        return
    import subprocess

    command = notify_cmd.format(state=state, work_dir=work_dir, csv=output_csv)
    try:
        subprocess.run(command, shell=True, timeout=120)
    except Exception as exc:
        print("notify-cmd failed: {}".format(exc), flush=True)


def _run_all_domains(args, work_dir):
    with open(os.path.join(args.input_dir, SPECS_FILENAME)) as handle:
        specs = json.load(handle)

    done_path = os.path.join(work_dir, DONE_FILENAME)
    if os.path.exists(done_path):  # stale sentinel from a previous run
        os.remove(done_path)

    done_domains = set()
    if args.resume and os.path.exists(args.output_csv):
        existing = pd.read_csv(args.output_csv)
        done_domains = set(existing["domain"].unique())
        print("Resuming; {} domains already done.".format(len(done_domains)))
    elif os.path.exists(args.output_csv):
        os.remove(args.output_csv)

    import time

    for spec_idx, spec in enumerate(specs):
        if spec["domain"] in done_domains:
            continue
        started = time.time()
        rows = compute_domain_rows(
            spec,
            input_dir=args.input_dir,
            n_bootstrap=args.n_bootstrap,
            frac=args.bootstrap_frac,
            # distinct seed block per domain, matching the in-script convention
            seed=args.bootstrap_seed + 1000 * (spec_idx + 1),
            n_jobs=args.n_jobs,
            foscttm_n_bootstrap=args.foscttm_n_bootstrap,
        )
        # append after each domain so a killed run keeps its finished work
        pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(
            args.output_csv,
            mode="a",
            header=not os.path.exists(args.output_csv),
            index=False,
        )
        print(
            "[{}/{}] {} done in {:.1f} min".format(
                spec_idx + 1, len(specs), spec["domain"], (time.time() - started) / 60.0
            ),
            flush=True,
        )

    _write_done_sentinel(work_dir, state="complete", output_csv=args.output_csv)
    print("Wrote {}".format(args.output_csv), flush=True)
    print("BOOTSTRAP COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    main()

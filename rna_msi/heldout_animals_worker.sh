#!/usr/bin/env bash
# Pull-based worker for the cross-animal SpatialJEPA ablation benchmark.
#
# Runs on any host that sees the queue directory (this machine, a remote host, SLURM nodes). It claims runs
# from <queue>/runs.tsv one at a time (atomic `mkdir <queue>/claims/<run_id>`), trains
# them with heldout_animals_train.py on the GPU in CUDA_VISIBLE_DEVICES, and exits
# when nothing claimable is left. Completion is the run directory's DONE file; failures
# are recorded in <queue>/failed/<run_id>.
#
# Usage: heldout_animals_worker.sh <queue_dir> [worker_tag]
# Started by heldout_animals_run.sh; settings come from <queue>/queue.env.

set -uo pipefail

QUEUE_DIR=${1:?usage: heldout_animals_worker.sh <queue_dir> [worker_tag]}
WORKER_TAG=${2:-$(hostname -s)_gpu${CUDA_VISIBLE_DEVICES:-na}_$$}
# shellcheck source=/dev/null
source "${QUEUE_DIR}/queue.env"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}
export ANIMAL_TRANSFER_RUNS_DIR=${RUNS_DIR}
# queues written before these settings existed fall back to the original benchmark
export ANIMAL_TRANSFER_BENCH_DIR=${BENCH_DIR:-${RUNS_DIR}}
export ANIMAL_TRANSFER_PANEL_TOP=${PANEL_TOP:-442}

log() { echo "[$(date '+%F %T')] [${WORKER_TAG}] $*"; }

# A broken driver (e.g. buckeridge) should cost nothing: check before claiming.
if ! "${PY}" -c "import sys, torch; ok = torch.cuda.is_available() and float(torch.ones(8, device='cuda').sum()) == 8.0; sys.exit(0 if ok else 3)" </dev/null; then
    log "GPU unusable on $(hostname -s) (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}); exiting without claiming"
    exit 3
fi
log "GPU ok: $("${PY}" -c "import torch; print(torch.cuda.get_device_name(0))" </dev/null 2>/dev/null)"

n_done=0
while IFS=$'\t' read -r run_id region train_animals config seed; do
    [[ -z "${run_id}" || "${run_id}" == "run_id" ]] && continue
    run_dir="${RUNS_DIR}/${region}/train-${train_animals}/${config}/seed${seed}"
    [[ -f "${run_dir}/DONE" ]] && continue
    [[ -e "${QUEUE_DIR}/failed/${run_id}" ]] && continue
    mkdir "${QUEUE_DIR}/claims/${run_id}" 2>/dev/null || continue

    printf 'host=%s\nworker=%s\ngpu=%s\npid=%s\nslurm_job=%s\nstarted=%s\n' \
        "$(hostname -s)" "${WORKER_TAG}" "${CUDA_VISIBLE_DEVICES:-}" "$$" \
        "${SLURM_JOB_ID:-}" "$(date '+%F %T')" > "${QUEUE_DIR}/claims/${run_id}/info"
    log "start ${run_id}"
    run_log="${QUEUE_DIR}/logs/${run_id}.log"
    # stdin from /dev/null: the loop itself is reading runs.tsv on stdin
    "${PY}" "${SCRIPT_DIR}/heldout_animals_train.py" \
        --region "${region}" \
        --train-animals ${train_animals//+/ } \
        --config "${config}" \
        --seed "${seed}" \
        --max-epoch "${MAX_EPOCH}" \
        > "${run_log}" 2>&1 < /dev/null
    status=$?

    if [[ ${status} -eq 0 && -f "${run_dir}/DONE" ]]; then
        log "done ${run_id}"
        n_done=$((n_done + 1))
        rm -rf "${QUEUE_DIR}/claims/${run_id}"
    elif [[ ${status} -eq 3 ]]; then
        # node problem, not a run problem: release the run for another worker
        log "GPU became unusable during ${run_id}; releasing it and exiting"
        rm -rf "${QUEUE_DIR}/claims/${run_id}"
        exit 3
    else
        log "FAILED ${run_id} (exit ${status}); see ${run_log}"
        printf 'exit=%s\nhost=%s\nworker=%s\nlog=%s\nat=%s\n' "${status}" "$(hostname -s)" \
            "${WORKER_TAG}" "${run_log}" "$(date '+%F %T')" > "${QUEUE_DIR}/failed/${run_id}"
        rm -rf "${QUEUE_DIR}/claims/${run_id}"
    fi
done < "${QUEUE_DIR}/runs.tsv"

log "queue exhausted after ${n_done} completed run(s)"

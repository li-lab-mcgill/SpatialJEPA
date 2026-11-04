#!/usr/bin/env bash
# Orchestrate the cross-animal SpatialJEPA ablation benchmark across GPU servers.
#
# Workers (heldout_animals_worker.sh) pull runs from a shared queue directory, so GPUs on this machine,
# a remote host and SLURM nodes balance automatically. Without a cluster, run 'plan' and 'prepare', then
# start one heldout_animals_worker.sh <queue_dir> per local GPU (or BACKENDS=local launch).
#
#   heldout_animals_run.sh plan          # write <queue>/runs.tsv from the settings below
#   heldout_animals_run.sh prepare       # build fold caches locally (CPU), before launch
#   heldout_animals_run.sh launch        # start workers on $BACKENDS
#   heldout_animals_run.sh status        # pending / running / done / failed, SLURM jobs, workers
#   heldout_animals_run.sh stop          # scancel SLURM workers, kill host workers, release claims
#   heldout_animals_run.sh reset-failed  # make failed runs claimable again
#   heldout_animals_run.sh reset-stale   # release claims older than $STALE_HOURS
#   heldout_animals_run.sh evaluate      # heldout_animals_evaluate.py on finished runs
#   heldout_animals_run.sh structure     # 9__: manuscript Fig. 5 dopamine metrics (2 envs)
#
# Settings (environment variables):
#   REGION=striatum|sn|both        DESIGN=pairwise|loao|both|split
#   CONFIGS="A A30 B C C30 D"      (opt-ins: A_perm A_induced G)
#   SEEDS="0 1 2"                  MAX_EPOCH=1000
#   QUEUE_NAME=main                ANIMAL_TRANSFER_RUNS_DIR=<output root>
#   ANIMAL_TRANSFER_BENCH_DIR=<fold caches + runs root>   ANIMAL_TRANSFER_PANEL_TOP=442|2000
#   BACKENDS="local" | "slurm remote local" (remote: $REMOTE_HOST over ssh; local: this machine)
#   SLURM_HOST, SLURM_ACCOUNT, SLURM_PARTITION, SLURM_QOS (site settings)   SLURM_JOBS=4   SLURM_NODES=auto | "n1 n2"
#   SLURM_SMALL_QOS=0|1 (one extra job on $SLURM_SMALL_QOS_NAME)   WORKERS_PER_GPU_SLURM=2
#   REMOTE_HOST, REMOTE_MAX_GPUS=8, LOCAL_MAX_GPUS=3, HOST_DOMAIN (appended to worker host names for ssh)   WORKERS_PER_GPU_HOST=1
#   GPU_FREE_MIB=500 GPU_FREE_UTIL=10 (a host GPU counts as free below both)
#   STALE_HOURS=12

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-${PY_MSI}}
# dopamine_embedding_benchmark.py (imported by 9__ --stage evaluate) needs scib_metrics/jax
PY_METRICS=${PY_METRICS:-${PY_ANALYSIS}}
SPATIALJEPA_OUT=${SPATIALJEPA_OUT:-${SPATIALJEPA_HOME}/outputs}
BENCH_DIR=${ANIMAL_TRANSFER_BENCH_DIR:-${SPATIALJEPA_OUT}/spatialjepa_benchmark/animal_transfer}
RUNS_DIR=${ANIMAL_TRANSFER_RUNS_DIR:-${BENCH_DIR}}
PANEL_TOP=${ANIMAL_TRANSFER_PANEL_TOP:-442}
export ANIMAL_TRANSFER_BENCH_DIR=${BENCH_DIR} ANIMAL_TRANSFER_PANEL_TOP=${PANEL_TOP}

REGION=${REGION:-striatum}
DESIGN=${DESIGN:-pairwise}
CONFIGS=${CONFIGS:-"A A30 B C C30 D"}
SEEDS=${SEEDS:-"0 1 2"}
MAX_EPOCH=${MAX_EPOCH:-1000}
QUEUE_NAME=${QUEUE_NAME:-main}
QUEUE_DIR=${QUEUE_DIR:-${RUNS_DIR}/queue/${QUEUE_NAME}}
BACKENDS=${BACKENDS:-"local"}

SLURM_HOST=${SLURM_HOST:-}
SLURM_ACCOUNT=${SLURM_ACCOUNT:-}
SLURM_PARTITION=${SLURM_PARTITION:-}
SLURM_QOS=${SLURM_QOS:-}
SLURM_SMALL_QOS_NAME=${SLURM_SMALL_QOS_NAME:-}
SLURM_JOBS=${SLURM_JOBS:-4}
SLURM_NODES=${SLURM_NODES:-auto}
SLURM_EXCLUDE=${SLURM_EXCLUDE:-}   # e.g. nodes whose GPUs torch 2.2.2+cu121 does not support
SLURM_SMALL_QOS=${SLURM_SMALL_QOS:-0}
SLURM_TIME=${SLURM_TIME:-2-00:00:00}
SLURM_CPUS=${SLURM_CPUS:-6}
SLURM_MEM=${SLURM_MEM:-32G}
WORKERS_PER_GPU_SLURM=${WORKERS_PER_GPU_SLURM:-2}

REMOTE_HOST=${REMOTE_HOST:-}
REMOTE_MAX_GPUS=${REMOTE_MAX_GPUS:-8}
LOCAL_MAX_GPUS=${LOCAL_MAX_GPUS:-3}
HOST_DOMAIN=${HOST_DOMAIN:-}
WORKERS_PER_GPU_HOST=${WORKERS_PER_GPU_HOST:-1}
GPU_FREE_MIB=${GPU_FREE_MIB:-500}
GPU_FREE_UTIL=${GPU_FREE_UTIL:-10}
STALE_HOURS=${STALE_HOURS:-12}

ANIMALS=(mPD1 mPD3 mPD4)
JOB_NAME="sjepa_xfer_${QUEUE_NAME}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)

die() { echo "ERROR: $*" >&2; exit 1; }

regions() {
    case "${REGION}" in
        both) echo "striatum sn" ;;
        striatum|sn) echo "${REGION}" ;;
        *) die "REGION must be striatum, sn or both" ;;
    esac
}

folds() {   # training-animal sets, "+"-joined
    local design=$1
    if [[ ${design} == pairwise || ${design} == both ]]; then
        printf '%s\n' "${ANIMALS[@]}"
    fi
    if [[ ${design} == loao || ${design} == both ]]; then
        echo "mPD1+mPD3"; echo "mPD1+mPD4"; echo "mPD3+mPD4"
    fi
    if [[ ${design} == split ]]; then   # train on half a section, hold out the other half
        local a
        for a in "${ANIMALS[@]}"; do echo "${a}@h0"; echo "${a}@h1"; done
    fi
}

require_queue() { [[ -f "${QUEUE_DIR}/runs.tsv" ]] || die "no queue at ${QUEUE_DIR}; run 'plan' first"; }

slurm() { "${SSH[@]}" "${SLURM_HOST}" "bash -lc 'module load slurm >/dev/null 2>&1; $*'"; }

cmd_plan() {
    mkdir -p "${QUEUE_DIR}"/{claims,failed,logs,workers,slurm}
    local tsv="${QUEUE_DIR}/runs.tsv" n=0
    printf 'run_id\tregion\ttrain_animals\tconfig\tseed\n' > "${tsv}"
    # seed-major order: the first wave completes seed 0 of every config and fold
    for seed in ${SEEDS}; do
        for region in $(regions); do
            while read -r fold; do
                for config in ${CONFIGS}; do
                    printf '%s__train-%s__%s__seed%s\t%s\t%s\t%s\t%s\n' \
                        "${region}" "${fold}" "${config}" "${seed}" "${region}" "${fold}" "${config}" "${seed}" >> "${tsv}"
                    n=$((n + 1))
                done
            done < <(folds "${DESIGN}")
        done
    done
    cat > "${QUEUE_DIR}/queue.env" <<EOF
PY=${PY}
SCRIPT_DIR=${SCRIPT_DIR}
RUNS_DIR=${RUNS_DIR}
BENCH_DIR=${BENCH_DIR}
PANEL_TOP=${PANEL_TOP}
MAX_EPOCH=${MAX_EPOCH}
EOF
    echo "planned ${n} runs -> ${tsv}"
    echo "  REGION=${REGION} DESIGN=${DESIGN} CONFIGS='${CONFIGS}' SEEDS='${SEEDS}' MAX_EPOCH=${MAX_EPOCH}"
    echo "  runs land under ${RUNS_DIR} (fold caches: ${BENCH_DIR}, panel ${PANEL_TOP})"
}

fold_list() {   # unique "region<TAB>train_animals" pairs in the queue
    tail -n +2 "${QUEUE_DIR}/runs.tsv" | cut -f2,3 | sort -u
}

cmd_prepare() {
    require_queue
    local log
    while IFS=$'\t' read -r region fold; do
        log="${QUEUE_DIR}/logs/prepare_${region}_train-${fold}.log"
        echo "[prepare] ${region} train-${fold}"
        ANIMAL_TRANSFER_RUNS_DIR="${RUNS_DIR}" MPLBACKEND=Agg "${PY}" "${SCRIPT_DIR}/heldout_animals_train.py" \
            --prepare-only --region "${region}" --train-animals ${fold//+/ } > "${log}" 2>&1 < /dev/null \
            || die "fold preparation failed; see ${log}"
        tail -n 1 "${log}"
    done < <(fold_list)
}

check_folds_ready() {
    local missing=0
    while IFS=$'\t' read -r region fold; do
        if [[ ! -f "${BENCH_DIR}/${region}/train-${fold}/data/fold.json" ]]; then
            echo "missing fold cache: ${region} train-${fold}" >&2
            missing=1
        fi
    done < <(fold_list)
    [[ ${missing} -eq 0 ]] || die "run 'prepare' first"
}

write_sbatch() {   # <file> <qos> <cpus> <mem> <workers> <node|auto>
    local file=$1 qos=$2 cpus=$3 mem=$4 workers=$5 node=$6 placement
    if [[ ${node} == auto ]]; then
        placement="#SBATCH --exclude=${SLURM_EXCLUDE}"
    else
        placement="#SBATCH --nodelist=${node}"
    fi
    cat > "${file}" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --qos=${qos}
${placement}
#SBATCH --gpus=1
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem=${mem}
#SBATCH --time=${SLURM_TIME}
#SBATCH --propagate=NONE
#SBATCH --chdir=${SCRIPT_DIR}
#SBATCH --output=${QUEUE_DIR}/workers/slurm_%j.out
#SBATCH --error=${QUEUE_DIR}/workers/slurm_%j.err

set -uo pipefail
export OMP_NUM_THREADS=2
pids=()
for n in \$(seq 1 ${workers}); do
    bash ${SCRIPT_DIR}/heldout_animals_worker.sh ${QUEUE_DIR} "slurm\${SLURM_JOB_ID}_\${SLURMD_NODENAME}_\${n}" \\
        > ${QUEUE_DIR}/workers/slurm_\${SLURM_JOB_ID}_\${n}.log 2>&1 &
    pids+=(\$!)
done
wait "\${pids[@]}"
EOF
}

launch_slurm() {
    local nodes i file job_id node
    read -r -a nodes <<< "${SLURM_NODES}"
    for ((i = 0; i < SLURM_JOBS; i++)); do
        node=${nodes[$((i % ${#nodes[@]}))]}
        file="${QUEUE_DIR}/slurm/worker_${i}.sbatch"
        write_sbatch "${file}" "${SLURM_QOS}" "${SLURM_CPUS}" "${SLURM_MEM}" "${WORKERS_PER_GPU_SLURM}" "${node}"
        job_id=$(slurm "sbatch --parsable ${file}" | tail -n 1)
        job_id=${job_id%%;*}
        [[ ${job_id} =~ ^[0-9]+$ ]] || die "sbatch failed: ${job_id}"
        echo "${job_id}" >> "${QUEUE_DIR}/slurm_jobs.txt"
        echo "[slurm] job ${job_id} (${SLURM_QOS}, ${node}, ${WORKERS_PER_GPU_SLURM} worker(s))"
    done
    if [[ ${SLURM_SMALL_QOS} == 1 ]]; then
        file="${QUEUE_DIR}/slurm/worker_small.sbatch"
        write_sbatch "${file}" "${SLURM_SMALL_QOS_NAME:?set SLURM_SMALL_QOS_NAME}" 3 30G 1 auto
        job_id=$(slurm "sbatch --parsable ${file}" | tail -n 1)
        job_id=${job_id%%;*}
        [[ ${job_id} =~ ^[0-9]+$ ]] || die "sbatch failed: ${job_id}"
        echo "${job_id}" >> "${QUEUE_DIR}/slurm_jobs.txt"
        echo "[slurm] job ${job_id} (${SLURM_SMALL_QOS_NAME})"
    fi
}

# Start one setsid worker per free GPU on a non-scheduler host. Prints "host pid gpu".
host_launcher() {   # <max_gpus>
    cat <<EOF
set -euo pipefail
host=\$(hostname -s)
free=\$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \\
    | awk -F', *' -v m=${GPU_FREE_MIB} -v u=${GPU_FREE_UTIL} '\$2 < m && \$3 < u {print \$1}' | head -n $1)
for g in \${free}; do
    for n in \$(seq 1 ${WORKERS_PER_GPU_HOST}); do
        tag="\${host}_gpu\${g}_\${n}"
        CUDA_VISIBLE_DEVICES=\${g} setsid nohup bash ${SCRIPT_DIR}/heldout_animals_worker.sh ${QUEUE_DIR} "\${tag}" \\
            > ${QUEUE_DIR}/workers/\${tag}.log 2>&1 < /dev/null &
        echo "\${host} \$! \${g}"
    done
done
EOF
}

launch_host() {   # <host|local> <max_gpus>
    local host=$1 max=$2 out
    if [[ ${host} == local ]]; then
        out=$(bash -c "$(host_launcher "${max}")")
    else
        out=$("${SSH[@]}" "${host}" "bash -s" <<< "$(host_launcher "${max}")")
    fi
    if [[ -z ${out} ]]; then
        echo "[${host}] no free GPUs (<${GPU_FREE_MIB} MiB, <${GPU_FREE_UTIL}% util)"
        return
    fi
    echo "${out}" >> "${QUEUE_DIR}/workers/pids.txt"
    echo "${out}" | awk '{print "[" $1 "] worker pid " $2 " on GPU " $3}'
}

cmd_launch() {
    require_queue
    check_folds_ready
    for backend in ${BACKENDS}; do
        case "${backend}" in
            slurm) launch_slurm ;;
            remote) launch_host "${REMOTE_HOST:?set REMOTE_HOST}" "${REMOTE_MAX_GPUS}" ;;
            local) launch_host local "${LOCAL_MAX_GPUS}" ;;
            *) die "unknown backend ${backend} (slurm|remote|local)" ;;
        esac
    done
}

count_done() {
    local n=0
    while IFS=$'\t' read -r _ region fold config seed; do
        [[ -f "${RUNS_DIR}/${region}/train-${fold}/${config}/seed${seed}/DONE" ]] && n=$((n + 1))
    done < <(tail -n +2 "${QUEUE_DIR}/runs.tsv")
    echo "${n}"
}

cmd_status() {
    require_queue
    local total done_n running failed
    total=$(($(wc -l < "${QUEUE_DIR}/runs.tsv") - 1))
    done_n=$(count_done)
    running=$(find "${QUEUE_DIR}/claims" -mindepth 1 -maxdepth 1 -type d | wc -l)
    failed=$(find "${QUEUE_DIR}/failed" -mindepth 1 -maxdepth 1 | wc -l)
    echo "queue ${QUEUE_DIR}"
    echo "  total ${total} | done ${done_n} | running ${running} | failed ${failed} | pending $((total - done_n - running - failed))"
    if [[ ${running} -gt 0 ]]; then
        echo "  running by host:"
        cat "${QUEUE_DIR}"/claims/*/info 2>/dev/null | awk -F= '$1 == "host" {print "    " $2}' | sort | uniq -c
        local stale
        stale=$(find "${QUEUE_DIR}/claims" -mindepth 1 -maxdepth 1 -type d -mmin "+$((STALE_HOURS * 60))" | wc -l)
        [[ ${stale} -gt 0 ]] && echo "  WARNING: ${stale} claim(s) older than ${STALE_HOURS} h (see reset-stale)"
    fi
    if [[ ${failed} -gt 0 ]]; then
        echo "  failed:"
        for f in "${QUEUE_DIR}"/failed/*; do
            echo "    $(basename "${f}"): $(grep -E '^(exit|host)=' "${f}" | tr '\n' ' ')"
        done
    fi
    if [[ -s "${QUEUE_DIR}/slurm_jobs.txt" ]]; then
        echo "  SLURM (${JOB_NAME}):"
        slurm "squeue -u \$USER -n ${JOB_NAME} -o \"%.10i %.9T %.12M %R\"" < /dev/null 2>/dev/null | sed 's/^/    /' || echo "    (squeue unavailable)"
    fi
    if [[ -s "${QUEUE_DIR}/workers/pids.txt" ]]; then
        echo "  host workers alive:"
        while read -r host pid gpu; do
            local alive
            if [[ ${host} == "$(hostname -s)" ]]; then
                kill -0 "${pid}" 2>/dev/null && alive=yes || alive=no
            else
                "${SSH[@]}" "${host}${HOST_DOMAIN}" "kill -0 ${pid}" < /dev/null 2>/dev/null && alive=yes || alive=no
            fi
            echo "    ${host} gpu${gpu} pid ${pid}: ${alive}"
        done < "${QUEUE_DIR}/workers/pids.txt"
    fi
}

cmd_stop() {
    require_queue
    if [[ -s "${QUEUE_DIR}/slurm_jobs.txt" ]]; then
        slurm "scancel $(tr '\n' ' ' < "${QUEUE_DIR}/slurm_jobs.txt")" || true
        echo "[slurm] cancelled $(wc -l < "${QUEUE_DIR}/slurm_jobs.txt") job(s)"
        mv "${QUEUE_DIR}/slurm_jobs.txt" "${QUEUE_DIR}/slurm_jobs.$(date +%s).stopped"
    fi
    if [[ -s "${QUEUE_DIR}/workers/pids.txt" ]]; then
        while read -r host pid gpu; do
            # setsid made each worker a process-group leader: kill the group (worker + python)
            if [[ ${host} == "$(hostname -s)" ]]; then
                kill -- "-${pid}" 2>/dev/null || true
            else
                "${SSH[@]}" "${host}${HOST_DOMAIN}" "kill -- -${pid}" < /dev/null 2>/dev/null || true
            fi
            echo "[${host}] stopped worker ${pid} (gpu${gpu})"
        done < "${QUEUE_DIR}/workers/pids.txt"
        mv "${QUEUE_DIR}/workers/pids.txt" "${QUEUE_DIR}/workers/pids.$(date +%s).stopped"
    fi
    # all workers are gone: their in-flight runs are incomplete and must be redone
    find "${QUEUE_DIR}/claims" -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
    echo "released all claims"
}

cmd_reset_failed() {
    require_queue
    find "${QUEUE_DIR}/failed" -mindepth 1 -maxdepth 1 -print -delete | sed 's/^/reset /'
}

cmd_reset_stale() {
    require_queue
    find "${QUEUE_DIR}/claims" -mindepth 1 -maxdepth 1 -type d -mmin "+$((STALE_HOURS * 60))" -print \
        -exec rm -rf {} + | sed 's/^/released /'
}

cmd_evaluate() {
    ANIMAL_TRANSFER_RUNS_DIR="${RUNS_DIR}" MPLBACKEND=Agg OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 \
        "${PY}" "${SCRIPT_DIR}/heldout_animals_evaluate.py" --runs-dir "${RUNS_DIR}" "$@"
}

cmd_structure() {
    # stage 1 encodes graph teachers in-sample: use a free local GPU if there is one
    local gpu
    gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk -F', *' -v m="${GPU_FREE_MIB}" '$2 < m {print $1; exit}')
    ANIMAL_TRANSFER_RUNS_DIR="${RUNS_DIR}" MPLBACKEND=Agg CUDA_VISIBLE_DEVICES="${gpu:-}" \
        "${PY}" "${SCRIPT_DIR}/heldout_animals_dopamine_structure.py" --stage prepare --runs-dir "${RUNS_DIR}"
    ANIMAL_TRANSFER_RUNS_DIR="${RUNS_DIR}" MPLBACKEND=Agg OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
        "${PY_METRICS}" "${SCRIPT_DIR}/heldout_animals_dopamine_structure.py" --stage evaluate \
        --runs-dir "${RUNS_DIR}" "$@"
}

case "${1:-help}" in
    plan) cmd_plan ;;
    prepare) cmd_prepare ;;
    launch) cmd_launch ;;
    status) cmd_status ;;
    stop) cmd_stop ;;
    reset-failed) cmd_reset_failed ;;
    reset-stale) cmd_reset_stale ;;
    evaluate) shift; cmd_evaluate "$@" ;;
    structure) shift; cmd_structure "$@" ;;
    help|-h|--help) sed -n '2,33p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    *) die "unknown subcommand '$1' (see: $0 help)" ;;
esac

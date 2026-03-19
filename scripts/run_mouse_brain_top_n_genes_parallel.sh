#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/mcb/users/dmannk/BAKLAVA_base/outputs/logs/mouse_brain_top_n_genes"

gpu_ids=(5 6 7 8 9)
top_n_genes_values=(3500 3750 4000 4250 4500)
top_n_peaks_values=(21000 22000 23000 24000 25000)

if [[ "${#gpu_ids[@]}" -ne "${#top_n_genes_values[@]}" ]]; then
    echo "gpu_ids and top_n_genes_values must have the same length." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"

eval "$(conda shell.bash hook)"
conda activate nichecompass

declare -a pids=()
declare -a run_labels=()

for idx in "${!gpu_ids[@]}"; do
    gpu_id="${gpu_ids[$idx]}"
    top_n_genes="${top_n_genes_values[$idx]}"
    top_n_peaks="${top_n_peaks_values[$idx]}"
    log_file="${LOG_DIR}/top_n_genes_${top_n_genes}_top_n_peaks_${top_n_peaks}_gpu_${gpu_id}.log"

    echo "Launching run on GPU ${gpu_id} with --top-n-genes=${top_n_genes} --top-n-peaks=${top_n_peaks}"

    (
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        python "${SCRIPT_DIR}/mouse_brain_spatial_rna_atac.py" \
            --stage1-epochs=10 \
            --stage2-epochs=0 \
            --stage1-dual-source-kd \
            --top-n-genes="${top_n_genes}" \
            --top-n-peaks="${top_n_peaks}"
    ) > "${log_file}" 2>&1 &

    pids+=("$!")
    run_labels+=("gpu=${gpu_id},top_n_genes=${top_n_genes},top_n_peaks=${top_n_peaks}")
done

failures=0

for idx in "${!pids[@]}"; do
    pid="${pids[$idx]}"
    label="${run_labels[$idx]}"

    if wait "${pid}"; then
        echo "Completed ${label}"
    else
        echo "Failed ${label}" >&2
        failures=1
    fi
done

conda deactivate

exit "${failures}"

#!/bin/bash
#
#SBATCH -p li-gpus
#SBATCH -A yueli-2026
#SBATCH -q li-qos
#SBATCH -c 1 # number of cores
#SBATCH --mem=16G
#SBATCH -t 0-2:00
#SBATCH --gpus 1
#SBATCH --propagate=NONE # IMPORTANT for long jobs
#SBATCH -o /home/mcb/users/dmannk/BAKLAVA_base/outputs/MultiGATE/slurm/mouse_brain_spatial_rna_atac_%j.out # STDOUT
#SBATCH -e /home/mcb/users/dmannk/BAKLAVA_base/outputs/MultiGATE/slurm/mouse_brain_spatial_rna_atac_%j.err # STDERR

# Batch jobs use non-login bash: `conda init` does not hook this shell; source conda.sh instead.
if [[ -f "/usr/local/pkgs/anaconda/etc/profile.d/conda.sh" ]]; then
    source "/usr/local/pkgs/anaconda/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "ERROR: conda not found (expected /usr/local/pkgs/anaconda or ~/miniconda3)." >&2
    exit 1
fi

conda activate nichecompass

python /home/mcb/users/dmannk/BAKLAVA_base/MultiGATE/scripts/mouse_brain_spatial_rna_atac.py \
--stage1-epochs=3000 \
--stage2-epochs=0 \
--top-n-genes=4000 \
--top-n-peaks=22500 \
--vgp-anchor-mode='spot' \
--stage1-dual-source-kd \
#--stage1-mlflow-cache-dir='20260331_153543'
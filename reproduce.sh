#!/usr/bin/env bash
# Reproduce the figures and reported numbers of the SpatialJEPA manuscript.
#
#   ./reproduce.sh figures            every figure, from the deposited models and intermediates (~2 h, 1 GPU)
#   ./reproduce.sh analyses           recompute the deposited intermediates (GPU; the held-out-animal benchmark
#                                     retrains its 162 runs)
#   ./reproduce.sh train              retrain the manuscript models (GPU); RETRAIN_SUFFIX=_retrained writes them
#                                     next to the deposited models instead of replacing them
#   ./reproduce.sh list               list the steps;  ./reproduce.sh <step> [<step> ...] runs single steps
#
# Settings come from configs/spatialjepa.env (copy configs/spatialjepa.env.example): SPATIALJEPA_HOME (the
# unpacked deposit plus the public data, docs/DATA.md) and the interpreters of the environments in envs/.
# Final figures are written to $SPATIALJEPA_MANUSCRIPT/figures/ under the manuscript's file names; the
# figure-to-step map is docs/REPRODUCIBILITY.md. GPU=<index> selects the GPU (default 0).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [[ -f $REPO/configs/spatialjepa.env ]]; then set -a; source "$REPO/configs/spatialjepa.env"; set +a; fi
if [[ ${1:-} == list ]]; then SPATIALJEPA_HOME=${SPATIALJEPA_HOME:-"<SPATIALJEPA_HOME>"}; fi   # listing needs no settings
: "${SPATIALJEPA_HOME:?set SPATIALJEPA_HOME in configs/spatialjepa.env}"
: "${PY_MSI:=python}" "${PY_ANALYSIS:=python}" "${PY_SCVI:=python}" "${PY_MULTIGRATE:=python}" "${PY_MOTIF:=python}"
export SPATIALJEPA_HOME PY_MSI PY_ANALYSIS PY_SCVI PY_MULTIGRATE PY_MOTIF
export SPATIALJEPA_MANUSCRIPT=${SPATIALJEPA_MANUSCRIPT:-$SPATIALJEPA_HOME/outputs/manuscript}
export MPLBACKEND=Agg PYTHONUTF8=1
GPU=${GPU:-0}

OUT=$SPATIALJEPA_HOME/outputs
FIG=$SPATIALJEPA_MANUSCRIPT/figures          # final figures (manuscript file names)
WORK=$OUT/manuscript_work                       # figure products before they are copied to $FIG
RA=$REPO/rna_atac; RM=$REPO/rna_msi; HC=$REPO/human_cah
CAH=$OUT/human_cah_transfer/seed0/cah          # human caudate-head transfer of the seed-0 model
SHARED=$OUT/rna_msi_runs/shared_components     # shared PLS components of the RNA-MSI runs
MOUSE_RUN=seed3                                 # mouse RNA-MSI manuscript model: V11L12-109_B1_seed3
if [[ ${1:-} != list ]]; then mkdir -p "$FIG/supplementary" "$FIG/panels" "$WORK"; fi
log() { echo "[$(date '+%F %T')] $*"; }
gpu() { CUDA_VISIBLE_DEVICES=$GPU "$@"; }
cpu() { CUDA_VISIBLE_DEVICES= "$@"; }

# Training settings of the RNA-MSI models (from the training logs; docs/REPRODUCIBILITY.md).
msi_env() {   # <sample ids> <run tag> <seed>
    export SPATIALJEPA_SAMPLE_IDS=$1 SPATIALJEPA_RUN_TAG=$2 SPATIALJEPA_SEED=$3 SPATIALJEPA_MAX_EPOCH=1000 \
           SPATIALJEPA_N_TARGET_TOP=2000 SPATIALJEPA_TEACHER_NEIGHBORHOOD=exact SPATIALJEPA_TEACHER_EARLY_STOP=1
}
mouse_env() {
    msi_env V11L12-109_B1 "_seed3${RETRAIN_SUFFIX:-}" 3
    export SPATIALJEPA_FIG_DIR=$OUT/rna_msi_runs/seed3 \
           SPATIALJEPA_NMF_CACHE_DIR=$OUT/spatialjepa_projection/V11L12-109_B1   # reference run: NMF dopamine signature
    mkdir -p "$SPATIALJEPA_FIG_DIR"
}

# ===================================================================================== figures
fig_rna_atac_heldout() {          # Fig. 2f-g, Supplemental Fig. S4
    log "Fig. 2f-g: held-out Multiome placement"
    cpu "$PY_ANALYSIS" "$RA/plot_heldout_multiome_placement.py" "$WORK/fig2"
    log "Supplemental Fig. S4: placement scored against the published annotations"
    cpu "$PY_ANALYSIS" "$RA/plot_heldout_vs_published_annotations.py" "$WORK/s4"
}
fig_rna_atac_transfer() {         # Fig. 3 panels and assembly, Supplemental Figs. S5-S6
    log "Fig. 3a,c-e (and the PLS fits reused by S5-S6)"
    cpu "$PY_ANALYSIS" "$RA/plot_transfer_panels.py"
    log "Fig. 3b: DE concordance of transferred clusters"
    cpu "$PY_ANALYSIS" "$RA/plot_transfer_clusters.py"
    log "Fig. 3f-g: linked-peak maps and bivariate Moran's I"
    cpu "$PY_ANALYSIS" "$RA/plot_linked_peak_maps.py"
    log "Fig. 3 assembly"
    # renders the SVG with the system librsvg and cairo (ctypes); the motifs environment finds them
    cpu "$PY_MOTIF" "$RA/assemble_transfer_figure.py" --output "$WORK/fig3_assembly"
    log "Supplemental Figs. S5-S6: ATAC PLS maps and marker-gene weights"
    cpu "$PY_SCVI" "$RA/plot_pls_maps_and_markers.py" --output "$WORK/s5_s6"
}
fig_rna_atac_supplement() {       # Supplemental Figs. S1-S3
    log "Supplemental Fig. S1: integration metrics"
    (cd "$RA" && cpu "$PY_ANALYSIS" "$RA/plot_integration_metrics.py")
    log "Supplemental Fig. S2: spatial-structure metrics"
    # pastes onto the manuscript layout (deposited under outputs/manuscript_templates, apart from collect's output)
    (export SPATIALJEPA_MANUSCRIPT=$OUT/manuscript_templates
     cpu "$PY_SCVI" "$RA/plot_spatial_structure.py" --output "$WORK/s2")
    log "Supplemental Fig. S3: position recovery by latent"
    cpu "$PY_SCVI" "$RA/plot_position_recovery_by_latent.py" --output "$WORK/s3"
}
fig_rna_msi_dopamine() {          # Fig. 4, Supplemental Figs. S12-S14
    local W=$WORK/dopamine_program
    mkdir -p "$W/$MOUSE_RUN/motifs_gc"
    # weight-independent caches (peak GC, genome-wide motif scan); the motif test itself always reruns
    cp -n "$SHARED/manuscript_layout_msi_svd_intact/peak_gc.csv" "$W/$MOUSE_RUN/"
    cp -n "$SHARED/manuscript_layout_msi_svd_intact/motifs_gc/peak_motif_hits.pkl" "$W/$MOUSE_RUN/motifs_gc/"
    log "Fig. 4 and S12, first pass (motif ranking; the S13 board needs the S14 maps and stops)"
    (cd "$RM" && cpu "$PY_ANALYSIS" plot_dopamine_program.py --run "$MOUSE_RUN" --msi-weights ot --out-dir "$W") || true
    log "Supplemental Fig. S14: motif deviations mapped onto the source section"
    (cd "$RM/supplement" && gpu "$PY_ANALYSIS" plot_spatial_motif_maps.py --stage compute "$WORK/motif_maps" \
        --run "$MOUSE_RUN" --ranking "$W/$MOUSE_RUN/motifs_gc/pls_C2_gmm_d2msn_motif_ranking.csv")
    (cd "$RM/supplement" && cpu "$PY_ANALYSIS" plot_spatial_motif_maps.py --stage draw "$WORK/motif_maps" --run "$MOUSE_RUN")
    log "Fig. 4, S12 and S13"
    (cd "$RM" && cpu "$PY_ANALYSIS" plot_dopamine_program.py --run "$MOUSE_RUN" --msi-weights ot --out-dir "$W" \
        --motif-maps "$WORK/motif_maps")
}
fig_rna_msi_supplement() {        # Supplemental Figs. S7-S11
    log "Supplemental Figs. S7-S8: held-out animals"
    cpu "$PY_ANALYSIS" "$RM/plot_heldout_animals.py" "$WORK/s7_s8"
    log "Supplemental Fig. S9: teacher- and student-imputed dopamine"
    (cd "$RM/supplement" && FIG567_RUN=$MOUSE_RUN cpu "$PY_ANALYSIS" plot_imputed_dopamine_teacher_student.py \
        "$WORK/s9/candidates" "$WORK/s9/S_Fig5_p22_imputed_dopamine")
    log "Supplemental Fig. S10: dopamine dose-response over random latent directions"
    (cd "$RM/supplement" && gpu "$PY_ANALYSIS" null_random_latent_directions.py "$MOUSE_RUN" "$WORK/s10" \
        && cpu "$PY_ANALYSIS" null_rna_pca_directions.py "$MOUSE_RUN" "$WORK/s10" \
        && cpu "$PY_ANALYSIS" plot_dopamine_dose_response.py "$WORK/s10" "$WORK/s10" "$MOUSE_RUN")
    log "Supplemental Fig. S11: metabolite-weight recipe over random directions"
    (cd "$RM/supplement" && gpu "$PY_ANALYSIS" null_ot_metabolite_weights.py "$WORK/s11" --run "$MOUSE_RUN")
}
fig_human_cah() {                 # Fig. 5, Supplemental Figs. S15-S17
    mkdir -p "$CAH/astro_c1_adnc" "$CAH/supp_metabolite_transfer" "$CAH/figures_pls_msi_ot_reg0.001_sec"
    cd "$CAH/astro_c1_adnc"
    log "Fig. 5: per-nucleus and per-donor tables";  cpu "$PY_ANALYSIS" "$HC/build_cell_table.py"
    cpu "$PY_ANALYSIS" "$HC/build_donor_table.py";  cpu "$PY_ANALYSIS" "$HC/add_early_cell_columns.py" cells.parquet
    log "Fig. 5d: astrocyte C1 by ADNC";             cpu "$PY_ANALYSIS" "$HC/astrocyte_c1_adnc.py"
    log "Fig. 5h: astrocyte C1 carried to Xenium"
    cpu "$PY_ANALYSIS" "$HC/astrocyte_c1_ridge.py"; cpu "$PY_ANALYSIS" "$HC/astrocyte_c1_checks.py"
    cpu "$PY_ANALYSIS" "$HC/astrocyte_c1_count_adjusted.py"
    log "Fig. 5";                                     cpu "$PY_ANALYSIS" "$HC/plot_cah_summary.py"
    log "Supplemental Figs. S15-S17"
    gpu "$PY_ANALYSIS" "$HC/metabolite_transfer_supplement.py"
    cpu "$PY_ANALYSIS" "$HC/component_c2.py";        cpu "$PY_ANALYSIS" "$HC/astrocyte_c2_xenium.py"
    cpu "$PY_ANALYSIS" "$HC/plot_metabolite_transfer_c1.py"; cpu "$PY_ANALYSIS" "$HC/plot_component_c2.py"
    cpu "$PY_ANALYSIS" "$HC/plot_metabolite_transfer_c2.py"
    cd "$REPO"
}
collect() {                       # copy the figures under the manuscript's file names
    local W=$WORK/dopamine_program/$MOUSE_RUN F=$CAH/figures_pls_msi_ot_reg0.001_sec S=$CAH/supp_metabolite_transfer
    local pairs=(
        "$W/SMA_mouse_dopamine_nmf_pls.png:Fig5_mouse_dopamine_component_early.png"
        "$F/sea_ad_cah_pls_astro.png:Fig7_human_cah_early_s0_astro.png"
        "$WORK/s2/3_spatial_ecdf_analysis_joint.png:3_spatial_ecdf_analysis.png"
        "$WORK/s3/S_Fig3_ecdf_by_latent.png:supplementary/spatial_domains.png"
        "$WORK/s4/FigS_heldout_published_reference.png:supplementary/heldout_published_reference.png"
        "$WORK/s5_s6/S_Fig4_atac_pls_maps.png:supplementary/atac_pls_maps.png"
        "$WORK/s5_s6/S_Fig4_pls_marker_genes.png:supplementary/pls_marker_genes.png"
        "$WORK/s7_s8/S_heldout_rnamsi_agreement.png:supplementary/heldout_rnamsi_agreement.png"
        "$WORK/s7_s8/S_heldout_rnamsi_fig5def.png:supplementary/heldout_rnamsi_fig5def.png"
        "$WORK/s9/S_Fig5_p22_imputed_dopamine.png:p22_imputed_dopamine_teacher_student.png"
        "$WORK/s10/draft_dopamine_dose_supplement.png:supplementary/dopamine_dose_rnapca_control.png"
        "$WORK/s11/S_Fig5_ot_direction_null.png:supplementary/ot_direction_null.png"
        "$W/dopamine_rna_weight_gmm.png:dopamine_rna_weight_gmm_pls.png"
        "$W/SMA_mouse_pls_enrichment.png:Fig6_mouse_regulatory_early_row.png"
        "$WORK/motif_maps/Fig6_draft_spatial_motif_maps.png:supplementary/spatial_motif_maps.png"
        "$S/supp_metabolite_transfer.png:supplementary/cah_metabolite_transfer_c1.png"
        "$S/supp_pls_c2.png:supplementary/cah_pls_c2.png"
        "$S/supp_metabolite_transfer_c2.png:supplementary/cah_metabolite_transfer_c2.png"
        "$WORK/fig2/Fig2f_heldout_multiome_row.png:panels/fig2fg_heldout_multiome.png"
        "$WORK/fig3_assembly/figure4_assembled.png:panels/fig3_assembled.png"
    )
    for p in "${pairs[@]}"; do
        local src=${p%%:*} dst=$FIG/${p#*:}
        if [[ -f $src ]]; then cp "$src" "$dst"; echo "  $dst"; else echo "  MISSING $src"; fi
    done
    echo "  $FIG/supplementary/S1_P22_scIB_v3.png (written by plot_integration_metrics.py)"
}

# ===================================================================================== analyses
# Commands that produced the deposited intermediates (from the original launchers; docs/REPRODUCIBILITY.md).
ana_rna_msi_mouse() {             # benchmark (Fig. 4d-f), target encodings, shared components, chromVAR
    mouse_env
    (cd "$RM" && JAX_PLATFORMS=cpu cpu "$PY_ANALYSIS" dopamine_embedding_benchmark.py --sample-id V11L12-109_B1)
    (cd "$RM" && gpu "$PY_MSI" encode_targets.py)
    for t in spatial multiome; do
        (cd "$RM" && gpu "$PY_ANALYSIS" shared_msi_components.py --target $t --run "$MOUSE_RUN" --tables-only)
    done
    (cd "$RM" && cpu "$PY_ANALYSIS" multiome_motif_deviations.py --stage assoc --engine pychromvar \
        --components msi_intact --run "$MOUSE_RUN")
}
ana_human_cah() {                 # SEA-AD caudate-head transfer for the three human seeds
    for s in 0 1 2; do
        (cd "$HC" && gpu "$PY_ANALYSIS" cah_cell_runner.py --stage impute \
            --model-dir "$OUT/spatialjepa_models/V11T17-102_seed$s" --out-root "$OUT/human_cah_transfer/seed$s")
        for stage in components validate; do
            (cd "$HC" && gpu "$PY_ANALYSIS" cah_components.py --stage $stage --run seed$s)
        done
    done
}
ana_rna_atac() {                  # held-out Multiome placement (Fig. 2f-g, S4)
    gpu "$PY_ANALYSIS" "$RA/encode_heldout_controls.py"
    cpu "$PY_ANALYSIS" "$RA/heldout_multiome_alignment.py" --clustering leiden
    cpu "$PY_ANALYSIS" "$RA/heldout_multiome_alignment.py" --clustering leiden --summarize
}
ana_heldout_animals() (           # cross-animal benchmark behind S7-S8 (subshell: settings stay local)
    # Trains the four queues of the manuscript (striatum, seeds 0-2, 2,000-gene panel, 1,000 epochs; 162 runs, one
    # worker per GPU in GPUS, default $GPU), then writes the tables plot_heldout_animals.py reads, in the order they
    # were made: student-teacher diagnostics, dopamine structure, teacher consistency, cluster agreement.
    B=$OUT/spatialjepa_benchmark
    export ANIMAL_TRANSFER_PANEL_TOP=2000 MAX_EPOCH=1000 REGION=striatum SEEDS="0 1 2" MPLBACKEND=Agg
    for spec in "animal_transfer_p2000 heldout both A B G" "animal_transfer_p2000 early pairwise A_early A_wd" \
                "animal_transfer_p2000_split split split A B G" "animal_transfer_p2000_split early_split split A_early A_wd"; do
        read -r root queue design configs <<< "$spec"
        log "held-out animals: training queue $queue ($root)"
        export ANIMAL_TRANSFER_BENCH_DIR=$B/$root ANIMAL_TRANSFER_RUNS_DIR=$B/$root DESIGN=$design CONFIGS="$configs" \
               QUEUE_NAME=$queue
        "$RM/heldout_animals_run.sh" plan
        "$RM/heldout_animals_run.sh" prepare
        for g in ${GPUS:-$GPU}; do
            CUDA_VISIBLE_DEVICES=$g "$RM/heldout_animals_worker.sh" "$B/$root/queue/$queue" "local_gpu$g" &
        done
        wait
    done
    for root in animal_transfer_p2000 animal_transfer_p2000_split; do
        log "held-out animals: student-teacher diagnostics and dopamine structure ($root)"
        export ANIMAL_TRANSFER_BENCH_DIR=$B/$root ANIMAL_TRANSFER_RUNS_DIR=$B/$root
        gpu "$PY_MSI" "$RM/student_teacher_diagnostics.py" --runs-dir "$B/$root" --configs A B --device cuda:0
        gpu "$PY_MSI" "$RM/student_teacher_diagnostics.py" --runs-dir "$B/$root" --configs A_early A_wd --base-only \
            --device cuda:0
        "$PY_MSI" "$RM/student_teacher_diagnostics.py" --runs-dir "$B/$root" --aggregate
        "$RM/heldout_animals_run.sh" structure
    done
    log "held-out animals: teacher consistency"
    gpu "$PY_MSI" "$RM/teacher_consistency.py" --part both --device cuda:0
    for root in animal_transfer_p2000 animal_transfer_p2000_split; do
        log "held-out animals: cluster agreement ($root)"
        ANIMAL_TRANSFER_BENCH_DIR=$B/$root ANIMAL_TRANSFER_RUNS_DIR=$B/$root OMP_NUM_THREADS=8 \
            "$PY_MSI" "$RM/cluster_agreement.py" --runs-dir "$B/$root"
    done
)

# ===================================================================================== training
train_rna_msi_mouse() {           # V11L12-109_B1_seed3 (bit-identical to the deposited model on our hardware)
    mouse_env
    (cd "$RM" && gpu "$PY_MSI" train_rna_msi.py)
}
train_rna_msi_human() {           # V11T17-102_seed0/1/2 (sections A1, C1, D1 of donor hPD1)
    for s in 0 1 2; do
        (msi_env V11T17-102_A1,V11T17-102_C1,V11T17-102_D1 "_seed$s${RETRAIN_SUFFIX:-}" $s; cd "$RM" && gpu "$PY_MSI" train_rna_msi.py)
    done
}
train_rna_atac() {                # MultiGATE-backbone teacher and student (MLflow run 20260505_190037)
    echo "rna_atac/train_rna_atac.py logs to MLflow (set MLFLOW_TRACKING_URI). The manuscript run's parameters are"
    echo "in \$SPATIALJEPA_HOME/outputs/rna_atac_models/20260505_190037/run.json (46 values, from MLflow); the run"
    echo "trained with MultiGATE commit a17e77a (docs/REPRODUCIBILITY.md). Retraining takes about 7 h on one GPU."
}

STAGE_figures=(fig_rna_atac_heldout fig_rna_atac_transfer fig_rna_atac_supplement fig_rna_msi_dopamine
               fig_rna_msi_supplement fig_human_cah collect)
STAGE_analyses=(ana_rna_atac ana_rna_msi_mouse ana_human_cah ana_heldout_animals)
STAGE_train=(train_rna_msi_mouse train_rna_msi_human train_rna_atac)

[[ $# -eq 0 ]] && set -- figures
for arg in "$@"; do
    case $arg in
        figures)  for s in "${STAGE_figures[@]}"; do $s; done ;;
        analyses) for s in "${STAGE_analyses[@]}"; do $s; done ;;
        train)    for s in "${STAGE_train[@]}"; do $s; done ;;
        list)     printf '%s\n' "figures:" "${STAGE_figures[@]/#/  }" "analyses:" "${STAGE_analyses[@]/#/  }" \
                                "train:" "${STAGE_train[@]/#/  }" ;;
        *)        declare -F "$arg" > /dev/null || { echo "unknown step: $arg (./reproduce.sh list)" >&2; exit 2; }
                  $arg ;;
    esac
done

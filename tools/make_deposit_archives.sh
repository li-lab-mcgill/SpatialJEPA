#!/usr/bin/env bash
# Pack a staged SPATIALJEPA_HOME deposit into the archives uploaded to Zenodo (at most 100 files per record),
# one at a time, with their SHA-256 sums. Unpack them all into SPATIALJEPA_HOME to reproduce the manuscript.
#   tools/make_deposit_archives.sh DEPOSIT_DIR OUT_DIR [archive ...]
set -euo pipefail
DEP=$(realpath "${1:?deposit dir}"); OUT=$(realpath -m "${2:?output dir}"); shift 2
mkdir -p "$OUT"
declare -A PARTS=(
  [rna_atac]="data/aligned_data outputs/MultiGATE outputs/rna_atac_models outputs/rna_atac_figures outputs/rna_atac_transfer_clusters outputs/multivi_mouse_brain_spatial_rna_atac outputs/multivi_mouse_brain_spatial_rna_atac_target"
  [rna_msi_models]="data/vicari_2023 outputs/spatialjepa_models outputs/rna_msi_runs outputs/spatialjepa_benchmark outputs/multigrate_mouse_sma outputs/target_gene_rankings outputs/manuscript_templates"
  [rna_msi_targets_seed3]="outputs/spatialjepa_projection/V11L12-109_B1_seed3"
  [rna_msi_targets_reference]="outputs/spatialjepa_projection/V11L12-109_B1"
  [human_cah]="outputs/human_cah_transfer"
)
names=${*:-"rna_atac rna_msi_models rna_msi_targets_seed3 rna_msi_targets_reference human_cah"}
for n in $names; do
    paths=(); for p in ${PARTS[$n]}; do [[ -e $DEP/$p ]] && paths+=("$p"); done
    echo "[archive] spatialjepa_$n.tar: ${paths[*]}"
    tar -C "$DEP" --dereference -cf "$OUT/spatialjepa_$n.tar" "${paths[@]}"
    (cd "$OUT" && sha256sum "spatialjepa_$n.tar" >> SHA256SUMS)
done
cp "$DEP/MANIFEST.tsv" "$OUT/"
# every staged file must be in exactly one archive
python3 - "$DEP" "${PARTS[rna_atac]} ${PARTS[rna_msi_models]} ${PARTS[rna_msi_targets_seed3]} ${PARTS[rna_msi_targets_reference]} ${PARTS[human_cah]}" <<'PY'
import sys, pathlib
dep, prefixes = pathlib.Path(sys.argv[1]), sys.argv[2].split()
rows = [l.split("\t")[0] for l in (dep / "MANIFEST.tsv").read_text().splitlines()[1:]]
miss = [r for r in rows if not any(r == p or r.startswith(p + "/") for p in prefixes)]
print("files outside the archives:", miss if miss else "none")
PY

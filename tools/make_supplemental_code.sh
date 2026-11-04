#!/usr/bin/env bash
# Supplemental_Code.zip for the journal: the repository at its current state (no data, no git history).
#   tools/make_supplemental_code.sh [OUT_ZIP]
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=$(realpath -m "${1:-$REPO/../Supplemental_Code.zip}")
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
rsync -a --exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude configs/spatialjepa.env \
      --exclude .pytest_cache --exclude 'condaenv.*.requirements.txt' "$REPO/" "$TMP/SpatialJEPA/"
rm -f "$OUT"; (cd "$TMP" && zip -qr "$OUT" SpatialJEPA)
echo "wrote $OUT ($(stat -c %s "$OUT" | numfmt --to=iec)B, $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files)"

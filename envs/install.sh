#!/usr/bin/env bash
# Create the conda environments of the analyses:
#   envs/install.sh                      all five (msi analysis scvi multigrate motifs)
#   envs/install.sh msi analysis         some of them
# Each environment is created from envs/spatialjepa-<name>.yml. The packages listed for it in envs/no-deps.txt are
# then added with `pip install --no-deps --no-build-isolation`; that file explains why. Finally the diffs in
# envs/patches/<environment>/ are applied to the installed packages (envs/patches/README.md).
# Needs conda (CONDA=mamba to use mamba), GNU patch, and a C/C++ compiler (MOODS-python and pybedtools are built
# from source).
# ENVS_DIR=<folder> creates the environments under <folder>/spatialjepa-<name> instead of by name.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONDA=${CONDA:-conda}
export PYTHONNOUSERSITE=1          # packages in ~/.local must not satisfy (or shadow) the pinned ones
for e in ${*:-msi analysis scvi multigrate motifs}; do
    name=spatialjepa-$e
    [[ -f $name.yml ]] || { echo "unknown environment: $e" >&2; exit 2; }
    if [[ -n ${ENVS_DIR:-} ]]; then target=(-p "$ENVS_DIR/$name"); else target=(-n "$name"); fi
    echo "[install] $name"
    "$CONDA" env create "${target[@]}" -f "$name.yml"
    mapfile -t reqs < <(awk -v n="$name" '$1 == n { $1 = ""; sub(/^ +/, ""); print }' no-deps.txt)
    if (( ${#reqs[@]} )); then       # editable git checkouts go to <environment>/src
        "$CONDA" run "${target[@]}" bash -c \
            'python -m pip install --no-deps --no-build-isolation --src "$CONDA_PREFIX/src" "$@"' no-deps "${reqs[@]}"
    fi
    if compgen -G "patches/$name/*.diff" > /dev/null; then   # local edits of installed packages (patches/README.md)
        site=$("$CONDA" run "${target[@]}" python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
        # one file per diff, written through stdout: patching in place copies file ownership, which some
        # network file systems refuse
        for p in "patches/$name"/*.diff; do
            f=$(sed -n 's|^+++ b/\([^[:space:]]*\).*|\1|p' "$p" | head -1)
            echo "[install] $name: $p -> $f"
            patch -s -p1 -d "${site:?}" -o - < "$p" > "${site:?}/${f:?}.patched"
            cat "${site:?}/${f:?}.patched" > "${site:?}/${f:?}"
            rm -f -- "${site:?}/${f:?}.patched"
        done
    fi
done

"""Load aligned RNA+MSI MuData exported from se.multi.list."""

from __future__ import annotations
try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402

from pathlib import Path

import mudata as mu

from build_mudata_from_mtx import load_sample_from_mtx


EXPORT_DIR = Path(f"{os.environ['SPATIALJEPA_HOME']}/outputs/SMA/h5mu_export")


def load_sample(sample_id: str, export_dir: Path = EXPORT_DIR) -> mu.MuData:
    h5mu_path = export_dir / f"{sample_id}.h5mu"
    if h5mu_path.exists():
        return mu.read_h5mu(h5mu_path)

    mtx_dir = export_dir / sample_id
    if mtx_dir.is_dir():
        return load_sample_from_mtx(mtx_dir)

    available = sorted(p.stem for p in export_dir.glob("*.h5mu"))
    raise FileNotFoundError(
        f"No export found for {sample_id}. Available: {', '.join(available)}"
    )


def list_samples(export_dir: Path = EXPORT_DIR) -> list[str]:
    return sorted(p.stem for p in export_dir.glob("*.h5mu"))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    try:
        import scanpy as sc
    except Exception as exc:
        sc = None
        print(f"scanpy import failed; skipping spatial plots: {exc}")

    for sample_id in list_samples():

        mdata = load_sample(sample_id)
        rna = mdata.mod["rna"]
        msi = mdata.mod["msi"]
        print(
            f"{sample_id}: {rna.n_obs} spots, "
            f"{rna.n_vars} genes, {msi.n_vars} MSI features"
        )

        assert rna.obs_names.equals(msi.obs_names)
        assert (rna.obsm['spatial'] == msi.obsm['spatial']).all()

        if sc is not None and "spatial" in rna.obsm:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            sc.pl.spatial(
                rna,
                color="RNA_clusters",
                basis="spatial",
                show=False,
                s=0.75,
                ax=axes[0],
            )
            sc.pl.spatial(
                msi,
                color="MM_clusters",
                basis="spatial",
                show=False,
                s=0.5,
                ax=axes[1],
            )
            plt.suptitle(sample_id)
            plt.tight_layout()
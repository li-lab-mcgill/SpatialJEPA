try:  # repository root on sys.path, then the settings (spatialjepa/paths.py, configs/spatialjepa.env)
    import spatialjepa.paths  # noqa: F401
except ModuleNotFoundError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import spatialjepa.paths  # noqa: F401
import os  # noqa: E402

#%%
import anndata as ad
from pathlib import Path

def main():
    DATA_ROOT = Path(os.environ["SPATIALJEPA_DATA"])

    cah_all = ad.read_h5ad(DATA_ROOT / "sea_ad_cah" / "CaH_all_nuclei.2025-12-29.h5ad", backed="r")
    final = cah_all[cah_all.obs['Used in analysis']].to_memory()
    final.write_h5ad(DATA_ROOT / "sea_ad_cah" / "CaH_final_nuclei.2025-12-29.h5ad")
    print("Final dataset written to: ", DATA_ROOT / "sea_ad_cah" / "CaH_final_nuclei.2025-12-29.h5ad")

    cah_xenium_all = ad.read_h5ad(DATA_ROOT / "sea_ad_cah" / "CaH_Xenium.2026-01-07.h5ad", backed="r")
    cah_xenium_final = cah_xenium_all[cah_xenium_all.obs['Used in analysis']].to_memory()
    cah_xenium_final.write_h5ad(DATA_ROOT / "sea_ad_cah" / "CaH_Xenium_final.2026-01-07.h5ad")
    print("Final Xenium dataset written to: ", DATA_ROOT / "sea_ad_cah" / "CaH_Xenium_final.2026-01-07.h5ad")

if __name__ == "__main__":
    main()

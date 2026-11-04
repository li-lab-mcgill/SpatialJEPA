#%%
import anndata as ad
from muon import MuData
from pybedtools import BedTool
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
from pyranges import PyRanges, read_gtf
import os
import subprocess
from scipy.sparse import eye

from nichecompass_utils import CustomNicheCompass

__all__ = ["DataAligner"]

def _safe_intersect_obs(mdata: MuData, *, modalities: list[str] | None = None) -> MuData:
    """Intersect obs across modalities without breaking aligned mappings.

    `muon.pp.intersect_obs` can leave stale `.obsm`/`.layers` entries when it
    reindexes `.obs` in-place. Here we subset with standard AnnData slicing,
    which keeps aligned mappings consistent.
    """
    mods = list(mdata.mod.keys()) if modalities is None else [m for m in modalities if m in mdata.mod]
    if len(mods) <= 1:
        return mdata

    base = mdata.mod[mods[0]].obs_names
    common = base
    for mod in mods[1:]:
        common = common.intersection(mdata.mod[mod].obs_names)

    # Preserve base modality order for stable downstream alignment.
    common_ordered = base[base.isin(common)]
    if common_ordered.size == base.size and all(
        mdata.mod[mod].n_obs == common_ordered.size for mod in mods
    ):
        return mdata

    new_modalities = {k: v[common_ordered].copy() for k, v in mdata.mod.items()}
    out = MuData(new_modalities)

    # Preserve MuData-level obs columns when possible.
    try:
        out.obs = mdata.obs.loc[common_ordered].copy()
    except Exception:
        pass

    # Preserve MuData-level metadata containers best-effort.
    try:
        out.uns = dict(getattr(mdata, "uns", {}))
    except Exception:
        pass

    return out

class DataAligner:
    def __init__(
        self,
        source_data: ad.AnnData | MuData,
        target_data: ad.AnnData | MuData,
        source_assembly: str,
        target_assembly: str,
        source_name: str = "source",
        target_name: str = "target",
        paired_target: bool = True,
    ):

        ## force source data to share same obs_names across modalitiies
        if not source_data['rna'].obs_names.equals(source_data['atac'].obs_names):
            print(f"[INFO] Intersecting source data obs_names across modalities")
            source_data = _safe_intersect_obs(source_data, modalities=["rna", "atac"])
        if paired_target and (not target_data['rna'].obs_names.equals(target_data['atac'].obs_names)):
            print(f"[INFO] Intersecting target data obs_names across modalities")
            target_data = _safe_intersect_obs(target_data, modalities=["rna", "atac"])

        ## set metadata
        source_data.obs["dataset_name"] = source_name
        target_data.obs["dataset_name"] = target_name
        source_data.obs["assembly"] = source_assembly
        target_data.obs["assembly"] = target_assembly

        ## set TSS if 'rna:strand' is present
        if 'strand' in source_data['rna'].var.columns:
            source_data['rna'].var['TSS'] = source_data['rna'].var.apply(lambda x: x['chromStart'] if x['strand'] == '+' else x['chromEnd'] if x['strand'] == '-' else np.nan, axis=1)
        if 'strand' in target_data['rna'].var.columns:
            target_data['rna'].var['TSS'] = target_data['rna'].var.apply(lambda x: x['chromStart'] if x['strand'] == '+' else x['chromEnd'] if x['strand'] == '-' else np.nan, axis=1)

        ## assign data to self
        self.source_data = source_data
        self.target_data = target_data

        ## check that data are from the same assembly
        if self.source_data.obs["assembly"].unique()[0] == self.target_data.obs["assembly"].unique()[0]:
            print(f"Source and target data are from the same assembly: {self.source_data.obs['assembly'].unique()[0]}")
        else:
            print(f"Source and target data are from different assemblies: {self.source_data.obs['assembly'].unique()[0]} and {self.target_data.obs['assembly'].unique()[0]}")
            from pyliftover import LiftOver
            self.do_liftOver()

        ## check that all gene names are unique
        assert self.source_data['rna'].var_names.is_unique, "Source RNA data must have unique gene names"
        assert self.target_data['rna'].var_names.is_unique, "Target RNA data must have unique gene names"

        ## check that all gene names are str and not castable as int
        assert all(isinstance(name, str) and not (name.lstrip('-+').isdigit() and name.lstrip('-+') != '') for name in self.source_data['rna'].var_names), "Gene names must be strings and not castable as int"
        assert all(isinstance(name, str) and not (name.lstrip('-+').isdigit() and name.lstrip('-+') != '') for name in self.target_data['rna'].var_names), "Gene names must be strings and not castable as int"

        ## check proper column names for ATAC peak intervals
        assert np.isin(['chrom', 'chromStart', 'chromEnd'], self.source_data['atac'].var.columns).all(), "ATAC data must have chrom, chromStart, and chromEnd columns"
        assert np.isin(['chrom', 'chromStart', 'chromEnd'], self.target_data['atac'].var.columns).all(), "ATAC data must have chrom, chromStart, and chromEnd columns"

        # Normalize ATAC interval dtypes to avoid string/float formatting mismatches
        for mdata in (self.source_data, self.target_data):
            atac_var = mdata["atac"].var
            atac_var["chrom"] = atac_var["chrom"].astype(str)
            atac_var["chromStart"] = pd.to_numeric(atac_var["chromStart"], errors="raise").astype(int)
            atac_var["chromEnd"] = pd.to_numeric(atac_var["chromEnd"], errors="raise").astype(int)

        ## set var_names to BED format for ATAC data
        self.source_data['atac'].var_names = self.source_data['atac'].var.apply(lambda x: f"{x['chrom']}:{x['chromStart']}-{x['chromEnd']}", axis=1)
        self.target_data['atac'].var_names = self.target_data['atac'].var.apply(lambda x: f"{x['chrom']}:{x['chromStart']}-{x['chromEnd']}", axis=1)

    def find_gene_overlap(self):
        source_genes = self.source_data['rna'].var_names.tolist()
        target_genes = self.target_data['rna'].var_names.tolist()
        self.gene_overlap = list(set(source_genes) & set(target_genes))

    def find_peak_overlap(self, peak_name_mapper_as_dict: bool = False):
        source_atac = self.source_data['atac']
        target_atac = self.target_data['atac']
        source_peaks = BedTool.from_dataframe(source_atac.var[["chrom", "chromStart", "chromEnd"]])
        target_peaks = BedTool.from_dataframe(target_atac.var[["chrom", "chromStart", "chromEnd"]])
        peak_overlap = source_peaks.intersect(target_peaks, wa=True, wb=True).to_dataframe() # Intersect with -wa -wb to return intervals from both sets

        # Group by target peak coordinates and select the row with highest "dispersion_norm" within each group
        # First, rename columns to identify source and target peaks
        peak_overlap.rename(columns={
            "chrom": "chrom_source", "start": "start_source", "end": "end_source",
            "name": "chrom_target", "score": "start_target", "strand": "end_target"
        }, inplace=True)
        
        idx_target = None
        idx_source = None

        if "dispersions_norm" in target_atac.var.columns:
            merged_target = (
                peak_overlap[['chrom_target', 'start_target', 'end_target']].astype(str)
                .merge(
                    target_atac.var.reset_index().astype(str),
                    left_on=['chrom_target', 'start_target', 'end_target'],
                    right_on=['chrom', 'chromStart', 'chromEnd'],
                    how='left'
                )
            )
            idx_target = merged_target.groupby(['chrom', 'chromStart', 'chromEnd'])['dispersions_norm'].idxmax()

        if "dispersions_norm" in source_atac.var.columns:
            merged_source = (
                peak_overlap[['chrom_source', 'start_source', 'end_source']].astype(str)
                .merge(
                    source_atac.var.reset_index().astype(str),
                    left_on=['chrom_source', 'start_source', 'end_source'],
                    right_on=['chrom', 'chromStart', 'chromEnd'],
                    how='left'
                )
            )
            idx_source = merged_source.groupby(['chrom', 'chromStart', 'chromEnd'])['dispersions_norm'].idxmax()

        if (idx_target is not None) and (idx_source is not None):
            idx = pd.concat([idx_target, idx_source])
        elif idx_target is not None:
            idx = idx_target
        elif idx_source is not None:
            idx = idx_source
        else:
            idx = None

        if idx is not None:
            idx = idx.dropna().astype(int)
            peak_overlap = peak_overlap.loc[idx.values]

        ## could also use mmn.atac.pp.add_positions(mdata['atac'])
        peak_overlap = peak_overlap.assign(
            peak_name_source = peak_overlap.apply(lambda x: f"{x['chrom_source']}:{x['start_source']}-{x['end_source']}", axis=1),
            peak_name_target = peak_overlap.apply(lambda x: f"{x['chrom_target']}:{x['start_target']}-{x['end_target']}", axis=1)
        )

        # Enforce a one-to-one mapping (greedy) to avoid duplicates causing silent drops later
        peak_overlap["overlap_len"] = (
            np.minimum(peak_overlap["end_source"].astype(int), peak_overlap["end_target"].astype(int))
            - np.maximum(peak_overlap["start_source"].astype(int), peak_overlap["start_target"].astype(int))
        )
        peak_overlap = peak_overlap.sort_values("overlap_len", ascending=False)
        peak_overlap = peak_overlap.drop_duplicates(subset=["peak_name_source"], keep="first")
        peak_overlap = peak_overlap.drop_duplicates(subset=["peak_name_target"], keep="first")

        peak_name_mapper = peak_overlap.set_index("peak_name_source").loc[:, "peak_name_target"]

        print(f"Number of duplicate peak name mappings: {peak_name_mapper.index.value_counts().ge(2).sum()}")
        self.peak_overlap_df = peak_overlap
        self.peak_name_mapper = peak_name_mapper.to_dict() if peak_name_mapper_as_dict else peak_name_mapper # to_dict() ignored because of duplicate values


    def align_features_by_overlap(self):
        source_rna = self.source_data['rna']
        target_rna = self.target_data['rna']
        source_atac = self.source_data['atac']
        target_atac = self.target_data['atac']

        # IMPORTANT: avoid get_indexer() here; -1 values silently select the last column and misalign features
        gene_overlap = [g for g in self.gene_overlap if (g in source_rna.var_names) and (g in target_rna.var_names)]
        source_rna = source_rna[:, gene_overlap].copy()
        target_rna = target_rna[:, gene_overlap].copy()
        assert (source_rna.var_names == target_rna.var_names).all(), "Source and target RNA data must have the same gene names"

        peak_name_mapper = self.peak_name_mapper
        if isinstance(peak_name_mapper, dict):
            peak_name_mapper = pd.Series(peak_name_mapper)

        # Filter to mappings that exist in both objects
        peak_name_mapper = peak_name_mapper.loc[
            peak_name_mapper.index.isin(source_atac.var_names)
            & peak_name_mapper.isin(target_atac.var_names)
        ]
        # Ensure uniqueness on both sides (defensive; find_peak_overlap tries to enforce this)
        peak_name_mapper = peak_name_mapper[~peak_name_mapper.index.duplicated(keep="first")]
        peak_name_mapper = peak_name_mapper[~peak_name_mapper.duplicated(keep="first")]

        # Subset by names (not integer positions) so we cannot accidentally select "-1" columns
        source_atac = source_atac[:, peak_name_mapper.index.tolist()].copy()
        target_atac = target_atac[:, peak_name_mapper.values.tolist()].copy()

        # Rename source peaks to the chosen target peak names (so names must match exactly)
        source_atac.var_names = pd.Index(peak_name_mapper.values.tolist())

        print('Proportion of overlapping peak names: ', (source_atac.var_names == target_atac.var_names).mean())
        assert (source_atac.var_names == target_atac.var_names).all(), "Source and target ATAC data must have the same peak names"

        # Preserve obs columns (e.g., 'assembly', 'dataset_name') when creating new MuData objects
        # MuData.obs is derived from modalities, so we need to explicitly preserve columns
        source_obs = self.source_data.obs.copy()  # Use RNA obs as reference (should match ATAC obs)
        target_obs = self.target_data.obs.copy()
        
        # Verify that RNA and ATAC obs indices match (they should for multimodal data)
        assert source_rna.obs_names.equals(source_atac.obs_names), "Source RNA and ATAC must have matching obs_names"
        
        self.source_data = MuData({"rna": source_rna, "atac": source_atac})
        self.target_data = MuData({"rna": target_rna, "atac": target_atac})
        
        # Explicitly set obs to preserve all columns (indices should match)
        assert self.source_data.obs_names.equals(source_obs.index), "MuData obs_names must match preserved obs index"
        assert self.target_data.obs_names.equals(target_obs.index), "MuData obs_names must match preserved obs index"
        self.source_data.obs = source_obs
        self.target_data.obs = target_obs

    @staticmethod
    def copy_annotations_to_target(
        source_rna: ad.AnnData,
        source_atac: ad.AnnData,
        target_rna: ad.AnnData,
        target_atac: ad.AnnData
    ):
        """
        Copy varm and uns annotations from source to target data.
        
        This static method copies gene program masks, indices, and other annotations
        from the aligned source data to the aligned target data. This is necessary
        because the target data needs the same annotations (e.g., GP masks, gene/peak
        indices) as the source data for model compatibility.
        
        This is a static method because it's often called after additional data
        manipulations (e.g., rebuilding GP masks) that happen outside of DataAligner.
        
        Parameters
        ----------
        source_rna : AnnData
            Source RNA AnnData object with annotations to copy
        source_atac : AnnData
            Source ATAC AnnData object with annotations to copy
        target_rna : AnnData
            Target RNA AnnData object (will be modified in place)
        target_atac : AnnData
            Target ATAC AnnData object (will be modified in place)
            
        Returns
        -------
        target_rna : AnnData
            Modified target RNA object (same object, mutated in place)
        target_atac : AnnData
            Modified target ATAC object (same object, mutated in place)
        """
        # Copy varm (variable metadata) from source to target
        for var_key in source_rna.varm.keys():
            try:
                target_rna.varm[var_key] = source_rna.varm[var_key].copy()
            except Exception as e:
                print(f"Could not copy RNA var_key '{var_key}' from source to target: {e}")
        
        for var_key in source_atac.varm.keys():
            try:
                target_atac.varm[var_key] = source_atac.varm[var_key].copy()
            except Exception as e:
                print(f"Could not copy ATAC var_key '{var_key}' from source to target: {e}")

        # Copy uns (unstructured metadata) from source to target
        for uns_key in source_rna.uns.keys():
            try:
                target_rna.uns[uns_key] = source_rna.uns[uns_key].copy()
            except Exception as e:
                print(f"Could not copy RNA uns_key '{uns_key}' from source to target: {e}")
        
        for uns_key in source_atac.uns.keys():
            try:
                target_atac.uns[uns_key] = source_atac.uns[uns_key].copy()
            except Exception as e:
                print(f"Could not copy ATAC uns_key '{uns_key}' from source to target: {e}")
        
        return target_rna, target_atac

    @staticmethod
    def set_target_spatial_connectivities(
        target_rna: ad.AnnData,
        target_atac: ad.AnnData,
        adj_type: str = "identity"
    ):
        """
        Set spatial_connectivities in obsp from connectivities for target data.
        
        This ensures that the target data has the spatial connectivity matrix
        in the expected key ('spatial_connectivities') that NicheCompass expects.
        
        This is a static method because it's often called after additional data
        manipulations that happen outside of DataAligner.
        
        Parameters
        ----------
        target_rna : AnnData
            Target RNA AnnData object (will be modified in place)
        target_atac : AnnData
            Target ATAC AnnData object (will be modified in place)
            
        Returns
        -------
        target_rna : AnnData
            Modified target RNA object (same object, mutated in place)
        target_atac : AnnData
            Modified target ATAC object (same object, mutated in place)
        """

        if (adj_type == "knn") and ('connectivities' in target_rna.obsp) and ('connectivities' in target_atac.obsp):
            target_rna.obsp['spatial_connectivities'] = target_rna.obsp['connectivities'].copy()
            target_atac.obsp['spatial_connectivities'] = target_atac.obsp['connectivities'].copy()

        elif (adj_type == "identity"):
            target_rna.obsp['spatial_connectivities'] = eye(target_rna.n_obs, format="csr")
            target_atac.obsp['spatial_connectivities'] = eye(target_atac.n_obs, format="csr")

        else:
            print("Warning: 'connectivities' not found in target_rna.obsp or target_atac.obsp. "
                  "Spatial connectivities not set for RNA or ATAC data.")
        
        return target_rna, target_atac

    def do_liftOver(self):
        print(f"Lifting over source data to target assembly: {self.source_data.obs['assembly'].unique()[0]} to {self.target_data.obs['assembly'].unique()[0]}")

        lo = LiftOver(self.source_data.obs['assembly'].unique()[0], self.target_data.obs['assembly'].unique()[0])
        
        ## lift over whole intervals from source to target assembly
        # NOTE: pyliftover maps single coordinates. For BED-like half-open intervals [start, end),
        # we map start and (end-1), then set new_end = mapped_end + 1.

        # Fetch source ATAC modality
        if hasattr(self.source_data, "mod") and ("atac" in self.source_data.mod):
            source_atac = self.source_data.mod["atac"]
        else:
            try:
                source_atac = self.source_data["atac"]
            except Exception as e:
                raise ValueError("Source data must contain an 'atac' modality to liftOver.") from e

        required_cols = {"chrom", "chromStart", "chromEnd"}
        missing = required_cols - set(source_atac.var.columns)
        if missing:
            raise ValueError(f"Source ATAC .var is missing required columns: {sorted(missing)}")

        var = source_atac.var.copy()
        var["chrom"] = var["chrom"].astype(str)
        var["chromStart"] = pd.to_numeric(var["chromStart"], errors="raise").astype(int)
        var["chromEnd"] = pd.to_numeric(var["chromEnd"], errors="raise").astype(int)

        def _normalize_chr(chrom: str) -> str:
            c = str(chrom)
            if c in {"M", "MT"}:
                c = "chrM"
            if not c.startswith("chr"):
                c = f"chr{c}"
            if c == "chrMT":
                c = "chrM"
            return c

        def _lift_interval(chrom: str, start: int, end: int):
            if end <= start:
                return None
            c = _normalize_chr(chrom)

            start_hits = lo.convert_coordinate(c, int(start))
            end_hits = lo.convert_coordinate(c, int(end) - 1)
            if not start_hits or not end_hits:
                return None

            # Prefer hits on the same target chrom and strand
            best = None
            for (c1, p1, s1, _sc1) in start_hits:
                for (c2, p2, s2, _sc2) in end_hits:
                    if (c1 == c2) and (s1 == s2):
                        best = (c1, int(p1), int(p2), s1)
                        break
                if best is not None:
                    break
            if best is None:
                return None

            new_chrom, p_start, p_end_last, _strand = best
            new_start = min(p_start, p_end_last)
            new_end = max(p_start, p_end_last) + 1  # convert back to half-open
            return new_chrom, new_start, new_end

        def _liftOver_diagnostics(lifted_kept: pd.DataFrame, n_total: int, do_tss_diagnostics: bool = False):

            ## number of mapped peaks
            n_kept = int(lifted_kept.shape[0])
            print(f"[Dx] liftOver mapped {n_kept}/{n_total} source peaks ({(n_kept / max(n_total, 1)):.2%}). Dropping {n_total - n_kept}.")

            ## width correlation
            orig_width = lifted_kept.index.str.split(':').str[1].str.split('-').to_series().apply(
                lambda x: int(x[1]) - int(x[0])).values
            lifted_width = (lifted_kept["chromEnd"] - lifted_kept["chromStart"]).values
            corr, _ = pearsonr(orig_width, lifted_width)
            print(f"[Dx] Correlation between original and lifted width: {corr:.2f}")

            ## TSS distance
            if do_tss_diagnostics:
                print("[PROGRESS] Performing TSS diagnostics...")
                # IMPORTANT:
                # - Parse original peaks from the *index* (original coordinates) and normalize chr naming
                # - Use lifted coordinates from columns
                # - Ensure Start/End are ints (PyRanges expects integer coordinates)
                source_assembly = self.source_data.obs["assembly"].unique()[0]
                target_assembly = self.target_data.obs["assembly"].unique()[0]
                source_tss_pr = _get_reference_tss(source_assembly)
                target_tss_pr = _get_reference_tss(target_assembly)

                orig_df = lifted_kept.index.to_series().str.extract(
                    r"^(?P<Chromosome>[^:]+):(?P<Start>\d+)-(?P<End>\d+)$"
                )
                orig_df["Chromosome"] = orig_df["Chromosome"].map(_normalize_chr)
                orig_df[["Start", "End"]] = orig_df[["Start", "End"]].astype(int)
                # Use stable original peak name as peak_id (robust to reindexing/sorting)
                peak_id = lifted_kept.index.astype(str)
                orig_df["peak_id"] = peak_id.values

                lifted_df = lifted_kept.rename(
                    columns={"chrom": "Chromosome", "chromStart": "Start", "chromEnd": "End"}
                ).copy()
                lifted_df["Chromosome"] = lifted_df["Chromosome"].astype(str).map(_normalize_chr)
                lifted_df[["Start", "End"]] = lifted_df[["Start", "End"]].astype(int)
                lifted_df["peak_id"] = peak_id.values

                # Nearest TSS in each assembly; some peaks on contigs without TSS annotations can be dropped,
                # so we merge on peak_id to compare only peaks present in both results.
                src_near = PyRanges(orig_df).nearest(source_tss_pr).df
                tgt_near = PyRanges(lifted_df).nearest(target_tss_pr).df

                # Keep minimal columns for comparison (gene_name comes from the TSS annotation)
                src_near = src_near.loc[:, ["peak_id", "Distance", "gene_name"]].rename(
                    columns={"Distance": "Distance_source", "gene_name": "gene_name_source"}
                )
                tgt_near = tgt_near.loc[:, ["peak_id", "Distance", "gene_name"]].rename(
                    columns={"Distance": "Distance_target", "gene_name": "gene_name_target"}
                )

                merged = src_near.merge(tgt_near, on="peak_id", how="inner")
                print(f"[Dx] TSS diagnostic comparable peaks: {merged.shape[0]}/{lifted_kept.shape[0]} ({(merged.shape[0]/max(lifted_kept.shape[0],1)):.2%})")

                prop_same_gene = (merged["gene_name_source"].values == merged["gene_name_target"].values).mean()
                print(f"[Dx] Proportion of same nearest gene name: {prop_same_gene:.2%}")

                delta = np.abs(merged["Distance_source"].to_numpy() - merged["Distance_target"].to_numpy())
                med = np.median(delta)
                p95 = np.percentile(delta, 95)

                from scipy.stats import wasserstein_distance
                wd = wasserstein_distance(merged["Distance_source"].to_numpy(), merged["Distance_target"].to_numpy())

                print(f"[Dx] |Δ| median={med:.1f}bp, P95={p95:.1f}bp; Wasserstein={wd:.1f}bp")


        def _get_reference_tss(assembly: str):
            annot_path = f"{os.environ['SPATIALJEPA_HOME']}/data/reference_tss"
            os.makedirs(annot_path, exist_ok=True)
            parquet_path = os.path.join(annot_path, f"{assembly}_tss.parquet")

            # Check if parquet file already exists
            if os.path.exists(parquet_path):
                info = pd.read_parquet(parquet_path)
            else:
                # Download and process GTF only if parquet doesn't exist
                if assembly == "mm10":
                    gtf_path = os.path.join(annot_path, "mm10_gencode.gtf.gz")
                    if not os.path.exists(gtf_path):
                        annot_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.annotation.gtf.gz"
                        subprocess.run(["wget", annot_url, "-O", gtf_path], check=True)
                    info = read_gtf(gtf_path)

                elif assembly == "mm39":
                    gtf_path = os.path.join(annot_path, "mm39_gencode.gtf.gz")
                    if not os.path.exists(gtf_path):
                        annot_url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M38/gencode.vM38.annotation.gtf.gz"
                        subprocess.run(["wget", annot_url, "-O", gtf_path], check=True)
                    info = read_gtf(gtf_path)

                else:
                    raise ValueError(f"Assembly {assembly} not supported")

                # Process GTF to extract TSS
                info = info[info.df['Feature'] == 'gene'].df
                info["tss"] = info["Start"].where(info["Strand"] == "+", info["End"])
                
                # Save to parquet for future use
                info.to_parquet(parquet_path)

            # Convert to PyRanges format
            tss_pr = PyRanges(
                info[['gene_name', 'Chromosome', 'tss']].assign(
                    tssp1 = info['tss']+1
                ).rename(columns={"tss": "Start", "tssp1": "End"})
            )
            return tss_pr

        ## perform liftOver
        lifted = var.apply(
            lambda r: _lift_interval(r["chrom"], r["chromStart"], r["chromEnd"]),
            axis=1,
        )
        keep_mask = lifted.notna().to_numpy()

        # Subset ATAC to successfully lifted peaks
        kept_index = var.index[keep_mask]
        source_atac = source_atac[:, keep_mask].copy()
        lifted_kept = pd.DataFrame(
            lifted[keep_mask].tolist(),
            index=kept_index,
            columns=["chrom", "chromStart", "chromEnd"],
        )
        # Defensive: ensure we didn't lose alignment when subsetting
        assert source_atac.var.index.equals(kept_index), "liftOver: source_atac.var.index no longer matches kept_index"

        ## perform liftOver diagnostics
        _liftOver_diagnostics(lifted_kept, int(var.shape[0]), do_tss_diagnostics=True)

        # Keep originals for auditing
        source_atac.var["chrom_original"] = source_atac.var["chrom"].astype(str)
        source_atac.var["chromStart_original"] = pd.to_numeric(source_atac.var["chromStart"], errors="coerce")
        source_atac.var["chromEnd_original"] = pd.to_numeric(source_atac.var["chromEnd"], errors="coerce")

        source_atac.var["chrom"] = lifted_kept["chrom"].astype(str)
        source_atac.var["chromStart"] = lifted_kept["chromStart"].astype(int)
        source_atac.var["chromEnd"] = lifted_kept["chromEnd"].astype(int)

        # Update source assembly metadata now that intervals are in target assembly coordinates
        self.source_data.obs["assembly"] = self.target_data.obs["assembly"].unique()[0]

        # Write back updated ATAC modality
        if hasattr(self.source_data, "mod") and ("atac" in self.source_data.mod):
            self.source_data.mod["atac"] = source_atac
        else:
            self.source_data["atac"] = source_atac


# %%

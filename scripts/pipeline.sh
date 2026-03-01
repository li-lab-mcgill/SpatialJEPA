#!/bin/bash

echo "Aligning source and target features..."
conda activate nichecompass
python ${BAKLAVA_ROOT}/scripts/align_source_target_features.py \
--source-rna-h5ad /home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain.h5ad \
--source-atac-h5ad /home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain_atac.h5ad \
--target-rna-h5ad /home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain_atac.h5ad \
--target-atac-h5ad /home/mcb/users/dmannk/BAKLAVA_base/data/Spatial_ATAC_RNA/mouse/spatial_omics/spatial_atac_rna_seq_mouse_brain_atac.h5ad \
--source-assembly mm10 \
--target-assembly mm10 \
--source-name Spatial_ATAC_RNA \
--target-name 10x_mouse_brain \
--aligned-outdir /home/mcb/users/dmannk/BAKLAVA_base/data/aligned_data \
--mappers-out /home/mcb/users/dmannk/BAKLAVA_base/data/aligned_data/mappers.pkl \
--source-atac-peak-col peak_region_id \
--target-atac-peak-col peak_region_id
conda deactivate

echo "Training Tangram..."
conda activate squidpy_scvi_gpu
python ${BAKLAVA_base}/MultiGATE/scripts/train_tangram_mouse_brain.py
conda deactivate

echo "Training MultiGATE..."
conda activate MultiGATEenv_py310_scib
python ${BAKLAVA_base}/MultiGATE/scripts/mouse_brain_spatial_rna_atac.py
conda deactivate

echo "Done!"
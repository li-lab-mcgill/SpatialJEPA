"""
Utilities for customizing NicheCompass components.

This module is intended to host project-specific subclasses or helpers that
override behavior from the NicheCompass package (e.g., custom VGPGAE forward).
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import copy
import os
import math
import time
import warnings
from contextlib import nullcontext
from collections import defaultdict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from anndata import AnnData
import scanpy as sc
from torch_geometric.data import Data
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.utils import add_self_loops, remove_self_loops

from ot import solve as ot_solve
from optuna.distributions import CategoricalDistribution

from nichecompass.data import (SpatialAnnTorchDataset,
                               dataprocessors)
from nichecompass.data.utils import encode_labels, sparse_mx_to_sparse_tensor
from nichecompass.models import NicheCompass
from nichecompass.modules import VGPGAE
from nichecompass.nn import Encoder
from nichecompass.train import Trainer
from nichecompass.train.metrics import eval_metrics
from nichecompass.train.utils import _cycle_iterable, print_progress

from evals_utils import foscttm_moscot, benchmark_embeddings

# Isolate functions from dataprocessors to avoid circular imports
edge_level_split = dataprocessors.edge_level_split
node_level_split_mask = dataprocessors.node_level_split_mask

class CustomNicheCompass(NicheCompass):
    """
    Project-specific NicheCompass with custom behavior.
    """

    HPARAMS_SAMPLE_DURING_HPO = [
        "contrastive_logits_neg_ratio",
        "contrastive_logits_pos_ratio",
        "encoder_input_key",
        "multimodal_embedding_size",
        "multimodal_temperature",
        "node_batch_size",
    ]

    HPARAMS = {
        # __init__ parameters
        "adata": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "adata_atac": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "counts_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["counts", "pseudocounts"]
            ),
            "default": "counts",
        },
        "encoder_input_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["counts", "pseudocounts"]
            ),
            "default": "counts",
        },
        "adj_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["spatial_connectivities"]
            ),
            "default": "spatial_connectivities",
        },
        "gp_names_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_gp_names"]
            ),
            "default": "nichecompass_gp_names",
        },
        "active_gp_names_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_active_gp_names"]
            ),
            "default": "nichecompass_active_gp_names",
        },
        "gp_targets_mask_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_gp_targets"]
            ),
            "default": "nichecompass_gp_targets",
        },
        "gp_targets_categories_mask_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_gp_targets_categories"]
            ),
            "default": "nichecompass_gp_targets_categories",
        },
        "targets_categories_label_encoder_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_targets_categories_label_encoder"]
            ),
            "default": "nichecompass_targets_categories_label_encoder",
        },
        "gp_sources_mask_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_gp_sources"]
            ),
            "default": "nichecompass_gp_sources",
        },
        "gp_sources_categories_mask_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_gp_sources_categories"]
            ),
            "default": "nichecompass_gp_sources_categories",
        },
        "sources_categories_label_encoder_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_sources_categories_label_encoder"]
            ),
            "default": "nichecompass_sources_categories_label_encoder",
        },
        "ca_targets_mask_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_ca_targets", None]
            ),
            "default": "nichecompass_ca_targets",
        },
        "ca_sources_mask_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_ca_sources", None]
            ),
            "default": "nichecompass_ca_sources",
        },
        "latent_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_latent"]
            ),
            "default": "nichecompass_latent",
        },
        "cat_covariates_embeds_keys": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "cat_covariates_embeds_injection": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": ["gene_expr_decoder", "chrom_access_decoder"],
        },
        "cat_covariates_keys": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "cat_covariates_no_edges": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "genes_idx_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_genes_idx"]
            ),
            "default": "nichecompass_genes_idx",
        },
        "target_genes_idx_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_target_genes_idx"]
            ),
            "default": "nichecompass_target_genes_idx",
        },
        "source_genes_idx_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_source_genes_idx"]
            ),
            "default": "nichecompass_source_genes_idx",
        },
        "peaks_idx_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_peaks_idx"]
            ),
            "default": "nichecompass_peaks_idx",
        },
        "target_peaks_idx_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_target_peaks_idx"]
            ),
            "default": "nichecompass_target_peaks_idx",
        },
        "source_peaks_idx_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_source_peaks_idx"]
            ),
            "default": "nichecompass_source_peaks_idx",
        },
        "gene_peaks_mask_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_gene_peaks"]
            ),
            "default": "nichecompass_gene_peaks",
        },
        "recon_adj_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_recon_connectivities", None]
            ),
            "default": "nichecompass_recon_connectivities",
        },
        "agg_weights_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_agg_weights", None]
            ),
            "default": "nichecompass_agg_weights",
        },
        "include_edge_recon_loss": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "include_gene_expr_recon_loss": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "include_chrom_access_recon_loss": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "include_cat_covariates_contrastive_loss": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "gene_expr_recon_dist": {
            "suggest_distribution": CategoricalDistribution(choices=["nb"]),
            "default": "nb",
        },
        "log_variational": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "node_label_method": {
            "suggest_distribution": CategoricalDistribution(
                choices=["one-hop-sum", "one-hop-norm", "one-hop-attention"]
            ),
            "default": "one-hop-norm",
        },
        "active_gp_thresh_ratio": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.001, 0.01, 0.05]
            ),
            "default": 0.01,
        },
        "active_gp_type": {
            "suggest_distribution": CategoricalDistribution(
                choices=["mixed", "separate"]
            ),
            "default": "separate",
        },
        "n_fc_layers_encoder": {
            "suggest_distribution": CategoricalDistribution(choices=[1, 2]),
            "default": 1,
        },
        "n_layers_encoder": {
            "suggest_distribution": CategoricalDistribution(choices=[1, 2]),
            "default": 1,
        },
        "n_hidden_encoder": {
            "suggest_distribution": CategoricalDistribution(
                choices=[None, 128, 256, 512]
            ),
            "default": None,
        },
        "conv_layer_encoder": {
            "suggest_distribution": CategoricalDistribution(
                choices=["gcnconv", "gatv2conv"]
            ),
            "default": "gcnconv",
        },
        "encoder_n_attention_heads": {
            "suggest_distribution": CategoricalDistribution(choices=[1, 2, 4, 8]),
            "default": 4,
        },
        "encoder_use_bn": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "dropout_rate_encoder": {
            "suggest_distribution": CategoricalDistribution(choices=[0.0, 0.1, 0.2]),
            "default": 0.0,
        },
        "dropout_rate_graph_decoder": {
            "suggest_distribution": CategoricalDistribution(choices=[0.0, 0.1, 0.2]),
            "default": 0.0,
        },
        "cat_covariates_cats": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "n_addon_gp": {
            "suggest_distribution": CategoricalDistribution(choices=[0, 50, 100]),
            "default": 100,
        },
        "multimodal_embedding_size": {
            "suggest_distribution": CategoricalDistribution(choices=[32, 128]),
            "default": 32,
        },
        "multimodal_layer_series": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "cat_covariates_embeds_nums": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "include_edge_kl_loss": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "use_cuda_if_available": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "seed": {
            "suggest_distribution": CategoricalDistribution(choices=[0, 1, 42]),
            "default": 0,
        },
        # train() parameters
        "n_epochs": {
            "suggest_distribution": CategoricalDistribution(choices=[50, 100, 200]),
            "default": 100,
        },
        "n_epochs_all_gps": {
            "suggest_distribution": CategoricalDistribution(choices=[10, 25, 50]),
            "default": 25,
        },
        "n_epochs_no_edge_recon": {
            "suggest_distribution": CategoricalDistribution(choices=[0, 5, 10]),
            "default": 0,
        },
        "n_epochs_no_cat_covariates_contrastive": {
            "suggest_distribution": CategoricalDistribution(choices=[0, 5, 10]),
            "default": 5,
        },
        "lr": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0001, 0.0005, 0.001]
            ),
            "default": 0.0001,
        },
        "weight_decay": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0, 0.00001, 0.0001]
            ),
            "default": 0.0,
        },
        "lambda_edge_recon": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0, 500000.0]
            ),
            "default": 500000.0,
        },
        "lambda_gene_expr_recon": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0, 100.0, 300.0]
            ),
            "default": 300.0,
        },
        "lambda_chrom_access_recon": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0, 10.0, 100.0, 300.0]
            ),
            "default": 300.0,
        },
        "lambda_cat_covariates_contrastive": {
            "suggest_distribution": CategoricalDistribution(choices=[0.0, 0.1, 1.0]),
            "default": 0.0,
        },
        "lambda_multimodal_contrastive_loss": {
            "suggest_distribution": CategoricalDistribution(
                choices=[1000.0, 10000.0, 100000.0]
            ),
            "default": 10000.0,
        },
        "lambda_knowledge_distillation": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0, 100.0, 1000.0]
            ),
            "default": 1000.0,
        },
        "multimodal_temperature": {
            "suggest_distribution": CategoricalDistribution(choices=[1.0, 2.5, 10.0]),
            "default": 2.5,
        },
        "multimodal_contrastive_anneal": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "contrastive_logits_pos_ratio": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0]
            ),
            "default": 0.0,
        },
        "contrastive_logits_neg_ratio": {
            "suggest_distribution": CategoricalDistribution(
                choices=[0.0]
            ),
            "default": 0.0,
        },
        "lambda_group_lasso": {
            "suggest_distribution": CategoricalDistribution(choices=[0.0, 1.0, 10.0]),
            "default": 0.0,
        },
        "lambda_l1_masked": {
            "suggest_distribution": CategoricalDistribution(choices=[0.0, 1.0, 10.0]),
            "default": 0.0,
        },
        "l1_targets_categories": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": ["target_gene"],
        },
        "l1_sources_categories": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "lambda_l1_addon": {
            "suggest_distribution": CategoricalDistribution(choices=[0.0, 10.0, 30.0]),
            "default": 30.0,
        },
        "edge_val_ratio": {
            "suggest_distribution": CategoricalDistribution(choices=[0.05, 0.1, 0.2]),
            "default": 0.1,
        },
        "node_val_ratio": {
            "suggest_distribution": CategoricalDistribution(choices=[0.05, 0.1, 0.2]),
            "default": 0.1,
        },
        "edge_batch_size": {
            "suggest_distribution": CategoricalDistribution(choices=[256, 512, 1000]),
            "default": 1000,
        },
        "node_batch_size": {
            "suggest_distribution": CategoricalDistribution(choices=[1000]),
            "default": 1000,
        },
        "paired_data": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "target_adata": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "target_adata_atac": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "target_holdout_frac": {
            "suggest_distribution": CategoricalDistribution(choices=[0.05, 0.1, 0.2]),
            "default": 0.1,
        },
        "target_holdout_n": {
            "suggest_distribution": CategoricalDistribution(choices=[None, 500, 2000]),
            "default": None,
        },
        "target_holdout_seed": {
            "suggest_distribution": CategoricalDistribution(choices=[0, 1, 42]),
            "default": 0,
        },
        "target_paired_data": {
            "suggest_distribution": CategoricalDistribution(choices=[True, False]),
            "default": True,
        },
        "target_encoder_input_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=[None, "counts", "pseudocounts"]
            ),
            "default": None,
        },
        "target_counts_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=[None, "counts", "pseudocounts"]
            ),
            "default": None,
        },
        "log_target_multimodal_contrastive": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": True,
        },
        "mlflow_experiment_id": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "mlflow_parent_run_id": {
            "suggest_distribution": CategoricalDistribution(choices=[None]),
            "default": None,
        },
        "retrieve_cat_covariates_embeds": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "retrieve_recon_edge_probs": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "retrieve_agg_weights": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "n_sampled_neighbors": {
            "suggest_distribution": CategoricalDistribution(choices=[-1, 4, 8, 16]),
            "default": 4,
        },
        "latent_dtype": {
            "suggest_distribution": CategoricalDistribution(
                choices=["float64", "float32"]
            ),
            "default": np.float64,
        },
        # trainer_kwargs commonly used with train()
        "target_latent_key": {
            "suggest_distribution": CategoricalDistribution(
                choices=["nichecompass_latent"]
            ),
            "default": "nichecompass_latent",
        },
        "use_early_stopping": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
        "verbose": {
            "suggest_distribution": CategoricalDistribution(choices=[False, True]),
            "default": False,
        },
    }

    @classmethod
    def get_hparams(cls, key=None):
        if key is not None:
            return copy.deepcopy(cls.HPARAMS[key])
        return copy.deepcopy(cls.HPARAMS)

    @classmethod
    def get_hparams_sample_during_hpo(cls):
        keys = copy.deepcopy(cls.HPARAMS_SAMPLE_DURING_HPO)
        unknown_keys = sorted(set(keys) - set(cls.HPARAMS))
        if unknown_keys:
            raise KeyError(
                f"Unknown HPARAMS keys in HPARAMS_SAMPLE_DURING_HPO: {unknown_keys}"
            )
        return keys

    @classmethod
    def _get_hparam_defaults(cls):
        """Extract default values from HPARAMS dictionary."""
        return {key: config['default'] for key, config in cls.HPARAMS.items() 
                if 'default' in config}
    
    @classmethod
    def _apply_hparam_defaults(cls, **kwargs):
        """
        Apply HPARAMS defaults to kwargs.
        Only applies defaults for keys that exist in HPARAMS and have None values.
        """
        hparam_defaults = cls._get_hparam_defaults()
        result = {}
        for key, value in kwargs.items():
            if key in hparam_defaults and value is None:
                result[key] = hparam_defaults[key]
            else:
                result[key] = value
        return result

    @staticmethod
    def _coerce_numpy_dtype(dtype: type) -> type:
        if isinstance(dtype, str):
            if dtype == "float32":
                return np.float32
            if dtype == "float64":
                return np.float64
            return np.dtype(dtype).type
        return dtype

    def __init__(self,
                 adata: AnnData,
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]=None,
                 encoder_input_key: Optional[str]=None,
                 adj_key: str=None,
                 gp_names_key: str=None,
                 active_gp_names_key: str=None,
                 gp_targets_mask_key: str=None,
                 gp_targets_categories_mask_key: str=None,
                 targets_categories_label_encoder_key: str=None,
                 gp_sources_mask_key: str=None,
                 gp_sources_categories_mask_key: str=None,
                 sources_categories_label_encoder_key: str=None,
                 ca_targets_mask_key: Optional[str]=None,
                 ca_sources_mask_key: Optional[str]=None,
                 latent_key: str=None,
                 cat_covariates_embeds_keys: Optional[List[str]]=None,
                 cat_covariates_embeds_injection: Optional[List[
                     Literal["encoder",
                             "gene_expr_decoder",
                             "chrom_access_decoder"]]]=None,
                 cat_covariates_keys: Optional[List[str]]=None,
                 cat_covariates_no_edges: Optional[List[bool]]=None,
                 genes_idx_key: str=None,
                 target_genes_idx_key: str=None,
                 source_genes_idx_key: str=None,
                 peaks_idx_key: str=None,
                 target_peaks_idx_key: str=None,
                 source_peaks_idx_key: str=None,
                 gene_peaks_mask_key: str=None,
                 recon_adj_key: Optional[str]=None,
                 agg_weights_key: Optional[str]=None,
                 include_edge_recon_loss: bool=None,
                 include_gene_expr_recon_loss: bool=None,
                 include_chrom_access_recon_loss: Optional[bool]=None,
                 include_cat_covariates_contrastive_loss: bool=None,
                 gene_expr_recon_dist: Literal["nb"]=None,
                 log_variational: bool=None,
                 node_label_method: Literal[
                    "one-hop-sum",
                    "one-hop-norm",
                    "one-hop-attention"]=None,
                 active_gp_thresh_ratio: float=None,
                 active_gp_type: Literal["mixed", "separate"]=None,
                 n_fc_layers_encoder: int=None,
                 n_layers_encoder: int=None,
                 n_hidden_encoder: Optional[int]=None,
                 conv_layer_encoder: Literal["gcnconv", "gatv2conv"]=None,
                 encoder_n_attention_heads: Optional[int]=None,
                 encoder_use_bn: bool=None,
                 dropout_rate_encoder: float=None,
                 dropout_rate_graph_decoder: float=None,
                 cat_covariates_cats: Optional[List[List]]=None,
                 n_addon_gp: int=None,
                 multimodal_embedding_size: Optional[int]=None,
                 multimodal_layer_series: Optional[bool]=None,
                 cat_covariates_embeds_nums: Optional[List[int]]=None,
                 include_edge_kl_loss: bool=None,
                 use_cuda_if_available: bool=None,
                 seed: int=None,
                 **kwargs):
        # Apply HPARAMS defaults centrally for constructor parameters.
        hparam_kwargs = self.__class__._apply_hparam_defaults(
            adata_atac=adata_atac,
            counts_key=counts_key,
            encoder_input_key=encoder_input_key,
            adj_key=adj_key,
            gp_names_key=gp_names_key,
            active_gp_names_key=active_gp_names_key,
            gp_targets_mask_key=gp_targets_mask_key,
            gp_targets_categories_mask_key=gp_targets_categories_mask_key,
            targets_categories_label_encoder_key=targets_categories_label_encoder_key,
            gp_sources_mask_key=gp_sources_mask_key,
            gp_sources_categories_mask_key=gp_sources_categories_mask_key,
            sources_categories_label_encoder_key=sources_categories_label_encoder_key,
            ca_targets_mask_key=ca_targets_mask_key,
            ca_sources_mask_key=ca_sources_mask_key,
            latent_key=latent_key,
            cat_covariates_embeds_keys=cat_covariates_embeds_keys,
            cat_covariates_embeds_injection=cat_covariates_embeds_injection,
            cat_covariates_keys=cat_covariates_keys,
            cat_covariates_no_edges=cat_covariates_no_edges,
            genes_idx_key=genes_idx_key,
            target_genes_idx_key=target_genes_idx_key,
            source_genes_idx_key=source_genes_idx_key,
            peaks_idx_key=peaks_idx_key,
            target_peaks_idx_key=target_peaks_idx_key,
            source_peaks_idx_key=source_peaks_idx_key,
            gene_peaks_mask_key=gene_peaks_mask_key,
            recon_adj_key=recon_adj_key,
            agg_weights_key=agg_weights_key,
            include_edge_recon_loss=include_edge_recon_loss,
            include_gene_expr_recon_loss=include_gene_expr_recon_loss,
            include_chrom_access_recon_loss=include_chrom_access_recon_loss,
            include_cat_covariates_contrastive_loss=include_cat_covariates_contrastive_loss,
            gene_expr_recon_dist=gene_expr_recon_dist,
            log_variational=log_variational,
            node_label_method=node_label_method,
            active_gp_thresh_ratio=active_gp_thresh_ratio,
            active_gp_type=active_gp_type,
            n_fc_layers_encoder=n_fc_layers_encoder,
            n_layers_encoder=n_layers_encoder,
            n_hidden_encoder=n_hidden_encoder,
            conv_layer_encoder=conv_layer_encoder,
            encoder_n_attention_heads=encoder_n_attention_heads,
            encoder_use_bn=encoder_use_bn,
            dropout_rate_encoder=dropout_rate_encoder,
            dropout_rate_graph_decoder=dropout_rate_graph_decoder,
            cat_covariates_cats=cat_covariates_cats,
            n_addon_gp=n_addon_gp,
            multimodal_embedding_size=multimodal_embedding_size,
            multimodal_layer_series=multimodal_layer_series,
            cat_covariates_embeds_nums=cat_covariates_embeds_nums,
            include_edge_kl_loss=include_edge_kl_loss,
            use_cuda_if_available=use_cuda_if_available,
            seed=seed,
        )
        adata_atac = hparam_kwargs["adata_atac"]
        counts_key = hparam_kwargs["counts_key"]
        encoder_input_key = hparam_kwargs["encoder_input_key"]
        adj_key = hparam_kwargs["adj_key"]
        gp_names_key = hparam_kwargs["gp_names_key"]
        active_gp_names_key = hparam_kwargs["active_gp_names_key"]
        gp_targets_mask_key = hparam_kwargs["gp_targets_mask_key"]
        gp_targets_categories_mask_key = hparam_kwargs["gp_targets_categories_mask_key"]
        targets_categories_label_encoder_key = hparam_kwargs[
            "targets_categories_label_encoder_key"
        ]
        gp_sources_mask_key = hparam_kwargs["gp_sources_mask_key"]
        gp_sources_categories_mask_key = hparam_kwargs["gp_sources_categories_mask_key"]
        sources_categories_label_encoder_key = hparam_kwargs[
            "sources_categories_label_encoder_key"
        ]
        ca_targets_mask_key = hparam_kwargs["ca_targets_mask_key"]
        ca_sources_mask_key = hparam_kwargs["ca_sources_mask_key"]
        latent_key = hparam_kwargs["latent_key"]
        cat_covariates_embeds_keys = hparam_kwargs["cat_covariates_embeds_keys"]
        cat_covariates_embeds_injection = hparam_kwargs[
            "cat_covariates_embeds_injection"
        ]
        cat_covariates_keys = hparam_kwargs["cat_covariates_keys"]
        cat_covariates_no_edges = hparam_kwargs["cat_covariates_no_edges"]
        genes_idx_key = hparam_kwargs["genes_idx_key"]
        target_genes_idx_key = hparam_kwargs["target_genes_idx_key"]
        source_genes_idx_key = hparam_kwargs["source_genes_idx_key"]
        peaks_idx_key = hparam_kwargs["peaks_idx_key"]
        target_peaks_idx_key = hparam_kwargs["target_peaks_idx_key"]
        source_peaks_idx_key = hparam_kwargs["source_peaks_idx_key"]
        gene_peaks_mask_key = hparam_kwargs["gene_peaks_mask_key"]
        recon_adj_key = hparam_kwargs["recon_adj_key"]
        agg_weights_key = hparam_kwargs["agg_weights_key"]
        include_edge_recon_loss = hparam_kwargs["include_edge_recon_loss"]
        include_gene_expr_recon_loss = hparam_kwargs["include_gene_expr_recon_loss"]
        include_chrom_access_recon_loss = hparam_kwargs[
            "include_chrom_access_recon_loss"
        ]
        include_cat_covariates_contrastive_loss = hparam_kwargs[
            "include_cat_covariates_contrastive_loss"
        ]
        gene_expr_recon_dist = hparam_kwargs["gene_expr_recon_dist"]
        log_variational = hparam_kwargs["log_variational"]
        node_label_method = hparam_kwargs["node_label_method"]
        active_gp_thresh_ratio = hparam_kwargs["active_gp_thresh_ratio"]
        active_gp_type = hparam_kwargs["active_gp_type"]
        n_fc_layers_encoder = hparam_kwargs["n_fc_layers_encoder"]
        n_layers_encoder = hparam_kwargs["n_layers_encoder"]
        n_hidden_encoder = hparam_kwargs["n_hidden_encoder"]
        conv_layer_encoder = hparam_kwargs["conv_layer_encoder"]
        encoder_n_attention_heads = hparam_kwargs["encoder_n_attention_heads"]
        encoder_use_bn = hparam_kwargs["encoder_use_bn"]
        dropout_rate_encoder = hparam_kwargs["dropout_rate_encoder"]
        dropout_rate_graph_decoder = hparam_kwargs["dropout_rate_graph_decoder"]
        cat_covariates_cats = hparam_kwargs["cat_covariates_cats"]
        n_addon_gp = hparam_kwargs["n_addon_gp"]
        multimodal_embedding_size = hparam_kwargs["multimodal_embedding_size"]
        multimodal_layer_series = hparam_kwargs["multimodal_layer_series"]
        cat_covariates_embeds_nums = hparam_kwargs["cat_covariates_embeds_nums"]
        include_edge_kl_loss = hparam_kwargs["include_edge_kl_loss"]
        use_cuda_if_available = hparam_kwargs["use_cuda_if_available"]
        seed = hparam_kwargs["seed"]
        self.adata = adata
        self.adata_atac = adata_atac
        self.counts_key_ = counts_key
        self.encoder_input_key_ = encoder_input_key
        self.adj_key_ = adj_key
        self.gp_names_key_ = gp_names_key
        self.active_gp_names_key_ = active_gp_names_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_targets_categories_mask_key_ = gp_targets_categories_mask_key
        self.targets_categories_label_encoder_key_ = (
            targets_categories_label_encoder_key)
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.gp_sources_categories_mask_key_ = gp_sources_categories_mask_key
        self.sources_categories_label_encoder_key_ = (
            sources_categories_label_encoder_key)
        self.ca_targets_mask_key_ = ca_targets_mask_key
        self.ca_sources_mask_key_ = ca_sources_mask_key
        self.latent_key_ = latent_key
        self.cat_covariates_embeds_keys_ = cat_covariates_embeds_keys
        self.cat_covariates_embeds_injection_ = cat_covariates_embeds_injection
        self.cat_covariates_keys_ = cat_covariates_keys
        self.cat_covariates_embeds_keys_ = cat_covariates_embeds_keys
        self.genes_idx_key_ = genes_idx_key
        self.target_genes_idx_key_ = target_genes_idx_key
        self.source_genes_idx_key_ = source_genes_idx_key
        self.peaks_idx_key_ = peaks_idx_key
        self.target_peaks_idx_key_ = target_peaks_idx_key
        self.source_peaks_idx_key_ = source_peaks_idx_key
        self.gene_peaks_mask_key_ = gene_peaks_mask_key
        self.recon_adj_key_ = recon_adj_key
        self.agg_weights_key_ = agg_weights_key
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.include_chrom_access_recon_loss_ = include_chrom_access_recon_loss
        self.include_cat_covariates_contrastive_loss_ = (
            include_cat_covariates_contrastive_loss)
        self.gene_expr_recon_dist_ = gene_expr_recon_dist
        self.log_variational_ = log_variational
        self.node_label_method_ = node_label_method
        self.active_gp_thresh_ratio_ = active_gp_thresh_ratio
        self.active_gp_type_ = active_gp_type
        self.include_edge_kl_loss_ = include_edge_kl_loss
        self.seed_ = seed

        # Set seed for reproducibility
        np.random.seed(self.seed_)
        if use_cuda_if_available & torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_)
            torch.manual_seed(self.seed_)
        else:
            torch.manual_seed(self.seed_)

        # Retrieve gene program masks
        if gp_targets_mask_key in adata.varm:
            # NOTE: dtype can be changed to bool and should be able to handle sparse
            # mask
            self.gp_targets_mask_ = torch.tensor(
                adata.varm[gp_targets_mask_key].T,
                dtype=torch.bool)
        else:
            raise ValueError("Please specify an adequate ´gp_targets_mask_key´ "
                             "for your adata object. The targets mask needs to "
                             "be stored in ´adata.varm[gp_targets_mask_key]´. "
                             " If you do not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s that"
                             " allows all gene program latent nodes to "
                             "reconstruct all genes.")

        if gp_sources_mask_key in adata.varm:
            # NOTE: dtype can be changed to bool and should be able to handle
            # sparse mask
            self.gp_sources_mask_ = torch.tensor(
                adata.varm[gp_sources_mask_key].T,
                dtype=torch.bool)
                                           
        else:
            raise ValueError("Please specify an adequate "
                             "´gp_sources_mask_key´ for your adata object. "
                             "The sources mask needs to be stored in "
                             "´adata.varm[gp_sources_mask_key]´. If you do "
                             "not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s "
                             " that allows all gene program latent nodes to"
                             " reconstruct all genes.")
            
        # Determine features scale factors
        self.features_scale_factors_ = torch.concat(
            (torch.tensor(self.adata.X.sum(0))[0],
             torch.tensor(self.adata.X.sum(0))[0]))
    
        # Retrieve chromatin accessibility masks
        if adata_atac is None:
            self.ca_targets_mask_ = None
            self.ca_sources_mask_ = None
            gene_peaks_mask = None
        else:
            gene_peaks_mask = adata.varm[gene_peaks_mask_key].tocoo()
            gene_peaks_mask = torch.sparse_coo_tensor(
                indices=[gene_peaks_mask.row, gene_peaks_mask.col],
                values=gene_peaks_mask.data,
                size=gene_peaks_mask.shape,
                dtype=torch.bool) # bool does not work with torch.mm
            if ca_targets_mask_key in adata_atac.varm:
                ca_targets_mask = adata_atac.varm[ca_targets_mask_key].T.tocoo()
            else:
                raise ValueError("Please specify an adequate "
                                 "´ca_targets_mask_key´ for your adata_atac "
                                 "object. The targets mask needs to be stored "
                                 "in ´adata_atac.varm[ca_targets_mask_key]´. If"
                                 " you do not want to mask chromatin "
                                 " accessibility reconstruction, you can create"
                                 " a mask of 1s that allows all gene program "
                                 "latent nodes to reconstruct all peaks.")
            self.ca_targets_mask_ = torch.sparse_coo_tensor(
                indices=[ca_targets_mask.row, ca_targets_mask.col],
                values=ca_targets_mask.data,
                size=ca_targets_mask.shape,
                dtype=torch.bool).to_dense() # for now
            if ca_sources_mask_key in adata_atac.varm:
                ca_sources_mask = adata_atac.varm[
                    ca_sources_mask_key].T.tocoo()
                self.ca_sources_mask_ = torch.sparse_coo_tensor(
                    indices=[ca_sources_mask.row, ca_sources_mask.col],
                    values=ca_sources_mask.data,
                    size=ca_sources_mask.shape,
                    dtype=torch.bool).to_dense() # for now
            else:
                raise ValueError("Please specify an adequate "
                                "´ca_sources_mask_key´ for your adata_atac "
                                "object. The sources mask needs to be "
                                "stored in "
                                "´adata_atac.varm[ca_sources_mask_key]´. If"
                                "you do not want to mask chromatin "
                                " accessibility reconstruction, you can "
                                "create a mask of 1s that allows all gene "
                                "program latent nodes to reconstruct all "
                                "peaks.")

        # Retrieve index of genes in gp mask and index of genes not in gp mask
        self.features_idx_dict_ = {}
        self.features_idx_dict_["masked_rna_idx"] = adata.uns[
            genes_idx_key]
        self.features_idx_dict_["unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["masked_rna_idx"]]
        self.features_idx_dict_["target_masked_rna_idx"] = list(
            adata.uns[target_genes_idx_key])
        self.features_idx_dict_["target_unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["target_masked_rna_idx"]]
        self.features_idx_dict_["source_masked_rna_idx"] = list(
            adata.uns[source_genes_idx_key])
        self.features_idx_dict_["source_unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["source_masked_rna_idx"]]
        
        # Retrieve index of peaks in ca mask and index of peaks not in ca mask
        if adata_atac is not None:
            self.peaks_idx_ = adata_atac.uns[peaks_idx_key]
            self.target_peaks_idx_ = adata_atac.uns[target_peaks_idx_key]
            self.source_peaks_idx_ = adata_atac.uns[source_peaks_idx_key]
            
            self.features_idx_dict_["masked_atac_idx"] = adata_atac.uns[
                peaks_idx_key]
            self.features_idx_dict_["unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["masked_atac_idx"]]
            self.features_idx_dict_["target_masked_atac_idx"] = list(
                adata_atac.uns[target_peaks_idx_key])
            self.features_idx_dict_["target_unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["target_masked_atac_idx"]]
            self.features_idx_dict_["source_masked_atac_idx"] = list(
                adata_atac.uns[source_peaks_idx_key])
            self.features_idx_dict_["source_unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["source_masked_atac_idx"]]

        # Determine VGPGAE inputs
        self.n_input_ = adata.n_vars
        self.n_output_genes_ = adata.n_vars
        if adata_atac is not None:
            self.modalities_ = ["rna", "atac"]
            #if not np.all(adata.obs.index == adata_atac.obs.index):
            #    raise ValueError("Please make sure that 'adata' and "
            #                     "'adata_atac' contain the same observations in"
            #                     " the same order.")
            # Peaks are concatenated to genes in input
            self.n_input_ += adata_atac.n_vars
            self.n_output_peaks_ = adata_atac.n_vars
        else:
            self.modalities_ = ["rna"]
            self.n_output_peaks_ = 0
        self.n_fc_layers_encoder_ = n_fc_layers_encoder
        self.n_layers_encoder_ = n_layers_encoder
        self.conv_layer_encoder_ = conv_layer_encoder
        if conv_layer_encoder == "gatv2conv":
            self.encoder_n_attention_heads_ = encoder_n_attention_heads
        else:
            self.encoder_n_attention_heads_ = 0
        self.encoder_use_bn_ = encoder_use_bn
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder
        self.n_prior_gp_ = len(self.gp_targets_mask_)
        self.n_addon_gp_ = n_addon_gp
        self.multimodal_embedding_size_ = multimodal_embedding_size
        
        if n_addon_gp > 0:
            # Add add-on gps to adata
            gp_list = list(self.adata.uns[self.gp_names_key_])
            for i in range(n_addon_gp):
                if f"Add-on_{i}_GP" not in gp_list:
                    gp_list.append(f"Add-on_{i}_GP")
            self.adata.uns[self.gp_names_key_] = np.array(gp_list)
        else:
            # Remove add-on gps from adata
            for gp_name in list(adata.uns[gp_names_key]):
                if "Add-on" in gp_name:
                    self.adata.uns[gp_names_key] = np.delete(
                        self.adata.uns[gp_names_key],
                        list(self.adata.uns[gp_names_key]).index(gp_name))

        # Retrieve categorical covariates categories
        if cat_covariates_cats is None:
            if cat_covariates_keys is not None:
                self.cat_covariates_cats_ = [
                    adata.obs[cat_covariate_key].unique().tolist() 
                    for cat_covariate_key in cat_covariates_keys]
            else:
                self.cat_covariates_cats_ = []
        else:
            self.cat_covariates_cats_ = cat_covariates_cats
        
        # Define dimensionality of categorical covariates embeddings as
        # number of categories of each categorical covariate respectively
        # if not provided explicitly
        if cat_covariates_embeds_nums is None:
            cat_covariates_embeds_nums = []
            for cat_covariate_cats in self.cat_covariates_cats_:
                cat_covariates_embeds_nums.append(len(cat_covariate_cats))
        self.cat_covariates_embeds_nums_ = cat_covariates_embeds_nums

        # Determine dimensionality of hidden encoder layer if not provided
        if n_hidden_encoder is None:
            if len(adata.var) > (self.n_prior_gp_ + self.n_addon_gp_):
                n_hidden_encoder = (self.n_prior_gp_ + self.n_addon_gp_)
            else:
                n_hidden_encoder = len(adata.var)
        self.n_hidden_encoder_ = n_hidden_encoder
        self.multimodal_layer_series_ = multimodal_layer_series
            
        # Define categorical covariates no edges as all 'True' if not
        # explicitly provided, so that they are excluded from the edge
        # reconstruction loss
        if ((cat_covariates_no_edges is None) &
            (len(self.cat_covariates_cats_) > 0)):
            self.cat_covariates_no_edges_ = (
                [True] * len(self.cat_covariates_cats_))
        else:
            self.cat_covariates_no_edges_ = cat_covariates_no_edges
        
        # Validate counts layer key and counts values
        if counts_key is not None and counts_key not in adata.layers:
            raise ValueError("Please specify an adequate ´counts_key´. By "
                             "default the counts are assumed to be stored in "
                             "data.layers['counts'].")
        if include_gene_expr_recon_loss and log_variational:
            if counts_key is None:
                x = adata.X
            else:
                x = adata.layers[counts_key]
            if (x < 0).sum() > 0:
                raise ValueError("Please make sure that "
                                 "´adata.layers[counts_key]´ contains the"
                                 " raw counts (not log library size "
                                 "normalized) if ´include_gene_expr_recon_loss´"
                                 " is ´True´ and ´log_variational´ is ´True´. "
                                 "If you want to use log library size "
                                 " normalized counts, make sure that "
                                 "´log_variational´ is ´False´.")

        # Validate encoder input key (if provided)
        if encoder_input_key is not None:
            if encoder_input_key not in adata.layers:
                raise ValueError(
                    "Please specify an adequate ´encoder_input_key´. "
                    "The key was not found in adata.layers.")
            if adata_atac is not None and encoder_input_key not in adata_atac.layers:
                raise ValueError(
                    "Please specify an adequate ´encoder_input_key´. "
                    "The key was not found in adata_atac.layers.")

        # Validate adjacency key
        if adj_key not in adata.obsp:
            raise ValueError("Please specify an adequate ´adj_key´. "
                             "By default the adjacency matrix is assumed to be "
                             "stored in adata.obsm['spatial_connectivities'].")

        # Validate gp key
        if gp_names_key not in adata.uns:
            raise ValueError("Please specify an adequate ´gp_names_key´. "
                             "By default the gene program names are assumed to "
                             "be stored in adata.uns['nichecompass_gp_names'].")

        # Validate categorical covariates keys
        if cat_covariates_keys is not None:
            for cat_covariate_key in cat_covariates_keys:
                if cat_covariate_key not in adata.obs:
                    raise ValueError(
                        "Please specify adequate ´cat_covariates_keys´. "
                        f"The key {cat_covariate_key} was not found in adata.")
        
        # Initialize model with Variational Gene Program Graph Autoencoder 
        # neural network module
        self.model = CustomVGPGAE(
            n_input=self.n_input_,
            n_fc_layers_encoder=self.n_fc_layers_encoder_,
            n_layers_encoder=self.n_layers_encoder_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_prior_gp=self.n_prior_gp_,
            n_addon_gp=self.n_addon_gp_,
            multimodal_embedding_size=self.multimodal_embedding_size_,
            multimodal_layer_series=self.multimodal_layer_series_,
            cat_covariates_embeds_nums=self.cat_covariates_embeds_nums_,
            n_output_genes=self.n_output_genes_,
            n_output_peaks=self.n_output_peaks_,
            target_rna_decoder_mask=self.gp_targets_mask_,
            source_rna_decoder_mask=self.gp_sources_mask_,
            target_atac_decoder_mask=self.ca_targets_mask_,
            source_atac_decoder_mask=self.ca_sources_mask_,
            features_idx_dict=self.features_idx_dict_,
            features_scale_factors=self.features_scale_factors_,
            gene_peaks_mask=gene_peaks_mask,
            cat_covariates_cats=self.cat_covariates_cats_,
            cat_covariates_no_edges=self.cat_covariates_no_edges_,
            conv_layer_encoder=self.conv_layer_encoder_,
            encoder_n_attention_heads=self.encoder_n_attention_heads_,
            encoder_use_bn=self.encoder_use_bn_,
            dropout_rate_encoder=self.dropout_rate_encoder_,
            dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_,
            include_edge_recon_loss=self.include_edge_recon_loss_,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss_,
            include_chrom_access_recon_loss=self.include_chrom_access_recon_loss_,
            include_cat_covariates_contrastive_loss=self.include_cat_covariates_contrastive_loss_,
            rna_recon_loss=self.gene_expr_recon_dist_,
            node_label_method=self.node_label_method_,
            active_gp_thresh_ratio=self.active_gp_thresh_ratio_,
            active_gp_type=self.active_gp_type_,
            log_variational=self.log_variational_,
            cat_covariates_embeds_injection=self.cat_covariates_embeds_injection_,
            include_edge_kl_loss=self.include_edge_kl_loss_)

        # Stage-2 distillation models are created after stage-1 training.
        self.teacher_model = None
        self.student_model = None
        self.stage2_mlflow_run_id_ = None

        self.is_trained_ = False

        # Store init params for saving and loading
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def load(cls,
             dir_path: str,
             adata: Optional[AnnData]=None,
             adata_atac: Optional[AnnData]=None,
             adata_file_name: str="adata.h5ad",
             adata_atac_file_name: Optional[str]="adata_atac.h5ad",
             use_cuda: bool=False,
             n_addon_gps: int=0,
             gp_names_key: Optional[str]=None,
             genes_idx_key: Optional[str]=None,
             unfreeze_all_weights: bool=False,
             unfreeze_addon_gp_weights: bool=False,
             unfreeze_cat_covariates_embedder_weights: bool=False
             ) -> torch.nn.Module:
        """
        Load a saved CustomNicheCompass model with ATAC defaults.
        """
        return super().load(
            dir_path=dir_path,
            adata=adata,
            adata_atac=adata_atac,
            adata_file_name=adata_file_name,
            adata_atac_file_name=adata_atac_file_name,
            use_cuda=use_cuda,
            n_addon_gps=n_addon_gps,
            gp_names_key=gp_names_key,
            genes_idx_key=genes_idx_key,
            unfreeze_all_weights=unfreeze_all_weights,
            unfreeze_addon_gp_weights=unfreeze_addon_gp_weights,
            unfreeze_cat_covariates_embedder_weights=(
                unfreeze_cat_covariates_embedder_weights))

    def train(self,
                n_epochs: int=None,
                n_epochs_all_gps: int=None,
                n_epochs_no_edge_recon: int=None,
                n_epochs_no_cat_covariates_contrastive: int=None,
                lr: float=None,
                weight_decay: float=None,
                lambda_edge_recon: Optional[float]=None, #500000.,
                lambda_gene_expr_recon: float=None, #300.,
                lambda_chrom_access_recon: float=None, #100.,
                lambda_cat_covariates_contrastive: float=None,
                lambda_multimodal_contrastive_loss: Optional[float]=None,
                lambda_knowledge_distillation: Optional[float]=None,
                multimodal_temperature: Optional[float]=None,
                multimodal_contrastive_anneal: Optional[bool]=None,
                contrastive_logits_pos_ratio: Optional[float]=None,
                contrastive_logits_neg_ratio: Optional[float]=None,
                lambda_group_lasso: float=None,
                lambda_l1_masked: float=None,
                l1_targets_categories: Optional[list]=None,
                l1_sources_categories: Optional[list]=None,
                lambda_l1_addon: float=None, #30.,
                edge_val_ratio: float=None,
                node_val_ratio: float=None,
                edge_batch_size: Optional[int]=None,
                node_batch_size: Optional[int]=None,
                paired_data: bool=None,
                target_adata: Optional[AnnData]=None,
                target_adata_atac: Optional[AnnData]=None,
                target_holdout_frac: float=None,
                target_holdout_n: Optional[int]=None,
                target_holdout_seed: int=None,
                target_paired_data: bool=None,
                target_encoder_input_key: Optional[str]=None,
                target_counts_key: Optional[str]=None,
                log_target_multimodal_contrastive: bool=None,
                mlflow_experiment_id: Optional[str]=None,
                mlflow_parent_run_id: Optional[str]=None,
                retrieve_cat_covariates_embeds: bool=None,
                retrieve_recon_edge_probs: bool=None,
                retrieve_agg_weights: bool=None,
                use_cuda_if_available: bool=None,
                n_sampled_neighbors: int=None,
                latent_dtype: type=None,
                **trainer_kwargs):
        """
        Train the CustomNicheCompass model using CustomTrainer.
        """
        hparam_kwargs = self.__class__._apply_hparam_defaults(
            n_epochs=n_epochs,
            n_epochs_all_gps=n_epochs_all_gps,
            n_epochs_no_edge_recon=n_epochs_no_edge_recon,
            n_epochs_no_cat_covariates_contrastive=n_epochs_no_cat_covariates_contrastive,
            lr=lr,
            weight_decay=weight_decay,
            lambda_edge_recon=lambda_edge_recon,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            lambda_multimodal_contrastive_loss=lambda_multimodal_contrastive_loss,
            lambda_knowledge_distillation=lambda_knowledge_distillation,
            multimodal_temperature=multimodal_temperature,
            multimodal_contrastive_anneal=multimodal_contrastive_anneal,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            lambda_group_lasso=lambda_group_lasso,
            lambda_l1_masked=lambda_l1_masked,
            l1_targets_categories=l1_targets_categories,
            l1_sources_categories=l1_sources_categories,
            lambda_l1_addon=lambda_l1_addon,
            edge_val_ratio=edge_val_ratio,
            node_val_ratio=node_val_ratio,
            edge_batch_size=edge_batch_size,
            node_batch_size=node_batch_size,
            paired_data=paired_data,
            target_adata=target_adata,
            target_adata_atac=target_adata_atac,
            target_holdout_frac=target_holdout_frac,
            target_holdout_n=target_holdout_n,
            target_holdout_seed=target_holdout_seed,
            target_paired_data=target_paired_data,
            target_encoder_input_key=target_encoder_input_key,
            target_counts_key=target_counts_key,
            log_target_multimodal_contrastive=log_target_multimodal_contrastive,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_parent_run_id=mlflow_parent_run_id,
            retrieve_cat_covariates_embeds=retrieve_cat_covariates_embeds,
            retrieve_recon_edge_probs=retrieve_recon_edge_probs,
            retrieve_agg_weights=retrieve_agg_weights,
            use_cuda_if_available=use_cuda_if_available,
            n_sampled_neighbors=n_sampled_neighbors,
            latent_dtype=latent_dtype,
        )
        n_epochs = hparam_kwargs["n_epochs"]
        n_epochs_all_gps = hparam_kwargs["n_epochs_all_gps"]
        n_epochs_no_edge_recon = hparam_kwargs["n_epochs_no_edge_recon"]
        n_epochs_no_cat_covariates_contrastive = hparam_kwargs[
            "n_epochs_no_cat_covariates_contrastive"
        ]
        lr = hparam_kwargs["lr"]
        weight_decay = hparam_kwargs["weight_decay"]
        lambda_edge_recon = hparam_kwargs["lambda_edge_recon"]
        lambda_gene_expr_recon = hparam_kwargs["lambda_gene_expr_recon"]
        lambda_chrom_access_recon = hparam_kwargs["lambda_chrom_access_recon"]
        lambda_cat_covariates_contrastive = hparam_kwargs[
            "lambda_cat_covariates_contrastive"
        ]
        lambda_multimodal_contrastive_loss = hparam_kwargs[
            "lambda_multimodal_contrastive_loss"
        ]
        lambda_knowledge_distillation = hparam_kwargs[
            "lambda_knowledge_distillation"
        ]
        multimodal_temperature = hparam_kwargs["multimodal_temperature"]
        multimodal_contrastive_anneal = hparam_kwargs[
            "multimodal_contrastive_anneal"
        ]
        contrastive_logits_pos_ratio = hparam_kwargs[
            "contrastive_logits_pos_ratio"
        ]
        contrastive_logits_neg_ratio = hparam_kwargs[
            "contrastive_logits_neg_ratio"
        ]
        lambda_group_lasso = hparam_kwargs["lambda_group_lasso"]
        lambda_l1_masked = hparam_kwargs["lambda_l1_masked"]
        l1_targets_categories = hparam_kwargs["l1_targets_categories"]
        l1_sources_categories = hparam_kwargs["l1_sources_categories"]
        lambda_l1_addon = hparam_kwargs["lambda_l1_addon"]
        edge_val_ratio = hparam_kwargs["edge_val_ratio"]
        node_val_ratio = hparam_kwargs["node_val_ratio"]
        edge_batch_size = hparam_kwargs["edge_batch_size"]
        node_batch_size = hparam_kwargs["node_batch_size"]
        paired_data = hparam_kwargs["paired_data"]
        target_adata = hparam_kwargs["target_adata"]
        target_adata_atac = hparam_kwargs["target_adata_atac"]
        target_holdout_frac = hparam_kwargs["target_holdout_frac"]
        target_holdout_n = hparam_kwargs["target_holdout_n"]
        target_holdout_seed = hparam_kwargs["target_holdout_seed"]
        target_paired_data = hparam_kwargs["target_paired_data"]
        target_encoder_input_key = hparam_kwargs["target_encoder_input_key"]
        target_counts_key = hparam_kwargs["target_counts_key"]
        log_target_multimodal_contrastive = hparam_kwargs[
            "log_target_multimodal_contrastive"
        ]
        mlflow_experiment_id = hparam_kwargs["mlflow_experiment_id"]
        mlflow_parent_run_id = hparam_kwargs["mlflow_parent_run_id"]
        retrieve_cat_covariates_embeds = hparam_kwargs[
            "retrieve_cat_covariates_embeds"
        ]
        retrieve_recon_edge_probs = hparam_kwargs["retrieve_recon_edge_probs"]
        retrieve_agg_weights = hparam_kwargs["retrieve_agg_weights"]
        use_cuda_if_available = hparam_kwargs["use_cuda_if_available"]
        n_sampled_neighbors = hparam_kwargs["n_sampled_neighbors"]
        latent_dtype = self.__class__._coerce_numpy_dtype(hparam_kwargs["latent_dtype"])

        # Apply HPARAM defaults for trainer kwargs passed through **trainer_kwargs.
        trainer_default_kwargs = self.__class__._apply_hparam_defaults(
            target_latent_key=trainer_kwargs.get("target_latent_key"),
            use_early_stopping=trainer_kwargs.get("use_early_stopping"),
            verbose=trainer_kwargs.get("verbose"),
        )
        for key, value in trainer_default_kwargs.items():
            if trainer_kwargs.get(key) is None:
                trainer_kwargs[key] = value

        self.trainer_stage1 = CustomTrainer(
            adata=self.adata,
            adata_atac=self.adata_atac,
            model=self.model,
            counts_key=self.counts_key_,
            encoder_input_key=self.encoder_input_key_,
            adj_key=self.adj_key_,
            gp_targets_mask_key=self.gp_targets_mask_key_,
            gp_sources_mask_key=self.gp_sources_mask_key_,
            cat_covariates_keys=self.cat_covariates_keys_,
            edge_val_ratio=edge_val_ratio,
            node_val_ratio=node_val_ratio,
            edge_batch_size=edge_batch_size,
            node_batch_size=node_batch_size,
            paired_data=paired_data,
            target_adata=target_adata,
            target_adata_atac=target_adata_atac,
            target_holdout_frac=target_holdout_frac,
            target_holdout_n=target_holdout_n,
            target_holdout_seed=target_holdout_seed,
            target_paired_data=target_paired_data,
            target_encoder_input_key=target_encoder_input_key,
            target_counts_key=target_counts_key,
            log_target_multimodal_contrastive=log_target_multimodal_contrastive,
            use_cuda_if_available=use_cuda_if_available,
            n_sampled_neighbors=n_sampled_neighbors,
            latent_dtype=latent_dtype,
            **trainer_kwargs)
        
        if lambda_l1_masked > 0.:
            # Create mask for l1 regularization loss
            if l1_targets_categories is None:
                l1_targets_categories_encoded = list(self.adata.uns[
                    self.targets_categories_label_encoder_key_].values())
            else:
                l1_targets_categories_encoded = [
                    self.adata.uns[
                        self.targets_categories_label_encoder_key_][category]
                    for category in l1_targets_categories if category in
                    self.adata.uns[self.targets_categories_label_encoder_key_]]
            if l1_sources_categories is None:
                l1_sources_categories_encoded = list(self.adata.uns[
                    self.sources_categories_label_encoder_key_].values())
            else:
                l1_sources_categories_encoded = [
                    self.adata.uns[
                        self.sources_categories_label_encoder_key_][category]
                    for category in l1_sources_categories if category in
                    self.adata.uns[self.sources_categories_label_encoder_key_]]
            l1_targets_mask = torch.from_numpy(np.isin(
                self.adata.varm[self.gp_targets_categories_mask_key_],
                l1_targets_categories_encoded))
            l1_sources_mask = torch.from_numpy(np.isin(
                self.adata.varm[self.gp_sources_categories_mask_key_],
                l1_sources_categories_encoded))
        else:
            l1_targets_mask = None
            l1_sources_mask = None

        self.trainer_stage1.train(
            n_epochs=n_epochs,
            n_epochs_no_edge_recon=n_epochs_no_edge_recon,
            n_epochs_no_cat_covariates_contrastive=n_epochs_no_cat_covariates_contrastive,
            n_epochs_all_gps=n_epochs_all_gps,
            lr=lr,
            weight_decay=weight_decay,
            lambda_edge_recon=lambda_edge_recon,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            lambda_multimodal_contrastive_loss=lambda_multimodal_contrastive_loss,
            multimodal_temperature=multimodal_temperature,
            multimodal_contrastive_anneal=multimodal_contrastive_anneal,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            lambda_group_lasso=lambda_group_lasso,
            lambda_l1_masked=lambda_l1_masked,
            l1_targets_mask=l1_targets_mask,
            l1_sources_mask=l1_sources_mask,
            lambda_l1_addon=lambda_l1_addon,
            lambda_knowledge_distillation=0.0,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_parent_run_id=mlflow_parent_run_id)

        self.trainer = self.trainer_stage1
        self.node_batch_size_ = self.trainer_stage1.node_batch_size_

        self.teacher_model = copy.deepcopy(self.model).to(self.trainer_stage1.device)
        self.teacher_model.eval()
        for teacher_param in self.teacher_model.parameters():
            teacher_param.requires_grad = False

        run_stage2 = (
            lambda_knowledge_distillation > 0
            and target_adata is not None
            and target_adata_atac is not None
        )
        if run_stage2:
            stage2_counts_key = (
                self.counts_key_ if target_counts_key is None else target_counts_key
            )
            stage2_encoder_input_key = (
                self.encoder_input_key_
                if target_encoder_input_key is None
                else target_encoder_input_key
            )
            self.student_model = copy.deepcopy(self.model).to(self.trainer_stage1.device)
            self.trainer_stage2 = CustomTrainer(
                adata=target_adata,
                adata_atac=target_adata_atac,
                model=self.student_model,
                counts_key=stage2_counts_key,
                encoder_input_key=stage2_encoder_input_key,
                adj_key=self.adj_key_,
                gp_targets_mask_key=self.gp_targets_mask_key_,
                gp_sources_mask_key=self.gp_sources_mask_key_,
                cat_covariates_keys=self.cat_covariates_keys_,
                edge_val_ratio=0.0,
                node_val_ratio=0.0,
                edge_batch_size=edge_batch_size,
                node_batch_size=node_batch_size,
                paired_data=False, # will need to be more flexible for future use cases
                target_adata=target_adata,
                target_adata_atac=target_adata_atac,
                target_holdout_frac=target_holdout_frac,
                target_holdout_n=target_holdout_n,
                target_holdout_seed=target_holdout_seed,
                target_paired_data=target_paired_data,
                target_encoder_input_key=target_encoder_input_key,
                target_counts_key=target_counts_key,
                log_target_multimodal_contrastive=log_target_multimodal_contrastive,
                teacher_model=self.teacher_model,
                use_cuda_if_available=use_cuda_if_available,
                n_sampled_neighbors=n_sampled_neighbors,
                latent_dtype=latent_dtype,
                **trainer_kwargs,
            )

            stage2_run_context = nullcontext()
            if mlflow_experiment_id is not None:
                active_run = mlflow.active_run()
                parent_run_name = None
                if active_run is not None:
                    parent_run_name = active_run.data.tags.get("mlflow.runName")
                stage2_run_name = (
                    "stage2_distillation"
                    if not parent_run_name
                    else f"{parent_run_name}_stage2"
                )
                stage2_run_context = mlflow.start_run(
                    run_name=stage2_run_name,
                    nested=True,
                )

            with stage2_run_context as stage2_run:
                self.stage2_mlflow_run_id_ = (
                    stage2_run.info.run_id if stage2_run is not None else None
                )
                if stage2_run is not None:
                    mlflow.set_tag("training_stage", "stage2_distillation")
                    mlflow.set_tag("teacher_student_distillation", "true")

                self.trainer_stage2.train(
                    n_epochs=n_epochs,
                    lr=self.trainer_stage1.lr_,
                    weight_decay=self.trainer_stage1.weight_decay_,
                    lambda_edge_recon=0.0,
                    lambda_gene_expr_recon=0.0,
                    lambda_chrom_access_recon=0.0,
                    lambda_cat_covariates_contrastive=0.0,
                    lambda_multimodal_contrastive_loss=0.0,
                    lambda_knowledge_distillation=lambda_knowledge_distillation,
                    multimodal_temperature=self.trainer_stage1.multimodal_temperature_,
                    multimodal_contrastive_anneal=self.trainer_stage1.multimodal_contrastive_anneal_,
                    contrastive_logits_pos_ratio=self.trainer_stage1.contrastive_logits_pos_ratio_,
                    contrastive_logits_neg_ratio=self.trainer_stage1.contrastive_logits_neg_ratio_,
                    lambda_group_lasso=0.0,
                    lambda_l1_masked=0.0,
                    lambda_l1_addon=0.0,
                    mlflow_experiment_id=mlflow_experiment_id,
                    mlflow_parent_run_id=mlflow_parent_run_id,
                )

            # Use the distilled student as active model while keeping the
            # frozen teacher available for analysis/debugging.
            self.model = self.student_model
            self.trainer = self.trainer_stage2
            self.node_batch_size_ = self.trainer_stage2.node_batch_size_
        else:
            self.trainer_stage2 = None
            self.student_model = None
            self.stage2_mlflow_run_id_ = None
            if lambda_knowledge_distillation > 0:
                warnings.warn(
                    "Skipping stage-2 knowledge distillation because target_adata "
                    "and/or target_adata_atac were not provided."
                )

        self.is_trained_ = True
        self.model.eval()

        self.adata.obsm[self.latent_key_], _ = self.get_latent_representation(
           adata=self.adata,
           counts_key=self.counts_key_,
           adj_key=self.adj_key_,
           cat_covariates_keys=self.cat_covariates_keys_,
           only_active_gps=True,
           return_mu_std=True,
           node_batch_size=self.node_batch_size_,
           dtype=latent_dtype)

        self.adata.uns[self.active_gp_names_key_] = self.get_active_gps()

        if ((len(self.cat_covariates_cats_) > 0) &
            retrieve_cat_covariates_embeds):
            for cat_covariates_embed_key, cat_covariate_embed in zip(
                self.cat_covariates_embeds_keys_,
                self.get_cat_covariates_embeddings()):
                self.adata.uns[cat_covariates_embed_key] = cat_covariate_embed

        if retrieve_recon_edge_probs:
            self.adata.obsp[self.recon_adj_key_] = self.get_recon_edge_probs()

        if retrieve_agg_weights:
            self.adata.obsp[self.agg_weights_key_] = (
                self.get_neighbor_importances(
                    node_batch_size=self.node_batch_size_))

        if mlflow_experiment_id is not None:
            mlflow.log_metric("n_active_gps",
                              len(self.adata.uns[self.active_gp_names_key_]))


    def get_latent_representation(
            self,
            adata: Optional[AnnData]=None,
            adata_atac: Optional[AnnData]=None,
            counts_key: Optional[str]=None,
            encoder_input_key: Optional[str]=None,
            adj_key: str=None,
            cat_covariates_keys: Optional[List[str]]=None,
            paired_data: bool=None,
            only_active_gps: bool=True,
            return_mu_std: bool=False,
            node_batch_size: Optional[int]=None,
            dtype: type=None,
            separate_modalities: bool=False,
            return_clip_embeddings: bool=False,
            ) -> np.ndarray:
        """
        Get the latent representation / gene program scores from a trained model.
        
        Parameters
        ----------
        separate_modalities : bool
            If True, returns separate RNA and ATAC latents instead of combined.
            When True and return_mu_std=True, returns (mu_rna, std_rna, mu_atac, std_atac).
            When True and return_mu_std=False, returns (z_rna, z_atac).
        return_clip_embeddings : bool
            If True (and `separate_modalities=True` and `return_mu_std=True`),
            also returns the outputs of forwarding the RNA/ATAC means through
            `self.model.multimodal_layer`. These are returned as
            `(clip_embeddings_rna, clip_embeddings_atac)`.
            Note: `clip_embeddings` are only valid when
            `separate_modalities=True` and `return_mu_std=True`.
        """
        self._check_if_trained(warn=False)

        device = next(self.model.parameters()).device

        if adata is None:
            adata = self.adata
        if (adata_atac is None) & hasattr(self, "adata_atac"):
            adata_atac = self.adata_atac
        if counts_key is None:
            counts_key = self.counts_key_
        if encoder_input_key is None:
            encoder_input_key = self.encoder_input_key_
        if adj_key is None:
            adj_key = self.adj_key_
        hparam_kwargs = self.__class__._apply_hparam_defaults(
            paired_data=paired_data,
            node_batch_size=node_batch_size,
            latent_dtype=dtype,
        )
        paired_data = hparam_kwargs["paired_data"]
        node_batch_size = hparam_kwargs["node_batch_size"]
        dtype = self.__class__._coerce_numpy_dtype(hparam_kwargs["latent_dtype"])

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=adata_atac,
            counts_key=counts_key,
            encoder_input_key=encoder_input_key,
            adj_key=adj_key,
            cat_covariates_keys=cat_covariates_keys,
            paired_data=paired_data,
            edge_val_ratio=0.,
            edge_test_ratio=0.,
            node_val_ratio=0.,
            node_test_ratio=0.)
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"))
        node_loader = loader_dict["node_train_loader"]

        # Get number of gene programs
        if only_active_gps:
            n_gps = self.get_active_gps().shape[0]
        else:
            n_gps = (self.n_prior_gp_ + self.n_addon_gp_ )

        n_obs = node_masked_data.num_nodes

        if return_clip_embeddings and (not separate_modalities):
            raise ValueError(
                "return_clip_embeddings=True requires separate_modalities=True "
                "because clip_embeddings are modality-specific.")
        if return_clip_embeddings and (not return_mu_std):
            raise ValueError(
                "return_clip_embeddings=True requires return_mu_std=True "
                "because clip_embeddings are computed from modality means (mu).")
        
        if separate_modalities:
            if return_clip_embeddings:
                clip_dim = int(
                    _get_module_out_features(self.model.multimodal_layer)
                    or n_gps
                )
                clip_embeddings_rna = np.empty(shape=(n_obs, clip_dim), dtype=dtype)
                clip_embeddings_atac = np.empty(shape=(n_obs, clip_dim), dtype=dtype)
            if return_mu_std:
                mu_rna = np.empty(shape=(n_obs, n_gps), dtype=dtype)
                std_rna = np.empty(shape=(n_obs, n_gps), dtype=dtype)
                mu_atac = np.empty(shape=(n_obs, n_gps), dtype=dtype)
                std_atac = np.empty(shape=(n_obs, n_gps), dtype=dtype)
            else:
                z_rna = np.empty(shape=(n_obs, n_gps), dtype=dtype)
                z_atac = np.empty(shape=(n_obs, n_gps), dtype=dtype)
        else:
            if return_mu_std:
                mu = np.empty(shape=(n_obs, n_gps), dtype=dtype)
                std = np.empty(shape=(n_obs, n_gps), dtype=dtype)
            else:
                z = np.empty(shape=(n_obs, n_gps), dtype=dtype)

        # Get latent representation for each batch of the dataloader and put it
        # into latent vectors
        # Get model dtype to ensure consistency
        model_dtype = next(self.model.parameters()).dtype
        for i, node_batch in enumerate(node_loader):
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
            node_batch = node_batch.to(device, non_blocking=True)
            # Ensure node_batch.x has the same dtype as the model
            if node_batch.x.dtype != model_dtype:
                node_batch.x = node_batch.x.to(model_dtype)
            
            if separate_modalities:
                if return_mu_std:
                    if return_clip_embeddings:
                        (mu_rna_batch,
                         std_rna_batch,
                         mu_atac_batch,
                         std_atac_batch,
                         clip_rna_batch,
                         clip_atac_batch) = self.model.get_latent_representation(
                            node_batch=node_batch,
                            only_active_gps=only_active_gps,
                            return_mu_std=True,
                            separate_modalities=True,
                            return_clip_embeddings=True)
                    else:
                        mu_rna_batch, std_rna_batch, mu_atac_batch, std_atac_batch = (
                            self.model.get_latent_representation(
                                node_batch=node_batch,
                                only_active_gps=only_active_gps,
                                return_mu_std=True,
                                separate_modalities=True,
                                return_clip_embeddings=False))
                    mu_rna[n_obs_before_batch:n_obs_after_batch, :] = (
                        mu_rna_batch.detach().cpu().numpy())
                    std_rna[n_obs_before_batch:n_obs_after_batch, :] = (
                        std_rna_batch.detach().cpu().numpy())
                    mu_atac[n_obs_before_batch:n_obs_after_batch, :] = (
                        mu_atac_batch.detach().cpu().numpy())
                    std_atac[n_obs_before_batch:n_obs_after_batch, :] = (
                        std_atac_batch.detach().cpu().numpy())
                    if return_clip_embeddings:
                        clip_embeddings_rna[n_obs_before_batch:n_obs_after_batch, :] = (
                            clip_rna_batch.detach().cpu().numpy())
                        clip_embeddings_atac[n_obs_before_batch:n_obs_after_batch, :] = (
                            clip_atac_batch.detach().cpu().numpy())
                else:
                    z_rna_batch, z_atac_batch = self.model.get_latent_representation(
                        node_batch=node_batch,
                        only_active_gps=only_active_gps,
                        return_mu_std=False,
                        separate_modalities=True,
                        return_clip_embeddings=False)
                    z_rna[n_obs_before_batch:n_obs_after_batch, :] = (
                        z_rna_batch.detach().cpu().numpy())
                    z_atac[n_obs_before_batch:n_obs_after_batch, :] = (
                        z_atac_batch.detach().cpu().numpy())
            else:
                if return_mu_std:
                    mu_batch, std_batch = self.model.get_latent_representation(
                        node_batch=node_batch,
                        only_active_gps=only_active_gps,
                        return_mu_std=True,
                        separate_modalities=False)
                    mu[n_obs_before_batch:n_obs_after_batch, :] = (
                        mu_batch.detach().cpu().numpy())
                    std[n_obs_before_batch:n_obs_after_batch, :] = (
                        std_batch.detach().cpu().numpy())
                else:
                    z_batch = self.model.get_latent_representation(
                        node_batch=node_batch,
                        only_active_gps=only_active_gps,
                        return_mu_std=False,
                        separate_modalities=False)
                    z[n_obs_before_batch:n_obs_after_batch, :] = (
                        z_batch.detach().cpu().numpy())
        
        if separate_modalities:
            if return_mu_std:
                if return_clip_embeddings:
                    return (mu_rna,
                            std_rna,
                            mu_atac,
                            std_atac,
                            clip_embeddings_rna,
                            clip_embeddings_atac)
                return mu_rna, std_rna, mu_atac, std_atac
            else:
                return z_rna, z_atac
        else:
            if return_mu_std:
                return mu, std
            else:
                return z

    def get_omics_decoder_outputs(
            self,
            adata: Optional[AnnData]=None,
            adata_atac: Optional[AnnData]=None,
            paired_data: bool=None,
            only_active_gps: bool=True,
            node_batch_size: Optional[int]=None,
            encoder_input_key: Optional[str]=None,
            ) -> dict:
        """
        Get the omics decoder outputs.
        """
        self._check_if_trained(warn=False)

        device = next(self.model.parameters()).device

        if adata is None:
            adata = self.adata
        if (adata_atac is None) & hasattr(self, "adata_atac"):
            adata_atac = self.adata_atac
        if encoder_input_key is None:
            encoder_input_key = self.encoder_input_key_
        hparam_kwargs = self.__class__._apply_hparam_defaults(
            paired_data=paired_data,
            node_batch_size=node_batch_size,
        )
        paired_data = hparam_kwargs["paired_data"]
        node_batch_size = hparam_kwargs["node_batch_size"]

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=adata_atac,
            counts_key=self.counts_key_,
            encoder_input_key=encoder_input_key,
            adj_key=self.adj_key_,
            cat_covariates_keys=self.cat_covariates_keys_,
            paired_data=paired_data,
            edge_val_ratio=0.,
            edge_test_ratio=0.,
            node_val_ratio=0.,
            node_test_ratio=0.)
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"))
        node_loader = loader_dict["node_train_loader"]

        n_obs = node_masked_data.num_nodes
        output = {}
        output["target_rna_nb_means"] = np.empty(
            shape=(n_obs, self.n_output_genes_))
        output["source_rna_nb_means"] = np.empty(
            shape=(n_obs, self.n_output_genes_))
        if "atac" in self.modalities_:
            output["target_atac_nb_means"] = np.empty(
                shape=(n_obs, self.n_output_peaks_))
            output["source_atac_nb_means"] = np.empty(
                shape=(n_obs, self.n_output_peaks_))

        # Get latent representation for each batch of the dataloader and put it
        # into latent vectors
        # Get model dtype to ensure consistency
        model_dtype = next(self.model.parameters()).dtype
        for i, node_batch in enumerate(node_loader):
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
            node_batch = node_batch.to(device, non_blocking=True)
            # Ensure node_batch.x has the same dtype as the model
            if node_batch.x.dtype != model_dtype:
                node_batch.x = node_batch.x.to(model_dtype)
            output_batch = self.model.get_omics_decoder_outputs(
                node_batch=node_batch,
                only_active_gps=only_active_gps)
            output["target_rna_nb_means"][
                n_obs_before_batch:n_obs_after_batch, :] = (
                    output_batch["target_rna_nb_means"].detach().cpu().numpy())
            output["source_rna_nb_means"][
                n_obs_before_batch:n_obs_after_batch, :] = (
                    output_batch["source_rna_nb_means"].detach().cpu().numpy())
            if "atac" in self.modalities_:
                output["target_atac_nb_means"][
                    n_obs_before_batch:n_obs_after_batch, :] = (
                        output_batch["target_atac_nb_means"].detach()
                        .cpu().numpy())
                output["source_atac_nb_means"][
                    n_obs_before_batch:n_obs_after_batch, :] = (
                        output_batch["source_atac_nb_means"].detach()
                        .cpu().numpy())
        return output

class CustomTrainer(Trainer):
    """
    Trainer that uses the project-specific prepare_data implementation.
    """

    def __init__(
        self,
        *args,
        paired_data: bool=None,
        encoder_input_key: Optional[str]=None,
        target_adata: Optional[AnnData]=None,
        target_adata_atac: Optional[AnnData]=None,
        target_holdout_frac: float=None,
        target_holdout_n: Optional[int]=None,
        target_holdout_seed: int=None,
        target_paired_data: bool=None,
        target_encoder_input_key: Optional[str]=None,
        target_counts_key: Optional[str]=None,
        log_target_multimodal_contrastive: bool=None,
        target_latent_key: str=None,
        teacher_model: Optional[torch.nn.Module]=None,
        **kwargs
    ):
        latent_dtype = kwargs.pop("latent_dtype", None)
        hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
            paired_data=paired_data,
            encoder_input_key=encoder_input_key,
            target_adata=target_adata,
            target_adata_atac=target_adata_atac,
            target_holdout_frac=target_holdout_frac,
            target_holdout_n=target_holdout_n,
            target_holdout_seed=target_holdout_seed,
            target_paired_data=target_paired_data,
            target_encoder_input_key=target_encoder_input_key,
            target_counts_key=target_counts_key,
            log_target_multimodal_contrastive=log_target_multimodal_contrastive,
            target_latent_key=target_latent_key,
            latent_dtype=latent_dtype,
            use_early_stopping=kwargs.get("use_early_stopping"),
            verbose=kwargs.get("verbose"),
        )
        paired_data = hparam_kwargs["paired_data"]
        encoder_input_key = hparam_kwargs["encoder_input_key"]
        target_adata = hparam_kwargs["target_adata"]
        target_adata_atac = hparam_kwargs["target_adata_atac"]
        target_holdout_frac = hparam_kwargs["target_holdout_frac"]
        target_holdout_n = hparam_kwargs["target_holdout_n"]
        target_holdout_seed = hparam_kwargs["target_holdout_seed"]
        target_paired_data = hparam_kwargs["target_paired_data"]
        target_encoder_input_key = hparam_kwargs["target_encoder_input_key"]
        target_counts_key = hparam_kwargs["target_counts_key"]
        log_target_multimodal_contrastive = hparam_kwargs[
            "log_target_multimodal_contrastive"
        ]
        target_latent_key = hparam_kwargs["target_latent_key"]
        kwargs["use_early_stopping"] = hparam_kwargs["use_early_stopping"]
        kwargs["verbose"] = hparam_kwargs["verbose"]
        self.latent_dtype_ = CustomNicheCompass._coerce_numpy_dtype(
            hparam_kwargs["latent_dtype"]
        )
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()
            for teacher_param in self.teacher_model.parameters():
                teacher_param.requires_grad = False

        data_dict = prepare_data(
            adata=self.adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=self.adata_atac,
            counts_key=self.counts_key,
            encoder_input_key=encoder_input_key,
            adj_key=self.adj_key,
            cat_covariates_keys=self.cat_covariates_keys,
            paired_data=paired_data,
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=0.,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=0.)

        self.node_masked_data = data_dict["node_masked_data"]
        self.edge_train_data = data_dict["edge_train_data"]
        self.edge_val_data = data_dict["edge_val_data"]
        self.n_nodes_train = self.node_masked_data.train_mask.sum().item()
        self.n_nodes_val = self.node_masked_data.val_mask.sum().item()
        self.n_edges_train = self.edge_train_data.edge_label_index.size(1)
        self.n_edges_val = self.edge_val_data.edge_label_index.size(1)

        if self.node_batch_size_ is None:
            self.node_batch_size_ = int(self.edge_batch_size_ / math.floor(
                self.n_edges_train / self.n_nodes_train))

        loader_dict = initialize_dataloaders(
            node_masked_data=self.node_masked_data,
            edge_train_data=self.edge_train_data,
            edge_val_data=self.edge_val_data,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            n_direct_neighbors=self.n_sampled_neighbors_,
            n_hops=self.loaders_n_hops_,
            edges_directed=False,
            neg_edge_sampling_ratio=1.,
            pin_memory=(self.device.type == "cuda"))
        self.edge_train_loader = loader_dict["edge_train_loader"]
        self.edge_val_loader = loader_dict.pop("edge_val_loader", None)
        self.node_train_loader = loader_dict["node_train_loader"]
        self.node_val_loader = loader_dict.pop("node_val_loader", None)

        self.log_target_multimodal_contrastive_ = log_target_multimodal_contrastive
        self.target_adata = target_adata
        self.target_adata_atac = target_adata_atac
        self.target_paired_data_ = target_paired_data
        self.target_encoder_input_key_ = (
            encoder_input_key if target_encoder_input_key is None
            else target_encoder_input_key)
        self.target_counts_key_ = (
            self.counts_key if target_counts_key is None
            else target_counts_key)
        self.target_holdout_adata = None
        self.target_holdout_adata_atac = None
        self.target_latent_key = target_latent_key
        self._source_umap_clip_rna_batches: List[np.ndarray] = []
        self._source_umap_clip_atac_batches: List[np.ndarray] = []
        self._source_umap_collected_n = 0
        self._source_umap_target_n = 0

        if (self.log_target_multimodal_contrastive_ and
            target_adata is not None and
            target_adata_atac is not None):
            rna, atac = self._subset_target_holdout(
                target_adata=target_adata,
                target_adata_atac=target_adata_atac,
                paired_data=target_paired_data,
                holdout_frac=target_holdout_frac,
                holdout_n=target_holdout_n,
                seed=target_holdout_seed,
            )
            if rna is not None and atac is not None:
                self.target_holdout_adata = rna
                self.target_holdout_adata_atac = atac

    def _subset_target_holdout(
        self,
        target_adata: AnnData,
        target_adata_atac: AnnData,
        paired_data: bool,
        holdout_frac: float,
        holdout_n: Optional[int],
        seed: int,
    ) -> Tuple[Optional[AnnData], Optional[AnnData]]:
        if paired_data:
            shared = target_adata.obs_names.intersection(target_adata_atac.obs_names)
        else:
            shared = target_adata.obs_names.intersection(target_adata_atac.obs_names)
            if len(shared) == 0:
                warnings.warn(
                    "Target holdout requires paired obs_names for "
                    "multimodal contrastive logging; skipping target logging.")
                return None, None

        if holdout_n is None:
            holdout_n = int(max(1, round(len(shared) * holdout_frac)))
        holdout_n = min(holdout_n, len(shared))
        rng = np.random.default_rng(seed)
        holdout_obs = pd.Index(rng.choice(shared, size=holdout_n, replace=False))
        return target_adata[holdout_obs].copy(), target_adata_atac[holdout_obs].copy()

    def _build_chunk_data(
        self,
        rna_chunk: AnnData,
        atac_chunk: Optional[AnnData],
        paired_data: bool,
    ) -> Data:
        dataset = CustomSpatialAnnTorchDataset(
            adata=rna_chunk,
            adata_atac=atac_chunk,
            counts_key=self.target_counts_key_,
            encoder_input_key=self.target_encoder_input_key_,
            adj_key=self.adj_key,
            cat_covariates_keys=self.cat_covariates_keys,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            paired_data=paired_data)
        data = Data(
            x=dataset.x,
            edge_index=dataset.edge_index,
            edge_attr=dataset.edge_index.t())
        data.x_counts = dataset.x_counts
        if dataset.modality_mask is not None:
            data.modality_mask = dataset.modality_mask
        if self.cat_covariates_keys is not None:
            data.cat_covariates_cats = dataset.cat_covariates_cats
        data.batch_size = data.num_nodes
        return data

    def _log_memory_optimized_umap(
        self,
        clip_embeddings_adata: AnnData,
        artifact_path: str,
        title: str,
    ) -> None:
        n_obs = int(clip_embeddings_adata.n_obs)
        n_vars = int(clip_embeddings_adata.n_vars)
        if n_obs < 3:
            warnings.warn(
                f"Skipping UMAP logging for '{artifact_path}' because n_obs={n_obs} < 3."
            )
            return

        clip_embeddings_adata.X = np.asarray(clip_embeddings_adata.X, dtype=np.float32)
        n_comps = min(30, n_vars, n_obs - 1)
        n_neighbors = min(30, n_obs - 1)
        if n_comps < 2 or n_neighbors < 2:
            warnings.warn(
                f"Skipping UMAP logging for '{artifact_path}' due to insufficient dimensions "
                f"(n_comps={n_comps}, n_neighbors={n_neighbors})."
            )
            return

        sc.pp.pca(clip_embeddings_adata, n_comps=n_comps)
        sc.pp.neighbors(
            clip_embeddings_adata,
            use_rep="X_pca",
            n_neighbors=n_neighbors,
            method="umap",
        )
        sc.tl.umap(clip_embeddings_adata, min_dist=0.3, random_state=0)

        umap_fig = None
        try:
            with plt.rc_context({"figure.dpi": 100, "savefig.dpi": 100}):
                umap_fig = sc.pl.umap(
                    clip_embeddings_adata,
                    color=["modality"],
                    size=10,
                    frameon=False,
                    show=False,
                    return_fig=True,
                    title=title,
                )
                for ax in umap_fig.axes:
                    for collection in ax.collections:
                        collection.set_rasterized(True)
                umap_fig.set_size_inches(7, 5)
                mlflow.log_figure(umap_fig, artifact_path)
        finally:
            if umap_fig is not None:
                plt.close(umap_fig)
            plt.close("all")

    def _log_source_umap_matched_to_target(self, target_holdout_n: int) -> None:
        if not self._source_umap_clip_rna_batches or not self._source_umap_clip_atac_batches:
            warnings.warn(
                "No source embeddings cached from training iterations; "
                "skipping source UMAP logging."
            )
            return

        source_clip_rna = np.concatenate(self._source_umap_clip_rna_batches, axis=0)
        source_clip_atac = np.concatenate(self._source_umap_clip_atac_batches, axis=0)
        available_n = min(source_clip_rna.shape[0], source_clip_atac.shape[0])
        if available_n <= 0:
            warnings.warn(
                "Cached source embeddings are empty; skipping source UMAP logging."
            )
            return

        source_n = target_holdout_n
        if available_n < target_holdout_n:
            warnings.warn(
                f"Only {available_n} unique source samples were cached from training iterations; "
                f"reusing cached samples to reach target size {target_holdout_n}."
            )
            rng = np.random.default_rng(0)
            sample_idx = np.concatenate(
                [
                    np.arange(available_n, dtype=np.int64),
                    rng.choice(
                        available_n,
                        size=target_holdout_n - available_n,
                        replace=True,
                    ),
                ]
            )
        else:
            sample_idx = np.arange(target_holdout_n, dtype=np.int64)

        modality_obs = pd.DataFrame(
            {"modality": ["rna"] * source_n + ["atac"] * source_n}
        )

        source_embeddings_adata = AnnData(
            X=np.concatenate(
                [
                    source_clip_rna[sample_idx].astype(np.float32, copy=False),
                    source_clip_atac[sample_idx].astype(np.float32, copy=False),
                ],
                axis=0,
            ),
            obs=modality_obs,
        )
        self._log_memory_optimized_umap(
            clip_embeddings_adata=source_embeddings_adata,
            artifact_path=f"umap/source_data_umap_epoch_{self.epoch + 1}.png",
            title="Source Data UMAP",
        )

    @torch.no_grad()
    def _get_target_latent_embeddings(
        self,
        adata: AnnData,
        adata_atac: Optional[AnnData],
        paired_data: bool,
        chunk_size: int=500,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        was_training = self.model.training
        if was_training:
            self.model.eval()

        clip_embeddings_rna = []
        clip_embeddings_atac = []
        multimodal_contrastive_loss = []

        model_dtype = next(self.model.parameters()).dtype
        for _, start, end in adata.chunked_X(chunk_size):
            rna_chunk = adata[start:end].copy()
            atac_chunk = adata_atac[start:end].copy() if adata_atac is not None else None
            if paired_data:
                assert rna_chunk.obs_names.equals(atac_chunk.obs_names), "RNA and ATAC chunks have different obs_names despite target data being paired"

            node_batch = self._build_chunk_data(
                rna_chunk=rna_chunk,
                atac_chunk=atac_chunk,
                paired_data=paired_data)
            node_batch = node_batch.to(self.device, non_blocking=True)
            if node_batch.x.dtype != model_dtype:
                node_batch.x = node_batch.x.to(model_dtype)

            _, _, _, _, clip_embeddings_rna_batch, clip_embeddings_atac_batch = (
                self.model.get_latent_representation(
                    node_batch=node_batch,
                    only_active_gps=self.use_only_active_gps,
                    return_mu_std=True,
                    separate_modalities=True,
                    return_clip_embeddings=True)
            )

            ## evaluate multimodal contrastive loss on target data
            similarity = self.model.get_multimodal_similarity(
                clip_embeddings_rna=clip_embeddings_rna_batch,
                clip_embeddings_atac=clip_embeddings_atac_batch)

            loss = self.model.compute_multimodal_contrastive_loss(
                similarity_matrix=similarity,
                temperature=self.multimodal_temperature_)
            multimodal_contrastive_loss.append(float(loss.item()))

            ## store clip embeddings
            clip_embeddings_rna.append(clip_embeddings_rna_batch.detach().cpu().numpy())
            if clip_embeddings_atac_batch is not None:
                clip_embeddings_atac.append(clip_embeddings_atac_batch.detach().cpu().numpy())            

        clip_embeddings_rna = np.concatenate(clip_embeddings_rna, axis=0)
        clip_embeddings_atac = np.concatenate(clip_embeddings_atac, axis=0) if clip_embeddings_atac else None

        multimodal_contrastive_loss = np.mean(multimodal_contrastive_loss)
        self.epoch_logs["target_multimodal_contrastive_loss"].append(multimodal_contrastive_loss)
        if self.mlflow_experiment_id is not None:
            mlflow.log_metric("target_multimodal_contrastive_loss", multimodal_contrastive_loss, step=self.epoch)

        if was_training:
            self.model.train()

        return clip_embeddings_rna, clip_embeddings_atac

    def _log_target_paired_metrics(
        self,
        temperature: float,
        clip_embeddings_rna: torch.Tensor,
        clip_embeddings_atac: torch.Tensor,
    ) -> None:

        foscttm_full = foscttm_moscot(
            clip_embeddings_rna,
            clip_embeddings_atac,
        )
        foscttm_mean = float(np.asarray(foscttm_full).mean())

        # calculate 1 - FOSCTTM, such that the higher the better
        one_m_foscttm = 1 - foscttm_mean

        # Log directly at epoch granularity to avoid mixing with per-iter logs.
        self.epoch_logs["target_one_minus_foscttm"].append(one_m_foscttm)
        if self.mlflow_experiment_id is not None:
            mlflow.log_metric("target_one_minus_foscttm", one_m_foscttm, step=self.epoch)

    def _log_target_unpaired_metrics(self) -> None:

        if self.target_latent_key not in self.target_holdout_adata.obsm:
            warnings.warn(
                "Target latent embeddings not found; skipping target metrics. "
                "Run embedding extraction before metric logging.")
            return

        ## cocnatenate target_holdout_adata and target_holdout_adata_atac
        clip_embeddings_adata = AnnData(
            X=np.concatenate([
                self.target_holdout_adata.obsm[self.target_latent_key].astype(np.float32, copy=False),
                self.target_holdout_adata_atac.obsm[self.target_latent_key].astype(np.float32, copy=False),
            ], axis=0),
            obs=pd.concat([
                self.target_holdout_adata.obs.assign(modality="rna"),
                self.target_holdout_adata_atac.obs.assign(modality="atac"),
            ], axis=0),
            uns={"label_key": self.target_holdout_adata.uns["label_key"]},
            #uns={"label_key": None},
        )

        scib_n_jobs = 1
        scib_n_jobs_env = os.environ.get("BAKLAVA_SCIB_N_JOBS")
        if scib_n_jobs_env is not None:
            try:
                scib_n_jobs = max(1, int(scib_n_jobs_env))
            except ValueError:
                warnings.warn(
                    f"Invalid BAKLAVA_SCIB_N_JOBS='{scib_n_jobs_env}', "
                    "falling back to 1."
                )

        results_dict = benchmark_embeddings(
            adata=clip_embeddings_adata,
            batch_key="modality",
            label_key=clip_embeddings_adata.uns['label_key'],
            embedding_obsm_keys=["nichecompass_latent"],
            n_jobs=scib_n_jobs,
        )
        for key, value in results_dict.items():
            self.epoch_logs[f"target_{key}".replace(" ", "_")].append(value)

        if self.mlflow_experiment_id is not None:
            for key, value in results_dict.items():
                mlflow.log_metric(f"target_{key}".replace(" ", "_"), value, step=self.epoch)

            if self.epoch == self.n_epochs_ - 1:
                try:
                    self._log_memory_optimized_umap(
                        clip_embeddings_adata=clip_embeddings_adata.copy(),
                        artifact_path=f"umap/target_holdout_umap_epoch_{self.epoch + 1}.png",
                        title="Target Holdout UMAP",
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Failed to generate/log target holdout UMAP to MLflow: {exc}"
                    )

                try:
                    self._log_source_umap_matched_to_target(
                        target_holdout_n=int(self.target_holdout_adata.n_obs),
                    )
                except Exception as exc:
                    warnings.warn(
                        f"Failed to generate/log source UMAP to MLflow: {exc}"
                    )


    def _get_multimodal_contrastive_weight(self, increasing: bool=True) -> float:
        if (not self.multimodal_contrastive_anneal_) or self.n_epochs_ <= 1:
            return self.lambda_multimodal_contrastive_loss_
        progress = self.epoch / max(1, self.n_epochs_ - 1)
        k = 0.5 * (1.0 + math.cos(math.pi * progress))
        if increasing:
            return (1-k) * self.lambda_multimodal_contrastive_loss_
        else:
            return k * self.lambda_multimodal_contrastive_loss_

    def train(self,
              n_epochs: int=None,
              n_epochs_all_gps: int=None,
              n_epochs_no_edge_recon: int=None,
              n_epochs_no_cat_covariates_contrastive: int=None,
              target_eval_interval: int=10,
              lr: float=None,
              weight_decay: float=None,
              lambda_edge_recon: Optional[float]=None,
              lambda_cat_covariates_contrastive: Optional[float]=None,
              lambda_multimodal_contrastive_loss: Optional[float]=None,
              lambda_knowledge_distillation: Optional[float]=None,
              multimodal_temperature: Optional[float]=None,
              multimodal_contrastive_anneal: Optional[bool]=None,
              contrastive_logits_pos_ratio: Optional[float]=None,
              contrastive_logits_neg_ratio: Optional[float]=None,
              lambda_gene_expr_recon: float=None,
              lambda_chrom_access_recon: float=None,
              lambda_group_lasso: float=None,
              lambda_l1_masked: float=None,
              l1_targets_mask: Optional[torch.Tensor]=None,
              l1_sources_mask: Optional[torch.Tensor]=None,
              lambda_l1_addon: float=None,
              mlflow_experiment_id: Optional[str]=None,
              mlflow_parent_run_id: Optional[str]=None):
        """
        Train the CustomNicheCompass model.
        """
        hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
            n_epochs=n_epochs,
            n_epochs_all_gps=n_epochs_all_gps,
            n_epochs_no_edge_recon=n_epochs_no_edge_recon,
            n_epochs_no_cat_covariates_contrastive=n_epochs_no_cat_covariates_contrastive,
            lr=lr,
            weight_decay=weight_decay,
            lambda_edge_recon=lambda_edge_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            lambda_multimodal_contrastive_loss=lambda_multimodal_contrastive_loss,
            lambda_knowledge_distillation=lambda_knowledge_distillation,
            multimodal_temperature=multimodal_temperature,
            multimodal_contrastive_anneal=multimodal_contrastive_anneal,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_group_lasso=lambda_group_lasso,
            lambda_l1_masked=lambda_l1_masked,
            lambda_l1_addon=lambda_l1_addon,
            mlflow_experiment_id=mlflow_experiment_id,
            mlflow_parent_run_id=mlflow_parent_run_id,
        )
        n_epochs = hparam_kwargs["n_epochs"]
        n_epochs_all_gps = hparam_kwargs["n_epochs_all_gps"]
        n_epochs_no_edge_recon = hparam_kwargs["n_epochs_no_edge_recon"]
        n_epochs_no_cat_covariates_contrastive = hparam_kwargs[
            "n_epochs_no_cat_covariates_contrastive"
        ]
        lr = hparam_kwargs["lr"]
        weight_decay = hparam_kwargs["weight_decay"]
        lambda_edge_recon = hparam_kwargs["lambda_edge_recon"]
        lambda_cat_covariates_contrastive = hparam_kwargs[
            "lambda_cat_covariates_contrastive"
        ]
        lambda_multimodal_contrastive_loss = hparam_kwargs[
            "lambda_multimodal_contrastive_loss"
        ]
        lambda_knowledge_distillation = hparam_kwargs[
            "lambda_knowledge_distillation"
        ]
        multimodal_temperature = hparam_kwargs["multimodal_temperature"]
        multimodal_contrastive_anneal = hparam_kwargs[
            "multimodal_contrastive_anneal"
        ]
        contrastive_logits_pos_ratio = hparam_kwargs[
            "contrastive_logits_pos_ratio"
        ]
        contrastive_logits_neg_ratio = hparam_kwargs[
            "contrastive_logits_neg_ratio"
        ]
        lambda_gene_expr_recon = hparam_kwargs["lambda_gene_expr_recon"]
        lambda_chrom_access_recon = hparam_kwargs["lambda_chrom_access_recon"]
        lambda_group_lasso = hparam_kwargs["lambda_group_lasso"]
        lambda_l1_masked = hparam_kwargs["lambda_l1_masked"]
        lambda_l1_addon = hparam_kwargs["lambda_l1_addon"]
        mlflow_experiment_id = hparam_kwargs["mlflow_experiment_id"]
        mlflow_parent_run_id = hparam_kwargs["mlflow_parent_run_id"]
        self.n_epochs_ = n_epochs
        self.n_epochs_all_gps_ = n_epochs_all_gps
        self.n_epochs_no_edge_recon_ = n_epochs_no_edge_recon
        self.n_epochs_no_cat_covariates_contrastive_ = (
            n_epochs_no_cat_covariates_contrastive)
        self.lr_ = lr
        self.weight_decay_ = weight_decay
        self.lambda_edge_recon_ = lambda_edge_recon
        self.lambda_gene_expr_recon_ = lambda_gene_expr_recon
        self.lambda_chrom_access_recon_ = lambda_chrom_access_recon
        self.lambda_cat_covariates_contrastive_ = (
            lambda_cat_covariates_contrastive)
        self.lambda_multimodal_contrastive_loss_ = (
            lambda_multimodal_contrastive_loss)
        self.lambda_knowledge_distillation_ = lambda_knowledge_distillation
        self.multimodal_temperature_ = multimodal_temperature
        self.multimodal_contrastive_anneal_ = multimodal_contrastive_anneal
        self.contrastive_logits_pos_ratio_ = contrastive_logits_pos_ratio
        self.contrastive_logits_neg_ratio_ = contrastive_logits_neg_ratio
        self.lambda_group_lasso_ = lambda_group_lasso
        self.lambda_l1_masked_ = lambda_l1_masked
        self.l1_targets_mask = l1_targets_mask
        self.l1_sources_mask = l1_sources_mask
        self.lambda_l1_addon_ = lambda_l1_addon
        self.mlflow_experiment_id = mlflow_experiment_id
        self.mlflow_parent_run_id = mlflow_parent_run_id

        if self.lambda_knowledge_distillation_ is None:
            self.lambda_knowledge_distillation_ = 0.0
        if self.lambda_knowledge_distillation_ > 0 and self.teacher_model is None:
            warnings.warn(
                "lambda_knowledge_distillation > 0 but no teacher_model was "
                "provided to CustomTrainer; disabling KD for this training run."
            )
            self.lambda_knowledge_distillation_ = 0.0

        print("\n--- MODEL TRAINING ---")

        if self.mlflow_experiment_id is not None:
            # Log hyperparameters to parent run if requested (e.g. HPO study run),
            # so the child run only has trial params and metrics.
            if self.mlflow_parent_run_id is not None:
                # MLflow does not allow switching the "active run" while a nested
                # run is active. Instead, we route param logging directly to the
                # parent run via MlflowClient, while keeping the child run active
                # for metrics/artifacts.
                from mlflow.tracking import MlflowClient

                client = MlflowClient()
                # Avoid crashing on repeated trials (MLflow params are immutable).
                existing_params = dict(client.get_run(self.mlflow_parent_run_id).data.params)

                orig_log_param = mlflow.log_param
                orig_log_params = mlflow.log_params

                def _log_param_to_parent(key, value, *args, **kwargs):
                    # Only log if key not already present on parent.
                    if key in existing_params:
                        return
                    client.log_param(self.mlflow_parent_run_id, key, str(value))
                    existing_params[key] = str(value)

                def _log_params_to_parent(d, *args, **kwargs):
                    for k, v in d.items():
                        _log_param_to_parent(k, v)

                try:
                    mlflow.log_param = _log_param_to_parent
                    mlflow.log_params = _log_params_to_parent
                    for attr, attr_value in self._get_public_attributes().items():
                        mlflow.log_param(attr, attr_value)
                    # This method likely uses mlflow.log_param internally; the
                    # monkeypatch above routes those params to the parent run.
                    self.model.log_module_hyperparams_to_mlflow()
                finally:
                    mlflow.log_param = orig_log_param
                    mlflow.log_params = orig_log_params
            else:
                for attr, attr_value in self._get_public_attributes().items():
                    mlflow.log_param(attr, attr_value)
                self.model.log_module_hyperparams_to_mlflow()

        start_time = time.time()
        self.epoch_logs = defaultdict(list)
        self.model.train()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=lr,
                                          weight_decay=weight_decay)

        for self.epoch in range(n_epochs):
            if self.epoch < self.n_epochs_no_edge_recon_:
                self.edge_recon_active = False
            else:
                self.edge_recon_active = True
            if self.epoch < self.n_epochs_all_gps_:
                self.use_only_active_gps = False
            else:
                self.use_only_active_gps = True
            if self.epoch < self.n_epochs_no_cat_covariates_contrastive_:
                self.cat_covariates_contrastive_active = False
            else:
                self.cat_covariates_contrastive_active = True

            self.multimodal_contrastive_weight_ = (
                self._get_multimodal_contrastive_weight())

            collect_source_umap_from_train = (
                self.epoch == self.n_epochs_ - 1
                and self.target_holdout_adata is not None
                and self.target_holdout_adata_atac is not None
            )
            self._source_umap_clip_rna_batches = []
            self._source_umap_clip_atac_batches = []
            self._source_umap_collected_n = 0
            self._source_umap_target_n = (
                int(self.target_holdout_adata.n_obs)
                if collect_source_umap_from_train
                else 0
            )

            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_val_iter"] = 0
            if not self.verbose_:
                # Avoid calling `.item()` every iteration (forces GPU sync).
                self.iter_logs["train_global_loss_sum"] = torch.zeros(
                    (), device=self.device
                )
                self.iter_logs["train_optim_loss_sum"] = torch.zeros(
                    (), device=self.device
                )
                self.iter_logs["train_multimodal_contrastive_loss_sum"] = torch.zeros(
                    (), device=self.device
                )
                self.iter_logs["train_multimodal_contrastive_loss_count"] = 0
                self.iter_logs["train_knowledge_distillation_loss_sum"] = torch.zeros(
                    (), device=self.device
                )
                self.iter_logs["train_knowledge_distillation_loss_count"] = 0

            for edge_train_data_batch, node_train_data_batch in zip(
                    self.edge_train_loader,
                    _cycle_iterable(self.node_train_loader)):

                # node-level model output
                node_train_data_batch = node_train_data_batch.to(self.device, non_blocking=True)

                node_train_model_output = self.model(
                    data_batch=node_train_data_batch,
                    decoder="omics",
                    use_only_active_gps=self.use_only_active_gps)

                if self.lambda_knowledge_distillation_ > 0:
                    with torch.no_grad():
                        teacher_node_train_model_output = self.teacher_model(
                            data_batch=node_train_data_batch,
                            decoder="omics",
                            use_only_active_gps=self.use_only_active_gps,
                        )
                    node_train_model_output["student_clip_rna"] = node_train_model_output.get(
                        "clip_rna"
                    )
                    node_train_model_output["student_clip_atac"] = node_train_model_output.get(
                        "clip_atac"
                    )
                    node_train_model_output["teacher_clip_rna"] = teacher_node_train_model_output.get(
                        "clip_rna"
                    )
                    node_train_model_output["teacher_clip_atac"] = teacher_node_train_model_output.get(
                        "clip_atac"
                    )

                if (
                    collect_source_umap_from_train
                    and self._source_umap_collected_n < self._source_umap_target_n
                ):
                    clip_rna_batch = node_train_model_output.get("clip_rna")
                    clip_atac_batch = node_train_model_output.get("clip_atac")

                    if clip_rna_batch is not None and clip_atac_batch is not None:
                        remaining = self._source_umap_target_n - self._source_umap_collected_n
                        take_n = min(
                            remaining,
                            int(clip_rna_batch.size(0)),
                            int(clip_atac_batch.size(0)),
                        )
                        if take_n > 0:
                            self._source_umap_clip_rna_batches.append(
                                clip_rna_batch[:take_n]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )
                            self._source_umap_clip_atac_batches.append(
                                clip_atac_batch[:take_n]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32, copy=False)
                            )
                            self._source_umap_collected_n += take_n

                # edge-level model output
                edge_train_data_batch = edge_train_data_batch.to(self.device, non_blocking=True)
                edge_train_model_output = self.model(
                    data_batch=edge_train_data_batch,
                    decoder="graph",
                    use_only_active_gps=self.use_only_active_gps)


                train_loss_dict = self.model.loss(
                    edge_model_output=edge_train_model_output,
                    node_model_output=node_train_model_output,
                    lambda_edge_recon=self.lambda_edge_recon_,
                    lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                    lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                    lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                    lambda_multimodal_contrastive_loss=self.multimodal_contrastive_weight_,
                    lambda_knowledge_distillation=self.lambda_knowledge_distillation_,
                    multimodal_temperature=self.multimodal_temperature_,
                    multimodal_contrastive_active=(
                        self.multimodal_contrastive_weight_ > 0),
                    contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                    contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
                    l1_targets_mask=self.l1_targets_mask,
                    l1_sources_mask=self.l1_sources_mask,
                    lambda_l1_addon=self.lambda_l1_addon_,
                    edge_recon_active=self.edge_recon_active,
                    cat_covariates_contrastive_active=self.cat_covariates_contrastive_active)

                train_global_loss = train_loss_dict["global_loss"]
                train_optim_loss = train_loss_dict["optim_loss"]

                if self.verbose_:
                    for key, value in train_loss_dict.items():
                        self.iter_logs[f"train_{key}"].append(value.item())
                else:
                    self.iter_logs["train_global_loss_sum"] += train_global_loss.detach()
                    self.iter_logs["train_optim_loss_sum"] += train_optim_loss.detach()
                    # Always log multimodal_contrastive_loss if present
                    if "multimodal_contrastive_loss" in train_loss_dict:
                        self.iter_logs["train_multimodal_contrastive_loss_sum"] += (
                            train_loss_dict["multimodal_contrastive_loss"].detach()
                        )
                        self.iter_logs["train_multimodal_contrastive_loss_count"] += 1
                    if "knowledge_distillation_loss" in train_loss_dict:
                        self.iter_logs["train_knowledge_distillation_loss_sum"] += (
                            train_loss_dict["knowledge_distillation_loss"].detach()
                        )
                        self.iter_logs["train_knowledge_distillation_loss_count"] += 1
                self.iter_logs["n_train_iter"] += 1

                self.optimizer.zero_grad()
                train_optim_loss.backward()
                if self.grad_clip_value_ > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                    self.grad_clip_value_)
                self.optimizer.step()

            if (self.edge_val_loader is not None and
                self.node_val_loader is not None):
                self.eval_epoch()
            elif (self.edge_val_loader is None and
            self.node_val_loader is not None):
                warnings.warn("You have specified a node validation set but no "
                              "edge validation set. Skipping validation...")
            elif (self.edge_val_loader is not None and
            self.node_val_loader is None):
                warnings.warn("You have specified an edge validation set but no"
                              " node validation set. Skipping validation...")

            if self.verbose_:
                for key in self.iter_logs:
                    if key.startswith("train"):
                        epoch_avg_loss = (
                            np.array(self.iter_logs[key]).sum() /
                            self.iter_logs["n_train_iter"])
                        self.epoch_logs[key].append(epoch_avg_loss)
                        # Log training losses to MLflow
                        if self.mlflow_experiment_id is not None:
                            mlflow.log_metric(key, epoch_avg_loss, step=self.epoch)
                    if key.startswith("val"):
                        epoch_avg_loss = (
                            np.array(self.iter_logs[key]).sum() /
                            self.iter_logs["n_val_iter"])
                        self.epoch_logs[key].append(epoch_avg_loss)
                        # Log validation losses to MLflow
                        if self.mlflow_experiment_id is not None:
                            mlflow.log_metric(key, epoch_avg_loss, step=self.epoch)
            else:
                n_train = max(1, int(self.iter_logs["n_train_iter"]))
                train_global_loss = (self.iter_logs["train_global_loss_sum"] / n_train).item()
                train_optim_loss = (self.iter_logs["train_optim_loss_sum"] / n_train).item()
                self.epoch_logs["train_global_loss"].append(train_global_loss)
                self.epoch_logs["train_optim_loss"].append(train_optim_loss)
                if self.mlflow_experiment_id is not None:
                    mlflow.log_metric("train_global_loss", train_global_loss, step=self.epoch)
                    mlflow.log_metric("train_optim_loss", train_optim_loss, step=self.epoch)

                mmc_count = int(self.iter_logs.get("train_multimodal_contrastive_loss_count", 0))
                if mmc_count > 0:
                    train_mmc_loss = (self.iter_logs["train_multimodal_contrastive_loss_sum"] / mmc_count).item()
                    self.epoch_logs["train_multimodal_contrastive_loss"].append(train_mmc_loss)
                    if self.mlflow_experiment_id is not None:
                        mlflow.log_metric("train_multimodal_contrastive_loss", train_mmc_loss, step=self.epoch)
                kd_count = int(self.iter_logs.get("train_knowledge_distillation_loss_count", 0))
                if kd_count > 0:
                    train_kd_loss = (
                        self.iter_logs["train_knowledge_distillation_loss_sum"] / kd_count
                    ).item()
                    self.epoch_logs["train_knowledge_distillation_loss"].append(train_kd_loss)
                    if self.mlflow_experiment_id is not None:
                        mlflow.log_metric(
                            "train_knowledge_distillation_loss",
                            train_kd_loss,
                            step=self.epoch,
                        )

                n_val = int(self.iter_logs.get("n_val_iter", 0))
                if n_val > 0 and "val_global_loss_sum" in self.iter_logs:
                    val_global_loss = (self.iter_logs["val_global_loss_sum"] / n_val).item()
                    val_optim_loss = (self.iter_logs["val_optim_loss_sum"] / n_val).item()
                    self.epoch_logs["val_global_loss"].append(val_global_loss)
                    self.epoch_logs["val_optim_loss"].append(val_optim_loss)
                    if self.mlflow_experiment_id is not None:
                        mlflow.log_metric("val_global_loss", val_global_loss, step=self.epoch)
                        mlflow.log_metric("val_optim_loss", val_optim_loss, step=self.epoch)

                    val_mmc_count = int(self.iter_logs.get("val_multimodal_contrastive_loss_count", 0))
                    if val_mmc_count > 0:
                        val_mmc_loss = (self.iter_logs["val_multimodal_contrastive_loss_sum"] / val_mmc_count).item()
                        self.epoch_logs["val_multimodal_contrastive_loss"].append(val_mmc_loss)
                        if self.mlflow_experiment_id is not None:
                            mlflow.log_metric("val_multimodal_contrastive_loss", val_mmc_loss, step=self.epoch)
                    val_kd_count = int(self.iter_logs.get("val_knowledge_distillation_loss_count", 0))
                    if val_kd_count > 0:
                        val_kd_loss = (
                            self.iter_logs["val_knowledge_distillation_loss_sum"] / val_kd_count
                        ).item()
                        self.epoch_logs["val_knowledge_distillation_loss"].append(val_kd_loss)
                        if self.mlflow_experiment_id is not None:
                            mlflow.log_metric(
                                "val_knowledge_distillation_loss",
                                val_kd_loss,
                                step=self.epoch,
                            )


            # Evaluate on target data
            if (self.epoch % target_eval_interval == 0) or (self.epoch == self.n_epochs_-1):
                target_holdout_clip_embeddings_rna = None
                target_holdout_clip_embeddings_atac = None

                # Fetch target embeddings once before any metrics are computed.
                if (self.log_target_multimodal_contrastive_  and self.target_holdout_adata is not None and self.target_holdout_adata_atac is not None):

                    target_holdout_clip_embeddings_rna, target_holdout_clip_embeddings_atac = (
                        self._get_target_latent_embeddings(
                            adata=self.target_holdout_adata,
                            adata_atac=self.target_holdout_adata_atac,
                            paired_data=self.target_paired_data_,
                            chunk_size=500,
                        )
                    )

                # Compute metrics after embeddings are available.
                if (
                    self.log_target_multimodal_contrastive_
                    and self.target_paired_data_
                    and target_holdout_clip_embeddings_rna is not None
                    and target_holdout_clip_embeddings_atac is not None
                ):

                    self._log_target_paired_metrics(
                            temperature=self.multimodal_temperature_,
                            clip_embeddings_rna=target_holdout_clip_embeddings_rna,
                            clip_embeddings_atac=target_holdout_clip_embeddings_atac,
                    )

                if (
                    self.target_adata is not None
                    and self.target_holdout_adata is not None
                    and target_holdout_clip_embeddings_rna is not None
                ):
                    self.target_holdout_adata.obsm[self.target_latent_key] = target_holdout_clip_embeddings_rna
                    if (
                        self.target_adata_atac is not None
                        and self.target_holdout_adata_atac is not None
                        and target_holdout_clip_embeddings_atac is not None
                    ):
                        self.target_holdout_adata_atac.obsm[self.target_latent_key] = target_holdout_clip_embeddings_atac

                    self._log_target_unpaired_metrics()

                ## create and log compound metric
                compound_metric = self.epoch_logs.copy()
                '''
                if "compound_metric" in compound_metric: # remove 'compound_metric' itself
                    compound_metric.pop("compound_metric")
                for key_ in [key for key in list(compound_metric.keys()) if 'loss' in key]: # remove all items with keys containing 'loss'
                    compound_metric.pop(key_)
                compound_metric = np.sum([values[-1] for values in compound_metric.values()]) / len(compound_metric)
                '''
                def _latest_finite_metric(metric_key: str) -> Optional[float]:
                    values = self.epoch_logs.get(metric_key, [])
                    if not values:
                        return None
                    value = float(values[-1])
                    if not np.isfinite(value):
                        return None
                    return value

                integration_metrics = [
                    _latest_finite_metric("target_one_minus_foscttm"),
                    _latest_finite_metric("target_iLISI"),
                ]
                integration_metrics = [
                    metric for metric in integration_metrics if metric is not None
                ]

                clustering_metrics = [
                    _latest_finite_metric("target_KMeans_ARI"),
                    _latest_finite_metric("target_KMeans_NMI"),
                    _latest_finite_metric("target_Silhouette_label"),
                ]
                clustering_metrics = [
                    metric for metric in clustering_metrics if metric is not None
                ]

                compound_components = []
                if integration_metrics:
                    compound_components.append(float(np.mean(integration_metrics)))
                if clustering_metrics:
                    compound_components.append(float(np.mean(clustering_metrics)))

                if compound_components:
                    compound_metric = float(np.mean(compound_components))
                    self.epoch_logs["compound_metric"].append(compound_metric)

                    if self.mlflow_experiment_id is not None:
                        mlflow.log_metric(
                            "compound_metric",
                            compound_metric,
                            step=self.epoch,
                        )
                else:
                    warnings.warn(
                        "Skipping compound_metric logging because no finite "
                        "target metrics are available for this epoch."
                    )

            if self.monitor_:
                print_progress(self.epoch, self.epoch_logs, self.n_epochs_)

            if self.use_early_stopping_:
                if self.is_early_stopping():
                    break

        self.training_time += (time.time() - start_time)
        minutes, seconds = divmod(self.training_time, 60)
        print(f"Model training finished after {int(minutes)} min {int(seconds)}"
              " sec.")
        if self.best_model_state_dict is not None and self.reload_best_model_:
            print("Using best model state, which was in epoch "
                  f"{self.best_epoch + 1}.")
            self.model.load_state_dict(self.best_model_state_dict)

        self.model.eval()

        if self.edge_val_loader is not None:
            self.eval_end()

    @torch.no_grad()
    def eval_epoch(self):
        """
        Epoch evaluation logic of CustomNicheCompass model used during training.
        """
        self.model.eval()

        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
        edge_same_cat_covariates_cat_val_accumulated = [
            np.array([]) for _ in range(self.n_cat_covariates)]
        edge_incl_val_accumulated = np.array([])

        multimodal_contrastive_weight = self._get_multimodal_contrastive_weight()
        if not self.verbose_:
            # Avoid per-iteration `.item()` calls (GPU sync).
            self.iter_logs["val_global_loss_sum"] = torch.zeros((), device=self.device)
            self.iter_logs["val_optim_loss_sum"] = torch.zeros((), device=self.device)
            self.iter_logs["val_multimodal_contrastive_loss_sum"] = torch.zeros(
                (), device=self.device
            )
            self.iter_logs["val_multimodal_contrastive_loss_count"] = 0
            self.iter_logs["val_knowledge_distillation_loss_sum"] = torch.zeros(
                (), device=self.device
            )
            self.iter_logs["val_knowledge_distillation_loss_count"] = 0

        for edge_val_data_batch, node_val_data_batch in zip(
                self.edge_val_loader, _cycle_iterable(self.node_val_loader)):
            node_val_data_batch = node_val_data_batch.to(self.device, non_blocking=True)
            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder="omics",
                use_only_active_gps=self.use_only_active_gps)
            if self.lambda_knowledge_distillation_ > 0:
                teacher_node_val_model_output = self.teacher_model(
                    data_batch=node_val_data_batch,
                    decoder="omics",
                    use_only_active_gps=self.use_only_active_gps,
                )
                node_val_model_output["student_clip_rna"] = node_val_model_output.get(
                    "clip_rna"
                )
                node_val_model_output["student_clip_atac"] = node_val_model_output.get(
                    "clip_atac"
                )
                node_val_model_output["teacher_clip_rna"] = teacher_node_val_model_output.get(
                    "clip_rna"
                )
                node_val_model_output["teacher_clip_atac"] = teacher_node_val_model_output.get(
                    "clip_atac"
                )

            edge_val_data_batch = edge_val_data_batch.to(self.device, non_blocking=True)
            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder="graph",
                use_only_active_gps=self.use_only_active_gps)

            val_loss_dict = self.model.loss(
                edge_model_output=edge_val_model_output,
                node_model_output=node_val_model_output,
                lambda_edge_recon=self.lambda_edge_recon_,
                lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                lambda_multimodal_contrastive_loss=multimodal_contrastive_weight,
                lambda_knowledge_distillation=self.lambda_knowledge_distillation_,
                multimodal_temperature=self.multimodal_temperature_,
                multimodal_contrastive_active=(multimodal_contrastive_weight > 0),
                contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                lambda_group_lasso=self.lambda_group_lasso_,
                lambda_l1_masked=self.lambda_l1_masked_,
                l1_targets_mask=self.l1_targets_mask,
                l1_sources_mask=self.l1_sources_mask,
                lambda_l1_addon=self.lambda_l1_addon_,
                edge_recon_active=True)

            val_global_loss = val_loss_dict["global_loss"]
            val_optim_loss = val_loss_dict["optim_loss"]
            if self.verbose_:
                for key, value in val_loss_dict.items():
                    self.iter_logs[f"val_{key}"].append(value.item())
            else:
                self.iter_logs["val_global_loss_sum"] += val_global_loss.detach()
                self.iter_logs["val_optim_loss_sum"] += val_optim_loss.detach()
                # Always log multimodal_contrastive_loss if present
                if "multimodal_contrastive_loss" in val_loss_dict:
                    self.iter_logs["val_multimodal_contrastive_loss_sum"] += (
                        val_loss_dict["multimodal_contrastive_loss"].detach()
                    )
                    self.iter_logs["val_multimodal_contrastive_loss_count"] += 1
                if "knowledge_distillation_loss" in val_loss_dict:
                    self.iter_logs["val_knowledge_distillation_loss_sum"] += (
                        val_loss_dict["knowledge_distillation_loss"].detach()
                    )
                    self.iter_logs["val_knowledge_distillation_loss_count"] += 1
            self.iter_logs["n_val_iter"] += 1

            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_same_cat_covariates_cat_val = (
                edge_val_model_output["edge_same_cat_covariates_cat"])
            edge_incl_val = edge_val_model_output["edge_incl"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
            if edge_same_cat_covariates_cat_val is not None:
                for i, edge_same_cat_covariate_cat_val in enumerate(
                        edge_same_cat_covariates_cat_val):
                    edge_same_cat_covariates_cat_val_accumulated[i] = np.append(
                        edge_same_cat_covariates_cat_val_accumulated[i],
                        edge_same_cat_covariate_cat_val.detach().cpu().numpy())
            if edge_incl_val is not None:
                edge_incl_val_accumulated = np.append(
                    edge_incl_val_accumulated,
                    edge_incl_val.detach().cpu().numpy())
            else:
                edge_same_cat_covariates_cat_val_accumulated = None
                edge_incl_val_accumulated = None
        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            edge_same_cat_covariates_cat=edge_same_cat_covariates_cat_val_accumulated,
            edge_incl=edge_incl_val_accumulated)
        if self.verbose_:
            self.epoch_logs["val_auroc_score"].append(
                val_eval_dict["auroc_score"])
            self.epoch_logs["val_auprc_score"].append(
                val_eval_dict["auprc_score"])
            self.epoch_logs["val_best_acc_score"].append(
                val_eval_dict["best_acc_score"])
            self.epoch_logs["val_best_f1_score"].append(
                val_eval_dict["best_f1_score"])

        # Log evaluation metrics to MLflow during training
        if self.mlflow_experiment_id is not None:
            for key, value in val_eval_dict.items():
                mlflow.log_metric(f"val_iters_{key}", value, step=self.epoch)

        self.model.train()

class EncoderWithFCHidden(Encoder):
    """
    Encoder that also returns the post-FC hidden features.

    This mirrors `nichecompass.nn.encoders.Encoder.forward` but additionally
    returns `hidden_fc`, the output after the dense fully-connected block (and
    optional categorical covariate injection) and before any graph layers.
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        cat_covariates_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if ((self.cat_covariates_embed_mode == "input") &
            (cat_covariates_embed is not None)):
            # Add categorical covariates embedding to input vector
            x = torch.cat((x, cat_covariates_embed), dim=1)

        # FC forward pass shared across all nodes
        hidden = self.dropout(self.activation(self.fc_l1(x)))
        if self.n_fc_layers == 2:
            hidden = self.dropout(self.activation(self.fc_l2(hidden)))
            hidden = self.fc_l2_bn(hidden)

        if ((self.cat_covariates_embed_mode == "hidden") &
            (cat_covariates_embed is not None)):
            # Add categorical covariates embedding to hidden vector
            hidden = torch.cat((hidden, cat_covariates_embed), dim=1)

        # Tap the post-FC representation (before any graph layers)
        hidden_fc = hidden.clone()

        if self.n_layers == 2:
            # Part of forward pass shared across all nodes
            hidden = self.dropout(self.activation(self.conv_l1(hidden, edge_index)))

        # Part of forward pass only for maskable latent nodes
        mu = self.conv_mu(hidden, edge_index)
        logstd = self.conv_logstd(hidden, edge_index)

        # Part of forward pass only for unmaskable add-on latent nodes
        if self.n_addon_latent != 0:
            mu = torch.cat((mu, self.addon_conv_mu(hidden, edge_index)), dim=1)
            logstd = torch.cat((logstd, self.addon_conv_logstd(hidden, edge_index)), dim=1)

        return mu, logstd, hidden_fc


class CustomVGPGAE(VGPGAE):
    """
    Project-specific VGPGAE with custom behavior.

    Override methods as needed; this is a safe default that calls the parent
    implementation while giving you a single place to modify logic.
    """
    def __init__(self, *args, **kwargs):
        hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
            multimodal_embedding_size=kwargs.get("multimodal_embedding_size"),
            multimodal_layer_series=kwargs.get("multimodal_layer_series"),
        )
        kwargs["multimodal_embedding_size"] = hparam_kwargs[
            "multimodal_embedding_size"
        ]
        kwargs["multimodal_layer_series"] = hparam_kwargs[
            "multimodal_layer_series"
        ]

        # Remove 'multimodal_embedding_size' from kwargs before passing to super().__init__
        kwargs_no_mme = dict(kwargs)
        kwargs_no_mme.pop("multimodal_embedding_size", None)
        kwargs_no_mme.pop("multimodal_layer_series", None)
        super().__init__(*args, **kwargs_no_mme)

        ## remove encoder created by parent class
        if hasattr(self, "encoder"):
            del self.encoder

        self.multimodal_embedding_size_ = kwargs.get("multimodal_embedding_size")
        self.multimodal_layer_series_ = bool(kwargs.get("multimodal_layer_series", False))

        n_cat_covariates_embed_input = (
            sum(self.cat_covariates_embeds_nums_)
            if ("encoder" in self.cat_covariates_embeds_injection_) &
            (self.n_cat_covariates_ > 0)
            else 0
        )

        # Separate encoders for RNA and ATAC inputs when available.
        self.encoder_rna = EncoderWithFCHidden(
            n_input=self.n_output_genes_,
            n_cat_covariates_embed_input=n_cat_covariates_embed_input,
            n_fc_layers=self.n_fc_layers_encoder_,
            n_layers=self.n_layers_encoder_,
            n_hidden=self.n_hidden_encoder_,
            n_latent=self.n_prior_gp_,
            n_addon_latent=self.n_addon_gp_,
            conv_layer=self.conv_layer_encoder_,
            n_attention_heads=self.encoder_n_attention_heads_,
            dropout_rate=self.dropout_rate_encoder_,
            activation=torch.relu,
            use_bn=self.encoder_use_bn_)

        if self.n_output_peaks_ > 0:
            self.encoder_atac = EncoderWithFCHidden(
                n_input=self.n_output_peaks_,
                n_cat_covariates_embed_input=n_cat_covariates_embed_input,
                n_fc_layers=self.n_fc_layers_encoder_,
                n_layers=self.n_layers_encoder_,
                n_hidden=self.n_hidden_encoder_,
                n_latent=self.n_prior_gp_,
                n_addon_latent=self.n_addon_gp_,
                conv_layer=self.conv_layer_encoder_,
                n_attention_heads=self.encoder_n_attention_heads_,
                dropout_rate=self.dropout_rate_encoder_,
                activation=torch.relu,
                use_bn=self.encoder_use_bn_)
        else:
            self.encoder_atac = None

        def _disable_add_self_loops(encoder: Optional[torch.nn.Module]) -> None:
            if encoder is None:
                return
            for attr in ("conv_l1", "conv_mu", "conv_logstd",
                         "addon_conv_mu", "addon_conv_logstd"):
                conv = getattr(encoder, attr, None)
                if conv is not None and hasattr(conv, "add_self_loops"):
                    conv.add_self_loops = False

        _disable_add_self_loops(self.encoder_rna)
        _disable_add_self_loops(self.encoder_atac)

        # Multimodal layer
        gp_embedding_size = self.n_prior_gp_ + self.n_addon_gp_
        # If no explicit multimodal embedding size is provided, default to the
        # full GP embedding size (i.e. keep dimensionality). This avoids
        # constructing layers with out_features=None.
        if self.multimodal_embedding_size_ is None:
            self.multimodal_embedding_size_ = gp_embedding_size
        if self.multimodal_layer_series_:
            if self.multimodal_embedding_size_ == gp_embedding_size:
                self.multimodal_encoder = torch.nn.Identity()
                self.multimodal_decoder = torch.nn.Identity()
                self.multimodal_layer = torch.nn.Identity()
            else:
                self.multimodal_encoder = torch.nn.Sequential(
                    torch.nn.Linear(
                        gp_embedding_size,
                        self.multimodal_embedding_size_),
                    torch.nn.BatchNorm1d(self.multimodal_embedding_size_)
                )
                self.multimodal_decoder = torch.nn.Linear(
                    self.multimodal_embedding_size_,
                    gp_embedding_size)
                self.multimodal_layer = torch.nn.Sequential(
                    self.multimodal_encoder,
                    self.multimodal_decoder)
        else:
            self.multimodal_encoder = None
            self.multimodal_decoder = None
            self.multimodal_layer = torch.nn.Sequential(
                torch.nn.Linear(
                    gp_embedding_size,
                    self.multimodal_embedding_size_),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(
                    self.multimodal_embedding_size_,
                    self.multimodal_embedding_size_),
                torch.nn.BatchNorm1d(self.multimodal_embedding_size_)
            )

    @torch.no_grad()
    def get_active_gp_mask(
            self,
            abs_gp_weights_agg_mode: Literal["sum",
                                             "nzmeans",
                                             "sum+nzmeans",
                                             "nzmedians",
                                             "sum+nzmedians"]="sum+nzmeans",
            return_gp_weights: bool=False,
            normalize_gp_weights_with_features_scale_factors: bool=False,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a mask of active gene programs based on the rna decoder gene weights
        of gene programs. Active gene programs are gene programs whose absolute
        gene weights aggregated over all genes are greater than
        ´self.active_gp_thresh_ratio_´ times the absolute gene weights
        aggregation of the gene program with the maximum value across all gene
        programs. Depending on ´abs_gp_weights_agg_mode´, the aggregation will
        be either a sum of absolute gene weights (prioritizes gene programs that
        reconstruct many genes) or a mean of non-zero absolute gene weights
        (normalizes for the number of genes that a gene program reconstructs) or
        a combination of the two.

        Parameters
        ----------
        abs_gp_weights_agg_mode:
            If ´sum´, uses sums of absolute gp weights for aggregation and
            active gp determination. If ´nzmeans´, uses means of non-zero
            absolute gp weights for aggregation and active gp determination. If
            ´sum+nzmeans´, uses a combination of sums and means of non-zero
            absolute gp weights for aggregation and active gp determination.
        return_gp_weights:
            If ´True´, in addition return the rna decoder gene weights of the
            active gene programs.

        Returns
        ----------
        active_gp_mask:
            Boolean tensor of gene programs which contains `True` for active
            gene programs and `False` for inactive gene programs.
        active_gp_weights:
            Tensor containing the rna decoder gene weights of active gene
            programs.
        """
        device = next(self.parameters()).device

        active_gp_mask = torch.zeros(self.n_prior_gp_ + self.n_addon_gp_,
                                     dtype=torch.bool,
                                     device=device)

        if self.active_gp_type_ == "mixed":
            gp_types = ["all"]
        elif (self.n_addon_gp_ > 0):
            gp_types = ["masked", "addon"]
        else:
            gp_types = ["masked"]

        for gp_type in gp_types:
            gp_weights = self.get_gp_weights(only_masked_features=False,
                                             gp_type=gp_type)[0]

            # Get index of gps based on ´gp_type´
            if gp_type == "masked":
                gp_idx = slice(None, self.n_prior_gp_)
            elif gp_type == "addon":
                gp_idx = slice(self.n_prior_gp_, None)
            elif gp_type == "all":
                gp_idx = slice(None, None)

            # Normalize gp weights with features scale factors
            if normalize_gp_weights_with_features_scale_factors:
                gp_weights_normalized = (gp_weights /
                                         self.features_scale_factors_[:, None].to(device))
            else:
                gp_weights_normalized = gp_weights

            # Normalize gp weights with running mean absolute gp scores
            gp_weights_normalized = (self.running_mean_abs_mu[gp_idx] *
                                     gp_weights_normalized)

            # Aggregate absolute normalized gp weights based on
            # ´abs_gp_weights_agg_mode´ and calculate thresholds of aggregated
            # absolute normalized gp weights and get active gp mask and (optionally)
            # active gp weights
            abs_gp_weights_sums = gp_weights_normalized.norm(p=1, dim=0)
            if abs_gp_weights_agg_mode in ["sum", "sum+nzmeans", "sum+nzmedians"]:
                max_abs_gp_weights_sum = abs_gp_weights_sums.amax()
                min_abs_gp_weights_sum_thresh = (self.active_gp_thresh_ratio_ *
                                                max_abs_gp_weights_sum)
                active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                    abs_gp_weights_sums >= min_abs_gp_weights_sum_thresh)

            if abs_gp_weights_agg_mode in ["nzmeans", "sum+nzmeans"]:
                abs_gp_weights_nzmeans = (
                    abs_gp_weights_sums /
                    torch.count_nonzero(gp_weights_normalized, dim=0))
                abs_gp_weights_nzmeans = torch.nan_to_num(abs_gp_weights_nzmeans)
                max_abs_gp_weights_nzmean = abs_gp_weights_nzmeans.amax()
                min_abs_gp_weights_nzmean_thresh = (self.active_gp_thresh_ratio_ *
                                                    max_abs_gp_weights_nzmean)
                if abs_gp_weights_agg_mode == "nzmeans":
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmeans >=
                        min_abs_gp_weights_nzmean_thresh)
                elif abs_gp_weights_agg_mode == "sum+nzmeans":
                    # Combine active gp mask
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmeans >=
                        min_abs_gp_weights_nzmean_thresh)
            if abs_gp_weights_agg_mode in ["nzmedians", "sum+nzmedians"]:
                zero_mask = (gp_weights_normalized == 0)
                abs_gp_weights_normalized_with_nan = torch.where(
                    zero_mask,
                    torch.tensor(float("nan")),
                    torch.abs(gp_weights_normalized),
                )
                abs_gp_weights_nzmedians = torch.nanmedian(
                    abs_gp_weights_normalized_with_nan, dim=0).values
                abs_gp_weights_nzmedians = torch.nan_to_num(abs_gp_weights_nzmedians)
                max_abs_gp_weights_nzmedian = torch.max(abs_gp_weights_nzmedians)
                min_abs_gp_weights_nzmedian_thresh = (0.01 *
                                                      max_abs_gp_weights_nzmedian)
                if abs_gp_weights_agg_mode == "nzmedians":
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmedians >=
                        min_abs_gp_weights_nzmedian_thresh)
                elif abs_gp_weights_agg_mode == "sum+nzmedians":
                    # Combine active gp mask
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmedians >=
                        min_abs_gp_weights_nzmedian_thresh)
        if return_gp_weights:
            active_gp_weights = gp_weights[:, active_gp_mask]
            return active_gp_mask, active_gp_weights
        else:
            return active_gp_mask


    def multiply_gaussians_log_space(self,
                                     mu_rna: torch.Tensor,
                                     logstd_rna: torch.Tensor,
                                     mu_atac: torch.Tensor,
                                     logstd_atac: torch.Tensor) -> tuple:
        """
        Combines two Gaussians using log-standard-deviations.
        s1, s2: natural log of the standard deviation
        """
        # Calculate variances in log-space for stability
        # var = exp(2*s)
        v1 = torch.exp(2 * logstd_rna)
        v2 = torch.exp(2 * logstd_atac)
        denom = v1 + v2
        
        # New Mean
        mu_new = (mu_rna * v2 + mu_atac * v1) / denom
        
        # New Log-Std: s_new = s1 + s2 - 0.5 * ln(exp(2*s1) + exp(2*s2))
        # We use np.logaddexp for numerical stability
        s_new = logstd_rna + logstd_atac - 0.5 * torch.logaddexp(
            2 * logstd_rna, 2 * logstd_atac)
        
        return mu_new, s_new

    def _combine_posteriors(self,
                            mu_rna: torch.Tensor,
                            logstd_rna: torch.Tensor,
                            mu_atac: torch.Tensor,
                            logstd_atac: torch.Tensor,
                            modality_mask: Optional[torch.Tensor]=None
                            ) -> tuple:
        if modality_mask is None:
            return self.multiply_gaussians_log_space(
                mu_rna, logstd_rna, mu_atac, logstd_atac)

        if modality_mask.dtype != torch.bool:
            modality_mask = modality_mask.to(torch.bool)
        has_rna = modality_mask[:, 0]
        has_atac = modality_mask[:, 1]
        both = has_rna & has_atac
        only_rna = has_rna & ~has_atac
        only_atac = ~has_rna & has_atac

        # Avoid Python control flow on GPU tensors (`if both.any():`), which can
        # force expensive device synchronization. Use purely tensorized logic.
        mu_both, logstd_both = self.multiply_gaussians_log_space(
            mu_rna, logstd_rna, mu_atac, logstd_atac)

        mu = torch.zeros_like(mu_rna)
        logstd = torch.zeros_like(logstd_rna)

        mu = torch.where(both[:, None], mu_both, mu)
        logstd = torch.where(both[:, None], logstd_both, logstd)
        mu = torch.where(only_rna[:, None], mu_rna, mu)
        logstd = torch.where(only_rna[:, None], logstd_rna, logstd)
        mu = torch.where(only_atac[:, None], mu_atac, mu)
        logstd = torch.where(only_atac[:, None], logstd_atac, logstd)

        return mu, logstd

    def forward(self,
                data_batch: Data,
                decoder: Literal["graph", "omics"],
                use_only_active_gps: bool=False,
                return_agg_weights: bool=False,
                update_atac_dynamic_decoder_mask: bool=False) -> dict:
        """
        Forward pass of the VGPGAE module.

        Parameters
        ----------
        data_batch:
            PyG Data object containing either an edge-level batch if 
            ´decoder == graph´ or a node-level batch if ´decoder == omics´.
        decoder:
            Decoder to use for the forward pass. Either ´graph´ for edge
            reconstruction or ´omics´ for gene expression and (if specified)
            chromatin accessibility reconstruction.
        use_only_active_gps:
            If ´True´, use only active gene programs as input to decoder.
        return_agg_weights:
            If ´True´, also return the aggregation weights of the node label
            aggregator.
        update_atac_dynamic_decoder_mask:
            If ´True´, turn off the mapped peaks for genes that have been
            turned off in a gene program (set peak gp weights to 0).

        Returns
        ----------
        output:
            Dictionary containing reconstructed edge logits if
            ´decoder == graph´ or the parameters of the omics feature
            distributions if ´decoder == omics´, as well as ´mu´ and ´logstd´ 
            from the latent space distribution.
        """
        x_input = data_batch.x # dim: n_obs x n_omics_features
        x_counts = getattr(data_batch, "x_counts", x_input)
        edge_index = data_batch.edge_index # dim: 2 x n_edges (incl. all edges
                                           # of sampled graph)
        
        # Get index of sampled nodes for current batch (neighbors of sampled
        # nodes are also part of the batch for message passing layers but
        # should be excluded in backpropagation)
        if decoder == "omics":
            # ´data_batch´ will be a node batch and first node_batch_size
            # elements are the sampled nodes, leading to a dim of ´batch_idx´ of
            # ´node_batch_size´
            batch_idx = slice(None, data_batch.batch_size)
        elif decoder == "graph":
            # ´data_batch´ will be an edge batch with sampled positive and
            # negative edges of size ´edge_batch_size´ respectively. Each edge
            # has a source and target node, leading to a dim of ´batch_idx´ of
            # 4 * ´edge_batch_size´
            batch_idx = torch.cat((data_batch.edge_label_index[0],
                                   data_batch.edge_label_index[1]), 0)

        # Logarithmitize omics feature vector (only) for encoder input for
        # numerical stability. This will not affect node labels.
        if self.log_variational_:
            x_enc = torch.log(1 + x_input)
        else:
            x_enc = x_input
            
        # Get categorical covariates embedding
        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                     data_batch.cat_covariates_cats[:, i]))
                self.cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            self.cat_covariates_embed = None         

        if self.encoder_atac is None:
            return super().forward(
                data_batch=data_batch,
                decoder=decoder,
                use_only_active_gps=use_only_active_gps,
                return_agg_weights=return_agg_weights,
                update_atac_dynamic_decoder_mask=update_atac_dynamic_decoder_mask,
            )

        output = {}

        # Separate rna and atac data
        x_enc_rna = x_enc[:, :self.n_output_genes_]
        x_enc_atac = x_enc[:, self.n_output_genes_:]
        
        # Use encoder to get latent distribution parameters for current batch
        # and reparameterization trick to get latent features (gp scores).
        # Filter for nodes in current batch

        # Encode rna data
        encoder_outputs_rna = self.encoder_rna(
            x=x_enc_rna,
            edge_index=edge_index,
            cat_covariates_embed=(self.cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        self.mu_rna = encoder_outputs_rna[0][batch_idx, :]
        self.logstd_rna = encoder_outputs_rna[1][batch_idx, :]
        output["mu_rna"] = self.mu_rna
        output["logstd_rna"] = self.logstd_rna

        # Send a copy of the post-FC encoder representation through multimodal_layer
        hidden_fc_rna = encoder_outputs_rna[2][batch_idx, :]

        # Encode atac data
        encoder_outputs_atac = self.encoder_atac(
            x=x_enc_atac,
            edge_index=edge_index,
            cat_covariates_embed=(self.cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        self.mu_atac = encoder_outputs_atac[0][batch_idx, :]
        self.logstd_atac = encoder_outputs_atac[1][batch_idx, :]
        output["mu_atac"] = self.mu_atac
        output["logstd_atac"] = self.logstd_atac

        # Send a copy of the post-FC encoder representation through multimodal_layer
        hidden_fc_atac = encoder_outputs_atac[2][batch_idx, :]

        # Get multimodal embeddings and store in output
        if self.multimodal_layer_series_:
            output["clip_rna"] = self.multimodal_encoder(hidden_fc_rna)
            output["clip_atac"] = self.multimodal_encoder(hidden_fc_atac)
        else:
            output["clip_rna"] = self.multimodal_layer(hidden_fc_rna)
            output["clip_atac"] = self.multimodal_layer(hidden_fc_atac)

        # Get modality mask
        modality_mask = getattr(data_batch, "modality_mask", None)
        if modality_mask is not None:
            modality_mask = modality_mask[batch_idx]

        # Combine rna and atac latent distributions using product of Gaussians
        self.mu, self.logstd = self._combine_posteriors(
            self.mu_rna,
            self.logstd_rna,
            self.mu_atac,
            self.logstd_atac,
            modality_mask=modality_mask)
        output["mu"] = self.mu
        output["logstd"] = self.logstd
        z = self.reparameterize(self.mu, self.logstd)

        if use_only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            
            # Set gp scores of inactive gene programs to 0 to not affect 
            # graph decoder
            z[:, ~active_gp_mask] = 0                
        if self.multimodal_layer_series_:
            z = self.multimodal_layer(z)

        if decoder == "omics":
            with torch.no_grad():
                if self.training:
                    # Update running mean absolute gp scores using exponential
                    # moving average with momentum of 0.1
                    mean_abs_mu = self.mu.norm(p=1, dim=0) / self.mu.size(0)
                    self.running_mean_abs_mu = (
                        0.1 * mean_abs_mu + 0.9 * self.running_mean_abs_mu)
                    
                if use_only_active_gps:
                    # Set running mean abs mu of inactive gene programs to 0 for
                    # active gp determination
                    self.running_mean_abs_mu[~active_gp_mask] = 0  

                    # Set dynamic mask to 0 for all inactive gene programs to
                    # not affect omics decoders
                    self.target_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0
                    self.source_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0

                    if "atac" in self.modalities_:
                        self.target_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0
                        self.source_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0
                    
            # Determine which features should be reconstructed based on
            # static and dynamic masks (if a feature is not connected to any
            # node it should not be reconstructed to not influence softmax
            # activation outputs). This can happen when no add-on gene programs
            # are present or when gene programs are turned off.
            if self.n_addon_gp_ > 0:
                target_rna_decoder_static_mask = torch.cat(
                    (self.target_rna_decoder_mask,
                     self.target_rna_decoder_addon_mask), dim=0)
                source_rna_decoder_static_mask = torch.cat(
                    (self.source_rna_decoder_mask,
                     self.source_rna_decoder_addon_mask), dim=0)
            else:
                target_rna_decoder_static_mask = self.target_rna_decoder_mask
                source_rna_decoder_static_mask = self.source_rna_decoder_mask

            self.target_n_gps_per_gene = (
                target_rna_decoder_static_mask
                * self.target_rna_dynamic_decoder_mask
                ).sum(0)
            self.features_idx_dict_["target_reconstructed_rna_idx"] = (
                torch.nonzero(self.target_n_gps_per_gene)).flatten().tolist()

            self.source_n_gps_per_gene = (
                source_rna_decoder_static_mask
                * self.source_rna_dynamic_decoder_mask
                ).sum(0)
            self.features_idx_dict_["source_reconstructed_rna_idx"] = (
                torch.nonzero(self.source_n_gps_per_gene)).flatten().tolist()

            self.target_rna_theta_reconstructed = self.target_rna_theta[
                self.features_idx_dict_["target_reconstructed_rna_idx"]]
            self.source_rna_theta_reconstructed = self.source_rna_theta[
                self.features_idx_dict_["source_reconstructed_rna_idx"]]
                    
            output["node_labels"] = {}

            # Get rna and atac part from omics feature vector (counts targets)
            x_counts_atac = x_counts[:, self.n_output_genes_:]
            x_counts_rna = x_counts[:, :self.n_output_genes_]
        
            # Compute aggregated neighborhood rna feature vector
            rna_node_label_aggregator_output = self.rna_node_label_aggregator(
                    x=x_counts_rna,
                    edge_index=edge_index,
                    return_agg_weights=return_agg_weights)
            x_neighbors = rna_node_label_aggregator_output[0]
 
            # Retrieve rna node labels and only keep nodes in current node batch
            # and reconstructed features
            assert x_counts_rna.size(1) == self.n_output_genes_
            assert x_neighbors.size(1) == self.n_output_genes_
            output["node_labels"]["target_rna"] = x_counts_rna[batch_idx][
                :, self.features_idx_dict_["target_reconstructed_rna_idx"]]
            output["node_labels"]["source_rna"] = x_neighbors[batch_idx][
                :, self.features_idx_dict_["source_reconstructed_rna_idx"]]
            
            # Use observed library size as scaling factor for the negative
            # binomial means of the rna distribution
            target_rna_library_size = output["node_labels"]["target_rna"].sum(
                1).unsqueeze(1)
            source_rna_library_size = output["node_labels"]["source_rna"].sum(
                1).unsqueeze(1)
            self.target_rna_log_library_size = torch.log(target_rna_library_size)
            self.source_rna_log_library_size = torch.log(source_rna_library_size)

            # Get gene expression reconstruction distribution parameters for
            # reconstructed genes
            output["target_rna_nb_means"] = self.target_rna_decoder(
                z=z,
                log_library_size=self.target_rna_log_library_size,
                cat_covariates_embed=(
                    self.cat_covariates_embed[batch_idx] if
                    (self.cat_covariates_embed is not None) &
                    ("gene_expr_decoder" in
                     self.cat_covariates_embeds_injection_)
                     else None))[
                    :, self.features_idx_dict_["target_reconstructed_rna_idx"]]
            output["source_rna_nb_means"] = self.source_rna_decoder(
                z=z,
                log_library_size=self.source_rna_log_library_size,
                cat_covariates_embed=(
                self.cat_covariates_embed[batch_idx] if
                (self.cat_covariates_embed is not None) &
                ("gene_expr_decoder" in
                 self.cat_covariates_embeds_injection_)
                 else None))[
                    :, self.features_idx_dict_["source_reconstructed_rna_idx"]]
            
            if "atac" in self.modalities_:
                # Determine which features should be reconstructed based on
                # masks (if a feature is not connected to any node it should not
                # be reconstructed to not influence softmax activation outputs)
                if self.n_addon_gp_ > 0:
                    target_atac_decoder_static_mask = torch.cat(
                        (self.target_atac_decoder_mask,
                         self.target_atac_decoder_addon_mask), dim=0)
                    source_atac_decoder_static_mask = torch.cat(
                        (self.source_atac_decoder_mask,
                         self.source_atac_decoder_addon_mask), dim=0)
                else:
                    target_atac_decoder_static_mask = self.target_atac_decoder_mask
                    source_atac_decoder_static_mask = self.source_atac_decoder_mask

                self.target_n_gps_per_peak = (
                    target_atac_decoder_static_mask
                    * self.target_atac_dynamic_decoder_mask
                    ).sum(0)
                self.features_idx_dict_["target_reconstructed_atac_idx"] = (
                    torch.nonzero(self.target_n_gps_per_peak)).flatten().tolist()

                self.source_n_gps_per_peak = (
                    source_atac_decoder_static_mask
                    * self.source_atac_dynamic_decoder_mask
                    ).sum(0)
                self.features_idx_dict_["source_reconstructed_atac_idx"] = (
                    torch.nonzero(self.source_n_gps_per_peak)).flatten().tolist()

                self.target_atac_theta_reconstructed = self.target_atac_theta[
                    self.features_idx_dict_["target_reconstructed_atac_idx"]]
                self.source_atac_theta_reconstructed = self.source_atac_theta[
                    self.features_idx_dict_["source_reconstructed_atac_idx"]]

                # Compute aggregated neighborhood atac feature vector
                atac_node_label_aggregator_output = (
                    self.atac_node_label_aggregator(
                        x=x_counts_atac,
                        edge_index=edge_index,
                        return_agg_weights=return_agg_weights))
                x_neighbors_atac = atac_node_label_aggregator_output[0]

                # Retrieve node labels and only keep nodes in current node batch
                # and reconstructed features
                assert x_counts_atac.size(1) == self.n_output_peaks_
                assert x_neighbors_atac.size(1) == self.n_output_peaks_
                output["node_labels"]["target_atac"] = x_counts_atac[batch_idx][
                    :, self.features_idx_dict_["target_reconstructed_atac_idx"]]  
                output["node_labels"]["source_atac"] = x_neighbors_atac[batch_idx][
                    :, self.features_idx_dict_["source_reconstructed_atac_idx"]]

                # Use observed library size as scaling factor for the negative
                # binomial means of the atac distribution
                target_atac_library_size = output["node_labels"][
                    "target_atac"].sum(1).unsqueeze(1)
                source_atac_library_size = output["node_labels"][
                    "source_atac"].sum(1).unsqueeze(1)
                self.target_atac_log_library_size = torch.log(
                    target_atac_library_size)
                self.source_atac_log_library_size = torch.log(
                    source_atac_library_size)
                
                if update_atac_dynamic_decoder_mask:
                    # Get atac dynamic decoder masks to turn off peaks that
                    # are mapped to only genes that are turned off
                    with torch.no_grad():
                        # Retrieve rna decoder gp weights
                        gp_weights = self.get_gp_weights(
                            only_masked_features=False)[0].detach().cpu()
                        
                        # Round to 4 decimals as genes are never completely
                        # turned off due to L1 being not differentiable at 0
                        gp_weights = torch.round(gp_weights, decimals=4)

                        # Get boolean mask of non zero target and source gene 
                        # weights
                        non_zero_gene_weights = torch.ne(
                                gp_weights, 
                                0) # dim: (2 x n_genes, n_gps)
                        non_zero_target_gene_weights = non_zero_gene_weights[
                            :self.n_output_genes_, :] # dim: (n_genes, n_gps)
                        non_zero_source_gene_weights = non_zero_gene_weights[
                            self.n_output_genes_:, :] # dim: (n_genes, n_gps)
                        
                        # Multiply boolean mask with gene peak mapping to remove
                        # peaks that are mapped to only turned off genes
                        target_atac_dynamic_decoder_mask = torch.mm(
                            non_zero_target_gene_weights.t().to(torch.float32), # dim: (n_gps,
                                                              #       n_genes)
                            self.gene_peaks_mask_.to(torch.float32)).to(torch.bool) # dim: (n_genes,
                                                   # n_peaks)
                            # dim: (n_gps, n_peaks)
                        source_atac_dynamic_decoder_mask = torch.mm(
                            non_zero_source_gene_weights.t().to(torch.float32),
                            self.gene_peaks_mask_.to(torch.float32)).to(torch.bool)
                        
                        # Create boolean mask of peaks (until here multiple
                        # active genes in a gp can be mapped to the same peak,
                        # resulting in values > 1.)
                        self.target_atac_dynamic_decoder_mask = (
                            self.target_atac_dynamic_decoder_mask & torch.ne(
                            target_atac_dynamic_decoder_mask, 
                            0)) # dim: (n_gps, n_peaks)
                        self.source_atac_dynamic_decoder_mask = (
                            self.source_atac_dynamic_decoder_mask & torch.ne(
                            source_atac_dynamic_decoder_mask, 
                            0))
                    
                # Get chromatin accessibility reconstruction distribution
                # parameters for reconstructed peaks
                output["target_atac_nb_means"] = self.target_atac_decoder(
                    z=z,
                    log_library_size=self.target_atac_log_library_size,
                    dynamic_mask=self.target_atac_dynamic_decoder_mask,
                    cat_covariates_embed=(
                        self.cat_covariates_embed[batch_idx] if
                        (self.cat_covariates_embed is not None) & 
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) else
                        None))[
                    :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
                output["source_atac_nb_means"] = self.source_atac_decoder(
                    z=z,
                    log_library_size=self.source_atac_log_library_size,
                    dynamic_mask=self.source_atac_dynamic_decoder_mask,
                    cat_covariates_embed=(
                        self.cat_covariates_embed[batch_idx] if
                        (self.cat_covariates_embed is not None) &
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) else
                        None))[
                    :, self.features_idx_dict_["source_reconstructed_atac_idx"]]
        elif decoder == "graph":
            # Store edge labels in output for loss computation
            output["edge_recon_labels"] = data_batch.edge_label
                
            # For each categorical covariate, create a boolean tensor to
            # indicate for each sampled edge (negative & positive edges) whether
            # the edge nodes have the same category and store in a list
            if len(self.cat_covariates_cats_) > 0:
                output["edge_same_cat_covariates_cat"] = []
                for cat_covariate_idx in range(len(self.cat_covariates_cats_)):
                    edge_same_cat_covariate_cat = (
                        data_batch.cat_covariates_cats[
                            data_batch.edge_label_index[0],
                            cat_covariate_idx] ==
                        data_batch.cat_covariates_cats[
                            data_batch.edge_label_index[1],
                            cat_covariate_idx])
                    output["edge_same_cat_covariates_cat"].append(
                        edge_same_cat_covariate_cat)
                
                # Based on the categorical covariate and its possibility for
                # edges to exist for different categories (this might only be
                # the case for certain categorical covariates, others might only
                # use disconnected neighbor graphs for different categories),
                # create a boolean mask for edges whether they should be
                # included in the edge reconstruction loss and edge
                # reconstruction performance evaluation
                cat_covariates_cat_edge_incl = []
                for cat_covariate_no_edge, edge_same_cat_covariate_cat in zip(
                    self.cat_covariates_no_edges_,
                    output["edge_same_cat_covariates_cat"]):
                    if not cat_covariate_no_edge:
                        cat_covariates_cat_edge_incl.append(
                            torch.ones_like(edge_same_cat_covariate_cat,
                                            dtype=torch.bool))
                    else:
                        cat_covariates_cat_edge_incl.append(
                            edge_same_cat_covariate_cat)
                output["edge_incl"] = torch.all(
                    torch.stack(cat_covariates_cat_edge_incl),
                                dim=0)
            else:
                output["edge_same_cat_covariates_cat"] = None
                output["edge_incl"] = None

            # Use decoder to get the edge reconstruction logits
            output["edge_recon_logits"] = self.graph_decoder(z=z)
        return output


    def get_latent_representation(
            self,
            node_batch: Data,
            only_active_gps: bool=True,
            return_mu_std: bool=False,
            separate_modalities: bool=False,
            return_clip_embeddings: bool=False,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Encode RNA + ATAC separately and combine latents via Gaussian product.
        
        Parameters
        ----------
        node_batch : Data
            Batch of nodes to encode.
        only_active_gps : bool
            Whether to return only active gene programs.
        return_mu_std : bool
            If True, returns (mu, std) instead of sampled z.
        separate_modalities : bool
            If True, returns separate RNA and ATAC latents instead of combined.
            When True and return_mu_std=True, returns (mu_rna, std_rna, mu_atac, std_atac).
            When True and return_mu_std=False, returns (z_rna, z_atac).
        return_clip_embeddings : bool
            If True (and `separate_modalities=True` and `return_mu_std=True`),
            also returns the outputs of forwarding the RNA/ATAC means through
            `self.multimodal_layer` as `(clip_embeddings_rna, clip_embeddings_atac)`.
            Note: `clip_embeddings` are only valid when
            `separate_modalities=True` and `return_mu_std=True`.
        """
        if return_clip_embeddings and (not separate_modalities):
            raise ValueError(
                "return_clip_embeddings=True requires separate_modalities=True "
                "because clip_embeddings are modality-specific.")
        if return_clip_embeddings and (not return_mu_std):
            raise ValueError(
                "return_clip_embeddings=True requires return_mu_std=True "
                "because clip_embeddings are computed from modality means (mu).")
        if self.encoder_atac is None:
            if separate_modalities:
                raise ValueError(
                    "separate_modalities=True requires a multimodal model with "
                    "separate RNA and ATAC encoders.")
            return super().get_latent_representation(
                node_batch=node_batch,
                only_active_gps=only_active_gps,
                return_mu_std=return_mu_std)

        if self.log_variational_:
            x_enc = torch.log(1 + node_batch.x)
        else:
            x_enc = node_batch.x

        x_enc_rna = x_enc[:, :self.n_output_genes_]
        x_enc_atac = x_enc[:, self.n_output_genes_:]

        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                    node_batch.cat_covariates_cats[:, i]))
                cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            cat_covariates_embed = None

        # RNA encoder
        encoder_outputs_rna = self.encoder_rna(
            x=x_enc_rna,
            edge_index=node_batch.edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_rna = encoder_outputs_rna[0][:node_batch.batch_size, :]
        logstd_rna = encoder_outputs_rna[1][:node_batch.batch_size, :]

        # ATAC encoder
        encoder_outputs_atac = self.encoder_atac(
            x=x_enc_atac,
            edge_index=node_batch.edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_atac = encoder_outputs_atac[0][:node_batch.batch_size, :]
        logstd_atac = encoder_outputs_atac[1][:node_batch.batch_size, :]

        # Send a copy of the post-FC encoder representation through multimodal_layer
        hidden_fc_rna = encoder_outputs_rna[2][:node_batch.batch_size, :]
        hidden_fc_atac = encoder_outputs_atac[2][:node_batch.batch_size, :]

        # Get multimodal embeddings and store in output
        if self.multimodal_layer_series_:
            clip_embeddings_rna = self.multimodal_encoder(hidden_fc_rna)
            clip_embeddings_atac = self.multimodal_encoder(hidden_fc_atac)
        else:
            clip_embeddings_rna = self.multimodal_layer(hidden_fc_rna)
            clip_embeddings_atac = self.multimodal_layer(hidden_fc_atac)

        if only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            mu_rna = mu_rna[:, active_gp_mask]
            logstd_rna = logstd_rna[:, active_gp_mask]
            mu_atac = mu_atac[:, active_gp_mask]
            logstd_atac = logstd_atac[:, active_gp_mask]
        else:
            active_gp_mask = None

        if separate_modalities:
            if return_clip_embeddings:
                if active_gp_mask is None:
                    clip_embeddings_rna = clip_embeddings_rna[:node_batch.batch_size, :]
                    clip_embeddings_atac = clip_embeddings_atac[:node_batch.batch_size, :]
                else:
                    # Project only-active GPs by masking the Linear weights.
                    # This avoids a dimension mismatch while ensuring inactive
                    # GPs do not contribute to the clip_embeddings.
                    if self.multimodal_layer_series_ and isinstance(
                            self.multimodal_encoder, torch.nn.Identity):
                        clip_embeddings_rna = hidden_fc_rna
                        clip_embeddings_atac = hidden_fc_atac
                    else:
                        raise NotImplementedError("Not implemented for active GP masking")
                        if self.multimodal_layer_series_:
                            w = self.multimodal_encoder.weight  # (out, in_full)
                            b = self.multimodal_encoder.bias
                        else:
                            w = self.multimodal_layer.weight  # (out, in_full)
                            b = self.multimodal_layer.bias
                        clip_embeddings_rna = F.linear(hidden_fc_rna, w[:, active_gp_mask], b)
                        clip_embeddings_atac = F.linear(hidden_fc_atac, w[:, active_gp_mask], b)

            if return_mu_std:
                std_rna = torch.exp(logstd_rna)
                std_atac = torch.exp(logstd_atac)
                if return_clip_embeddings:
                    return mu_rna, std_rna, mu_atac, std_atac, clip_embeddings_rna, clip_embeddings_atac
                return mu_rna, std_rna, mu_atac, std_atac
            else:
                z_rna = self.reparameterize(mu_rna, logstd_rna)
                z_atac = self.reparameterize(mu_atac, logstd_atac)
                if self.multimodal_layer_series_:
                    z_rna = self.multimodal_layer(z_rna)
                    z_atac = self.multimodal_layer(z_atac)
                return z_rna, z_atac

        # Combine modalities (original behavior)
        modality_mask = getattr(node_batch, "modality_mask", None)
        if modality_mask is not None:
            modality_mask = modality_mask[:node_batch.batch_size]

        mu, logstd = self._combine_posteriors(
            mu_rna,
            logstd_rna,
            mu_atac,
            logstd_atac,
            modality_mask=modality_mask)

        if return_mu_std:
            std = torch.exp(logstd)
            return mu, std
        z = self.reparameterize(mu, logstd)

        return z


    def get_omics_decoder_outputs(
            self,
            node_batch: Data,
            only_active_gps: bool=True,
            ) -> dict:
        """
        Decode using a combined RNA+ATAC latent (product-of-Gaussians).

        Mirrors `nichecompass.modules.vgpgae.VGPGAE.get_omics_decoder_outputs`,
        but uses `encoder_rna` / `encoder_atac` and combines their posteriors
        before decoding.
        """
        if self.encoder_atac is None:
            return super().get_omics_decoder_outputs(
                node_batch=node_batch,
                only_active_gps=only_active_gps)

        x_input = node_batch.x  # dim: n_obs x n_omics_features
        x_counts = getattr(node_batch, "x_counts", x_input)
        edge_index = node_batch.edge_index
        batch_idx = slice(None, node_batch.batch_size)

        # Logarithmitize omics feature vector if done during training
        if self.log_variational_:
            x_enc = torch.log(1 + x_input)
        else:
            x_enc = x_input

        # Get categorical covariate embeddings
        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                    node_batch.cat_covariates_cats[:, i]))
                cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            cat_covariates_embed = None

        # Split encoder inputs by modality
        x_enc_rna = x_enc[:, :self.n_output_genes_]
        x_enc_atac = x_enc[:, self.n_output_genes_:]

        # Encode RNA + ATAC and combine posteriors
        encoder_outputs_rna = self.encoder_rna(
            x=x_enc_rna,
            edge_index=edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_rna = encoder_outputs_rna[0][batch_idx, :]
        logstd_rna = encoder_outputs_rna[1][batch_idx, :]

        encoder_outputs_atac = self.encoder_atac(
            x=x_enc_atac,
            edge_index=edge_index,
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu_atac = encoder_outputs_atac[0][batch_idx, :]
        logstd_atac = encoder_outputs_atac[1][batch_idx, :]

        modality_mask = getattr(node_batch, "modality_mask", None)
        if modality_mask is not None:
            modality_mask = modality_mask[batch_idx]
        mu, logstd = self._combine_posteriors(
            mu_rna,
            logstd_rna,
            mu_atac,
            logstd_atac,
            modality_mask=modality_mask)
        z = self.reparameterize(mu, logstd)

        if only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            # Set gp scores of inactive gene programs to 0 to not affect decoders
            z[:, ~active_gp_mask] = 0
        if self.multimodal_layer_series_:
            z = self.multimodal_layer(z)

        output = {}
        output["node_labels"] = {}

        # Get rna and atac part from omics feature vector (counts targets)
        x_counts_atac = x_counts[:, self.n_output_genes_:]
        x_counts_rna = x_counts[:, :self.n_output_genes_]

        # Compute aggregated neighborhood rna feature vector
        rna_node_label_aggregator_output = self.rna_node_label_aggregator(
                x=x_counts_rna,
                edge_index=edge_index,
                return_agg_weights=False)
        x_neighbors = rna_node_label_aggregator_output[0]

        # Retrieve rna node labels and only keep nodes in current node batch
        assert x_counts_rna.size(1) == self.n_output_genes_
        assert x_neighbors.size(1) == self.n_output_genes_
        output["node_labels"]["target_rna"] = x_counts_rna[batch_idx]
        output["node_labels"]["source_rna"] = x_neighbors[batch_idx]

        # Use observed library size as scaling factor for the NB means
        target_rna_library_size = output["node_labels"]["target_rna"].sum(
            1).unsqueeze(1)
        source_rna_library_size = output["node_labels"]["source_rna"].sum(
            1).unsqueeze(1)
        target_rna_log_library_size = torch.log(target_rna_library_size)
        source_rna_log_library_size = torch.log(source_rna_library_size)

        output["target_rna_nb_means"] = self.target_rna_decoder(
            z=z,
            log_library_size=target_rna_log_library_size,
            cat_covariates_embed=(
                cat_covariates_embed[batch_idx] if
                (cat_covariates_embed is not None) &
                ("gene_expr_decoder" in
                 self.cat_covariates_embeds_injection_)
                 else None))
        output["source_rna_nb_means"] = self.source_rna_decoder(
            z=z,
            log_library_size=source_rna_log_library_size,
            cat_covariates_embed=(
            cat_covariates_embed[batch_idx] if
            (cat_covariates_embed is not None) &
            ("gene_expr_decoder" in
             self.cat_covariates_embeds_injection_)
             else None))

        if "atac" in self.modalities_:
            # Compute aggregated neighborhood atac feature vector
            atac_node_label_aggregator_output = (
                self.atac_node_label_aggregator(
                    x=x_counts_atac,
                    edge_index=edge_index,
                    return_agg_weights=False))
            x_neighbors_atac = atac_node_label_aggregator_output[0]

            # Retrieve node labels and only keep nodes in current node batch
            assert x_counts_atac.size(1) == self.n_output_peaks_
            assert x_neighbors_atac.size(1) == self.n_output_peaks_
            output["node_labels"]["target_atac"] = x_counts_atac[batch_idx][
                :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
            output["node_labels"]["source_atac"] = x_neighbors_atac[batch_idx][
                :, self.features_idx_dict_["source_reconstructed_atac_idx"]]

            # Use observed library size as scaling factor for the NB means
            target_atac_library_size = output["node_labels"][
                "target_atac"].sum(1).unsqueeze(1)
            source_atac_library_size = output["node_labels"][
                "source_atac"].sum(1).unsqueeze(1)
            target_atac_log_library_size = torch.log(target_atac_library_size)
            source_atac_log_library_size = torch.log(source_atac_library_size)

            output["target_atac_nb_means"] = self.target_atac_decoder(
                z=z,
                log_library_size=target_atac_log_library_size,
                dynamic_mask=self.target_atac_dynamic_decoder_mask,
                cat_covariates_embed=(
                    cat_covariates_embed[batch_idx] if
                    (cat_covariates_embed is not None) &
                    ("chrom_access_decoder" in
                     self.cat_covariates_embeds_injection_) else
                    None))[
                :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
            output["source_atac_nb_means"] = self.source_atac_decoder(
                z=z,
                log_library_size=source_atac_log_library_size,
                dynamic_mask=self.source_atac_dynamic_decoder_mask,
                cat_covariates_embed=(
                    cat_covariates_embed[batch_idx] if
                    (cat_covariates_embed is not None) &
                    ("chrom_access_decoder" in
                     self.cat_covariates_embeds_injection_) else
                    None))[
                :, self.features_idx_dict_["source_reconstructed_atac_idx"]]
        return output

    def loss(self,
             edge_model_output: dict,
             node_model_output: dict,
             lambda_l1_masked: float,
             l1_targets_mask: torch.Tensor,
             l1_sources_mask: torch.Tensor,
             lambda_l1_addon: float,
             lambda_group_lasso: float,
             lambda_gene_expr_recon: float=None,
             lambda_chrom_access_recon: float=None,
             lambda_edge_recon: Optional[float]=None,
             lambda_cat_covariates_contrastive: Optional[float]=None,
             contrastive_logits_pos_ratio: Optional[float]=None,
             contrastive_logits_neg_ratio: Optional[float]=None,
             edge_recon_active: bool=True,
             cat_covariates_contrastive_active: bool=True,
             lambda_multimodal_contrastive_loss: Optional[float]=None,
             multimodal_temperature: Optional[float]=None,
             multimodal_contrastive_active: bool=True,
             lambda_knowledge_distillation: Optional[float]=None) -> dict:

        hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_edge_recon=lambda_edge_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            lambda_multimodal_contrastive_loss=lambda_multimodal_contrastive_loss,
            multimodal_temperature=multimodal_temperature,
            lambda_knowledge_distillation=lambda_knowledge_distillation,
        )
        lambda_gene_expr_recon = hparam_kwargs["lambda_gene_expr_recon"]
        lambda_chrom_access_recon = hparam_kwargs["lambda_chrom_access_recon"]
        lambda_edge_recon = hparam_kwargs["lambda_edge_recon"]
        lambda_cat_covariates_contrastive = hparam_kwargs[
            "lambda_cat_covariates_contrastive"
        ]
        contrastive_logits_pos_ratio = hparam_kwargs[
            "contrastive_logits_pos_ratio"
        ]
        contrastive_logits_neg_ratio = hparam_kwargs[
            "contrastive_logits_neg_ratio"
        ]
        lambda_multimodal_contrastive_loss = hparam_kwargs[
            "lambda_multimodal_contrastive_loss"
        ]
        multimodal_temperature = hparam_kwargs["multimodal_temperature"]
        lambda_knowledge_distillation = hparam_kwargs["lambda_knowledge_distillation"]

        loss_dict = super().loss(
            edge_model_output=edge_model_output,
            node_model_output=node_model_output,
            lambda_l1_masked=lambda_l1_masked,
            l1_targets_mask=l1_targets_mask,
            l1_sources_mask=l1_sources_mask,
            lambda_l1_addon=lambda_l1_addon,
            lambda_group_lasso=lambda_group_lasso,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_edge_recon=lambda_edge_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            edge_recon_active=edge_recon_active,
            cat_covariates_contrastive_active=cat_covariates_contrastive_active)

        if ("atac" in self.modalities_ and
                lambda_multimodal_contrastive_loss > 0 and
                multimodal_contrastive_active and
                "mu_rna" in node_model_output and
                "mu_atac" in node_model_output):

            similarity_matrix = self.get_multimodal_similarity(
                clip_embeddings_rna=node_model_output["clip_rna"],
                clip_embeddings_atac=node_model_output["clip_atac"])

            loss_dict["multimodal_contrastive_loss"] = (
                lambda_multimodal_contrastive_loss *
                self.compute_multimodal_contrastive_loss(
                    similarity_matrix,
                    temperature=multimodal_temperature))

            loss_dict["global_loss"] += loss_dict[
                "multimodal_contrastive_loss"]
            loss_dict["optim_loss"] += loss_dict[
                "multimodal_contrastive_loss"]

        if (
                lambda_knowledge_distillation > 0
                and "student_clip_rna" in node_model_output
                and "student_clip_atac" in node_model_output
                and "teacher_clip_rna" in node_model_output
                and "teacher_clip_atac" in node_model_output):

            student_similarity_matrix = self.get_multimodal_similarity(
                clip_embeddings_rna=node_model_output["student_clip_rna"],
                clip_embeddings_atac=node_model_output["student_clip_atac"])
            teacher_similarity_matrix = self.get_multimodal_similarity(
                clip_embeddings_rna=node_model_output["teacher_clip_rna"],
                clip_embeddings_atac=node_model_output["teacher_clip_atac"])

            soft_KD_loss = (
                lambda_knowledge_distillation *
                self.compute_knowledge_distillation_loss(
                    student_similarity_matrix,
                    teacher_similarity_matrix))

            ot_clip_loss = (
                lambda_knowledge_distillation *
                self.compute_ot_clip_loss(
                    student_similarity_matrix,
                    teacher_similarity_matrix))

            lambd = 0.1
            eclare_loss = lambd * soft_KD_loss + (1 - lambd) * ot_clip_loss

            loss_dict["global_loss"] += eclare_loss
            loss_dict["optim_loss"] += eclare_loss

        return loss_dict

    def get_multimodal_similarity(
            self,
            clip_embeddings_rna: torch.Tensor,
            clip_embeddings_atac: torch.Tensor,
            ) -> torch.Tensor:
        """
        Compute cross-modal similarity using *precomputed* CLIP embeddings.

        Note
        ----
        This method intentionally does **not** project `mu_*` through
        `multimodal_encoder` / `multimodal_layer`. Callers must provide the
        projected embeddings (e.g. from `forward()` output or from
        `get_latent_representation(..., return_clip_embeddings=True)`).
        """

        clip_rna_normed = F.normalize(clip_embeddings_rna, p=2, dim=1)
        clip_atac_normed = F.normalize(clip_embeddings_atac, p=2, dim=1)
        return torch.matmul(clip_rna_normed, clip_atac_normed.t())

    def compute_multimodal_contrastive_loss(
            self,
            similarity_matrix: torch.Tensor,
            temperature: Optional[float]=None) -> torch.Tensor:
        hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
            multimodal_temperature=temperature,
        )
        temperature = hparam_kwargs["multimodal_temperature"]
        temperature = max(temperature, 1e-8)
        logits = similarity_matrix / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_rna = F.cross_entropy(logits, labels)
        loss_atac = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_rna + loss_atac)

    def compute_knowledge_distillation_loss(
            self,
            student_similarity_matrix: torch.Tensor,
            teacher_similarity_matrix: torch.Tensor) -> torch.Tensor:

        kd_cols = F.kl_div(
            F.log_softmax(student_similarity_matrix, dim=1),
            F.log_softmax(teacher_similarity_matrix, dim=1),
            reduction="batchmean",
            log_target=True)

        kd_rows = F.kl_div(
            F.log_softmax(student_similarity_matrix, dim=0),
            F.log_softmax(teacher_similarity_matrix, dim=0),
            reduction="batchmean",
            log_target=True)

        return 0.5 * (kd_cols.mean() + kd_rows.mean())

    def compute_ot_clip_loss(
            self,
            student_similarity_matrix: torch.Tensor,
            teacher_similarity_matrix: torch.Tensor) -> torch.Tensor:

            multimodal_temperature = torch.tensor(1.)
            teacher_cost = 1 - (teacher_similarity_matrix / torch.exp(1/multimodal_temperature))

            plan = ot_solve(teacher_cost).plan
            plan_T = ot_solve(teacher_cost.T).plan  # empirically, plan_T != plan.T

            labels = torch.argmax(plan, dim=1)
            labels_T = torch.argmax(plan_T, dim=1)  # plan_T != plan.T

            ot_clip_loss = torch.nn.functional.cross_entropy(student_similarity_matrix, labels, reduction='none')
            ot_clip_loss_T = torch.nn.functional.cross_entropy(student_similarity_matrix.T, labels_T, reduction='none')

            return 0.5 * (ot_clip_loss.mean() + ot_clip_loss_T.mean())


class CustomSpatialAnnTorchDataset(SpatialAnnTorchDataset):
    """
    Project-specific dataset wrapper for NicheCompass.

    This is the place to customize how RNA + ATAC are read/concatenated into the
    node feature matrix `self.x`, or how covariates / graph are prepared.

    By default this simply calls the upstream implementation.
    """

    def __init__(self,
                 adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]=None,
                 encoder_input_key: Optional[str]=None,
                 adj_key: str=None,
                 edge_label_adj_key: str="edge_label_spatial_connectivities",
                 self_loops: bool=True,
                 cat_covariates_keys: Optional[List[str]]=None,
                 paired_data: bool=None):
        hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
            counts_key=counts_key,
            encoder_input_key=encoder_input_key,
            adj_key=adj_key,
            paired_data=paired_data,
        )
        counts_key = hparam_kwargs["counts_key"]
        encoder_input_key = hparam_kwargs["encoder_input_key"]
        adj_key = hparam_kwargs["adj_key"]
        paired_data = hparam_kwargs["paired_data"]
        input_key = encoder_input_key if encoder_input_key is not None else counts_key

        if input_key is None:
            x_rna_input = adata.X
        else:
            x_rna_input = adata.layers[input_key]

        if counts_key is None:
            x_rna_counts = adata.X
        else:
            x_rna_counts = adata.layers[counts_key]

        # Store features in dense format
        if sp.issparse(x_rna_input):
            self.x_rna = torch.tensor(x_rna_input.toarray())
        else:
            self.x_rna = torch.tensor(x_rna_input)

        if sp.issparse(x_rna_counts):
            self.x_rna_counts = torch.tensor(x_rna_counts.toarray())
        else:
            self.x_rna_counts = torch.tensor(x_rna_counts)

        # Store ATAC features in dense format if provided
        if adata_atac is not None:
            if input_key is None:
                x_atac_input = adata_atac.X
            else:
                x_atac_input = adata_atac.layers[input_key]

            if counts_key is None:
                x_atac_counts = adata_atac.X
            else:
                x_atac_counts = adata_atac.layers[counts_key]

            if paired_data:
                if ((adata.n_obs != adata_atac.n_obs) or
                    (not adata.obs_names.equals(adata_atac.obs_names))):
                    raise ValueError(
                        "adata and adata_atac must have matching obs_names in "
                        "the same order when paired_data=True. Set "
                        "paired_data=False for unpaired forward passes.")

                if sp.issparse(x_atac_input):
                    self.x_atac = torch.tensor(x_atac_input.toarray())
                else:
                    self.x_atac = torch.tensor(x_atac_input)

                if sp.issparse(x_atac_counts):
                    self.x_atac_counts = torch.tensor(x_atac_counts.toarray())
                else:
                    self.x_atac_counts = torch.tensor(x_atac_counts)

                self.x = torch.cat((self.x_rna, self.x_atac), axis=1)
                self.x_counts = torch.cat((self.x_rna_counts, self.x_atac_counts), axis=1)
                # For paired data, every observation has both modalities, so we
                # can omit the modality mask and take the fast path in
                # posterior-combining logic.
                self.modality_mask = None
            else:
                rna_obs = adata.obs_names
                atac_obs = adata_atac.obs_names
                extra_atac_obs = atac_obs[~atac_obs.isin(rna_obs)]
                union_obs = rna_obs.append(extra_atac_obs)
                n_obs_total = len(union_obs)

                rna_positions = np.arange(adata.n_obs)
                atac_positions = union_obs.get_indexer(atac_obs)
                if (atac_positions < 0).any():
                    raise ValueError("Unpaired union index is missing ATAC obs.")

                x_rna_full = torch.zeros(
                    (n_obs_total, adata.n_vars),
                    dtype=self.x_rna.dtype)
                x_rna_full[rna_positions] = self.x_rna

                if sp.issparse(x_atac_input):
                    x_atac = torch.tensor(x_atac_input.toarray(),
                                          dtype=self.x_rna.dtype)
                else:
                    x_atac = torch.tensor(x_atac_input,
                                          dtype=self.x_rna.dtype)
                x_atac_full = torch.zeros(
                    (n_obs_total, adata_atac.n_vars),
                    dtype=self.x_rna.dtype)
                x_atac_full[atac_positions] = x_atac

                x_rna_counts_full = torch.zeros(
                    (n_obs_total, adata.n_vars),
                    dtype=self.x_rna_counts.dtype)
                x_rna_counts_full[rna_positions] = self.x_rna_counts

                if sp.issparse(x_atac_counts):
                    x_atac_counts = torch.tensor(x_atac_counts.toarray(),
                                                 dtype=self.x_rna_counts.dtype)
                else:
                    x_atac_counts = torch.tensor(x_atac_counts,
                                                 dtype=self.x_rna_counts.dtype)
                x_atac_counts_full = torch.zeros(
                    (n_obs_total, adata_atac.n_vars),
                    dtype=self.x_rna_counts.dtype)
                x_atac_counts_full[atac_positions] = x_atac_counts

                self.x_rna = x_rna_full
                self.x_atac = x_atac_full
                self.x = torch.cat((self.x_rna, self.x_atac), axis=1)
                self.x_counts = torch.cat(
                    (x_rna_counts_full, x_atac_counts_full), axis=1)

                modality_mask = torch.zeros((n_obs_total, 2), dtype=torch.bool)
                modality_mask[rna_positions, 0] = True
                modality_mask[atac_positions, 1] = True
                self.modality_mask = modality_mask
        else:
            self.x_atac = None
            self.x = self.x_rna
            self.x_counts = self.x_rna_counts
            self.modality_mask = torch.stack(
                (torch.ones(self.x.size(0), dtype=torch.bool),
                 torch.zeros(self.x.size(0), dtype=torch.bool)),
                dim=1)

        # Store adjacency matrix in torch_sparse SparseTensor format
        if paired_data or adata_atac is None:
            adj_rna = (adata.obsp[adj_key] if sp.issparse(adata.obsp[adj_key])
                       else sp.csr_matrix(adata.obsp[adj_key]))
            self.adj = sparse_mx_to_sparse_tensor(adj_rna)
        else:
            adj_rna = (adata.obsp[adj_key] if sp.issparse(adata.obsp[adj_key])
                       else sp.csr_matrix(adata.obsp[adj_key]))
            if adj_key not in adata_atac.obsp:
                raise ValueError("Please specify an adequate 'adj_key' for "
                                 "adata_atac when paired_data=False.")
            atac_obs = adata_atac.obs_names
            atac_only_mask = ~atac_obs.isin(adata.obs_names)
            if atac_only_mask.any():
                atac_only_idx = np.where(atac_only_mask)[0]
                adj_atac_full = (adata_atac.obsp[adj_key] if sp.issparse(
                    adata_atac.obsp[adj_key]) else sp.csr_matrix(
                        adata_atac.obsp[adj_key]))
                adj_atac = adj_atac_full[atac_only_idx][:, atac_only_idx]
                adj = sp.block_diag((adj_rna, adj_atac), format="csr")
            else:
                adj = adj_rna
            self.adj = sparse_mx_to_sparse_tensor(adj)
            
        # Store edge label adjacency matrix
        if edge_label_adj_key in adata.obsp:
            self.edge_label_adj = sp.csr_matrix(adata.obsp[edge_label_adj_key])
        else:
            self.edge_label_adj = None

        # Validate adjacency matrix symmetry
        if (self.adj.nnz() != self.adj.t().nnz()):
            raise ImportError("The input adjacency matrix has to be symmetric.")
        
        self.edge_index = self.adj.to_torch_sparse_coo_tensor()._indices()

        if self_loops:
            # Add self loops to account for autocrine communication
            # Remove self loops in case there are already before adding new ones
            self.edge_index, _ = remove_self_loops(self.edge_index)
            self.edge_index, _ = add_self_loops(self.edge_index,
                                                num_nodes=self.x.size(0))
            
        if cat_covariates_keys is not None:
            self.cat_covariates_cats = []
            for cat_covariate_key, cat_covariate_label_encoder in zip(
                cat_covariates_keys,
                cat_covariates_label_encoders):
                if paired_data or adata_atac is None:
                    cat_covariate_cats = torch.tensor(
                        encode_labels(adata,
                                      cat_covariate_label_encoder,
                                      cat_covariate_key),
                        dtype=torch.long)
                else:
                    cat_covariate_cats = -1 * np.ones(
                        self.x.size(0), dtype=np.int64)
                    rna_encoded = encode_labels(
                        adata, cat_covariate_label_encoder, cat_covariate_key)
                    rna_encoded = rna_encoded.astype(np.int64)
                    cat_covariate_cats[:adata.n_obs] = rna_encoded
                    atac_obs = adata_atac.obs_names
                    atac_only_mask = ~atac_obs.isin(adata.obs_names)
                    if atac_only_mask.any():
                        atac_encoded = encode_labels(
                            adata_atac,
                            cat_covariate_label_encoder,
                            cat_covariate_key).astype(np.int64)
                        extra_positions = np.arange(
                            adata.n_obs, self.x.size(0))
                        cat_covariate_cats[extra_positions] = (
                            atac_encoded[atac_only_mask])
                    cat_covariate_cats = torch.tensor(
                        cat_covariate_cats, dtype=torch.long)
                self.cat_covariates_cats.append(cat_covariate_cats)
            self.cat_covariates_cats = torch.stack(self.cat_covariates_cats,
                                                   dim=1)            

        self.n_node_features = self.x.size(1)
        self.size_factors = self.x_counts.sum(1) # fix for ATAC case

    def __len__(self):
        """Return the number of observations stored in SpatialAnnTorchDataset"""
        return self.x.size(0)


def prepare_data(adata: AnnData,
                 cat_covariates_label_encoders: List[dict],
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]=None,
                 encoder_input_key: Optional[str]=None,
                 adj_key: str=None,
                 cat_covariates_keys: Optional[List[str]]=None,
                 paired_data: bool=None,
                 edge_val_ratio: float=None,
                 edge_test_ratio: float=0.,
                 node_val_ratio: float=None,
                 node_test_ratio: float=0.) -> dict:
    """
    Project-specific prepare_data function imitating nichecompass.data.dataprocessors.prepare_data().
    """
    data_dict = {}
    hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
        counts_key=counts_key,
        encoder_input_key=encoder_input_key,
        adj_key=adj_key,
        paired_data=paired_data,
        edge_val_ratio=edge_val_ratio,
        node_val_ratio=node_val_ratio,
    )
    counts_key = hparam_kwargs["counts_key"]
    encoder_input_key = hparam_kwargs["encoder_input_key"]
    adj_key = hparam_kwargs["adj_key"]
    paired_data = hparam_kwargs["paired_data"]
    edge_val_ratio = hparam_kwargs["edge_val_ratio"]
    node_val_ratio = hparam_kwargs["node_val_ratio"]

    dataset = CustomSpatialAnnTorchDataset(
        adata=adata,
        adata_atac=adata_atac,
        counts_key=counts_key,
        encoder_input_key=encoder_input_key,
        adj_key=adj_key,
        cat_covariates_keys=cat_covariates_keys,
        cat_covariates_label_encoders=cat_covariates_label_encoders,
        paired_data=paired_data)

    # PyG Data object (has 2 edge index pairs for one edge because of symmetry;
    # one edge index pair will be removed in the edge-level split).
    data = Data(
        x=dataset.x,
        edge_index=dataset.edge_index,
        edge_attr=dataset.edge_index.t()) # store index of edge nodes as
                                          # edge attribute for
                                          # aggregation weight retrieval
                                          # in mini batches

    # Keep per-modality views for debugging / custom logic.
    data.x_rna = dataset.x_rna
    if dataset.x_atac is not None:
        data.x_atac = dataset.x_atac
    data.x_counts = dataset.x_counts

    if dataset.modality_mask is not None:
        data.modality_mask = dataset.modality_mask

    if cat_covariates_keys is not None:
        data.cat_covariates_cats = dataset.cat_covariates_cats

    # Edge-level split for edge reconstruction
    edge_train_data, edge_val_data, edge_test_data = edge_level_split(
        data=data,
        edge_label_adj=dataset.edge_label_adj,
        val_ratio=edge_val_ratio,
        test_ratio=edge_test_ratio)
    data_dict["edge_train_data"] = edge_train_data
    data_dict["edge_val_data"] = edge_val_data
    data_dict["edge_test_data"] = edge_test_data

    # Node-level split for gene expression reconstruction
    data_dict["node_masked_data"] = node_level_split_mask(
        data=data,
        val_ratio=node_val_ratio,
        test_ratio=node_test_ratio)
    return data_dict
    

def _get_module_out_features(module: torch.nn.Module) -> Optional[int]:
    """
    Extract output feature size from a module (e.g. Sequential or Linear).

    For nn.Sequential, returns the out_features of the last nn.Linear.
    For nn.Linear, returns out_features. For nn.Identity, returns None.
    """
    if isinstance(module, torch.nn.Linear):
        return module.out_features
    if isinstance(module, torch.nn.Sequential):
        for layer in reversed(module):
            if isinstance(layer, torch.nn.Linear):
                return layer.out_features
    return None


def initialize_dataloaders(node_masked_data: Data,
                           edge_train_data: Optional[Data]=None,
                           edge_val_data: Optional[Data]=None,
                           edge_batch_size: Optional[int]=None,
                           node_batch_size: Optional[int]=None,
                           n_direct_neighbors: int=-1,
                           n_hops: int=1,
                           shuffle: bool=True,
                           edges_directed: bool=False,
                           neg_edge_sampling_ratio: float=1.,
                           pin_memory: bool=False,
                           num_workers: int=0,
                           persistent_workers: bool=False,
                           prefetch_factor: int=2) -> dict:
    """
    Initialize edge-level and node-level training and validation dataloaders.

    Parameters
    ----------
    node_masked_data:
        PyG Data object with node-level split masks.
    edge_train_data:
        PyG Data object containing the edge-level training set.
    edge_val_data:
        PyG Data object containing the edge-level validation set.
    edge_batch_size:
        Batch size for the edge-level dataloaders.
    node_batch_size:
        Batch size for the node-level dataloaders.
    n_direct_neighbors:
        Number of sampled direct neighbors of the current batch nodes to be 
        included in the batch. Defaults to ´-1´, which means to include all 
        direct neighbors.
    n_hops:
        Number of neighbor hops / levels for neighbor sampling of nodes to be 
        included in the current batch. E.g. ´2´ means to not only include 
        sampled direct neighbors of current batch nodes but also sampled 
        neighbors of the direct neighbors.
    shuffle:
        If `True`, shuffle the dataloaders.
    edges_directed:
        If `False`, both symmetric edge index pairs are included in the same 
        edge-level batch (1 edge has 2 symmetric edge index pairs).
    neg_edge_sampling_ratio:
        Negative sampling ratio of edges. This is currently implemented in an
        approximate way, i.e. negative edges may contain false negatives.
    pin_memory:
        If `True`, pin CPU memory for faster host→device copies.
    num_workers:
        Number of worker processes for data loading.
    persistent_workers:
        Keep workers alive between epochs (only valid when num_workers > 0).
    prefetch_factor:
        Number of batches prefetched per worker (only valid when num_workers > 0).

    Returns
    ----------
    loader_dict:
        Dictionary containing training and validation PyG LinkNeighborLoader 
        (for edge reconstruction) and NeighborLoader (for gene expression 
        reconstruction) objects.
    """
    loader_dict = {}
    hparam_kwargs = CustomNicheCompass._apply_hparam_defaults(
        edge_batch_size=edge_batch_size,
        node_batch_size=node_batch_size,
    )
    edge_batch_size = hparam_kwargs["edge_batch_size"]
    node_batch_size = hparam_kwargs["node_batch_size"]

    loader_kwargs = {
        "pin_memory": pin_memory,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Node-level dataloaders
    loader_dict["node_train_loader"] = NeighborLoader(
        node_masked_data,
        num_neighbors=[n_direct_neighbors] * n_hops,
        batch_size=node_batch_size,
        directed=False,
        shuffle=shuffle,
        input_nodes=node_masked_data.train_mask,
        **loader_kwargs)
    if node_masked_data.val_mask.sum() != 0:
        loader_dict["node_val_loader"] = NeighborLoader(
            node_masked_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=node_batch_size,
            directed=False,
            shuffle=shuffle,
            input_nodes=node_masked_data.val_mask,
            **loader_kwargs)
        
    # Edge-level dataloaders
    if edge_train_data is not None:
        loader_dict["edge_train_loader"] = LinkNeighborLoader(
            edge_train_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=edge_train_data.edge_label_index[:, edge_train_data.edge_label.bool()], # limit the edges to the ones from the edge_label_adj
            directed=edges_directed,
            shuffle=shuffle,
            neg_sampling_ratio=neg_edge_sampling_ratio,
            **loader_kwargs)
    if edge_val_data is not None and edge_val_data.edge_label.sum() != 0:
        loader_dict["edge_val_loader"] = LinkNeighborLoader(
            edge_val_data,
            num_neighbors=[n_direct_neighbors] * n_hops,
            batch_size=edge_batch_size,
            edge_label=None, # will automatically be added as 1 for all edges
            edge_label_index=edge_val_data.edge_label_index[:, edge_val_data.edge_label.bool()], # limit the edges to the ones from the edge_label_adj
            directed=edges_directed,
            shuffle=shuffle,
            neg_sampling_ratio=neg_edge_sampling_ratio,
            **loader_kwargs)

    return loader_dict


__all__ = [
    "CustomNicheCompass",
    "CustomTrainer",
    "CustomVGPGAE",
    "CustomSpatialAnnTorchDataset",
    "prepare_data",
]

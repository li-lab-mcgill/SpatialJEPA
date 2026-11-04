"""Shared SpatialJEPA model machinery.

This module holds the GCN-encoder surgery and (re)construction helpers shared
between the training script (``train_rna_msi.py``) and the target
projection script (``encode_targets.py``). Keeping them here gives a
single source of truth for how a ``ConditionalVAESTSM`` is turned into a
spatial (teacher) or graph-free (student) SpatialJEPA model.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Union

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Normal
from torch.distributions import kl_divergence as kld

import spatialmeta as smt
from spatialmeta.util.compat import Literal
from spatialmeta.util.logger import get_tqdm
from spatialmeta.util.loss import LossFunction


def build_spatial_edge_index(
    coords: np.ndarray,
    n_neighbors: int = 6,
    device: str = "cpu",
    section_labels=None,
):
    """kNN graph on spot coordinates for PyG GCNConv (spots = nodes).

    With ``section_labels`` (one label per row of ``coords``), the kNN graph is
    built **per section** and unioned into a single block-diagonal ``edge_index``
    (global node indexing, no edges between sections). This is required for
    horizontal integration: Visium pixel coordinates overlap across slides, so a
    single kNN over the pooled coordinates would create spurious cross-section
    edges.
    """
    from sklearn.neighbors import kneighbors_graph
    from torch_geometric.utils import from_scipy_sparse_matrix

    def _section_edges(sub_coords):
        k = min(n_neighbors, sub_coords.shape[0] - 1)
        if k < 1:
            return torch.empty((2, 0), dtype=torch.long)
        adj = kneighbors_graph(
            sub_coords,
            n_neighbors=k,
            mode="connectivity",
            include_self=False,
        )
        adj = adj.maximum(adj.T)
        edge_index, _ = from_scipy_sparse_matrix(adj.tocoo())
        return edge_index.long()

    coords = np.asarray(coords)
    if section_labels is None:
        return _section_edges(coords).to(device)

    section_labels = np.asarray(section_labels)
    global_idx = np.arange(coords.shape[0])
    edge_blocks = []
    for section in np.unique(section_labels):
        mask = section_labels == section
        local_edges = _section_edges(coords[mask])
        if local_edges.numel() == 0:
            continue
        # map per-section local node indices back to global row indices
        local_to_global = torch.as_tensor(global_idx[mask], dtype=torch.long)
        edge_blocks.append(local_to_global[local_edges])
    if not edge_blocks:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edge_blocks, dim=1).to(device)


def build_identity_edge_index(
    num_nodes: int,
    device: str = "cpu",
):
    """Self-loop-only graph: each node only sends messages to itself."""
    node_idx = torch.arange(num_nodes, dtype=torch.long, device=device)
    return torch.stack((node_idx, node_idx), dim=0)


def build_permuted_spatial_edge_index(
    coords: np.ndarray,
    n_neighbors: int = 6,
    device: str = "cpu",
    section_labels=None,
    seed: int = 0,
):
    """Spatial kNN graph with node labels permuted within each section.

    The permuted graph is isomorphic to the spatial one (same degree sequence and
    topology) but its edges no longer connect tissue neighbours -- a degree-matched
    control for "graph message passing" vs "tissue adjacency". The permutation uses a
    private generator so it does not consume the global numpy/torch RNG streams.
    """
    edge_index = build_spatial_edge_index(
        coords, n_neighbors=n_neighbors, device="cpu", section_labels=section_labels
    )
    n = np.asarray(coords).shape[0]
    labels = np.zeros(n, dtype=int) if section_labels is None else np.asarray(section_labels)
    rng = np.random.default_rng(seed)
    mapping = np.arange(n)
    for section in np.unique(labels):
        members = np.flatnonzero(labels == section)
        mapping[members] = rng.permutation(members)
    mapping = torch.as_tensor(mapping, dtype=torch.long)
    return mapping[edge_index].to(device)


def replace_fc_encoder_with_gcn(
    sae,
    edge_index,
    num_nodes: int,
    layer_idx: int = 0,
    neighborhood: str = "exact",
) -> None:
    """Replace one SAE encoder FCLayer with GCNConv + the same BN/ReLU/Dropout tail.

    ``neighborhood`` controls mini-batch message passing:

    - ``"induced"`` (legacy): a mini-batch only keeps edges with both endpoints in the
      batch. With random 128-spot batches over ~2.4k spots a node keeps on average
      k*127/(N-1) neighbours (~0.3 at k=6), so the graph is mostly absent in training.
    - ``"exact"``: mini-batch outputs equal the full-graph layer outputs for the batch
      nodes. The layer acts on the model *input*, and GCNConv computes
      ``A_hat @ lin(x) + b`` with a bias-free ``lin``, so ``lin(A_hat @ x)[idx] + b`` is
      exact. ``A_hat @ x`` is precomputed once by :func:`prepare_exact_neighborhood`.
    """
    from torch_geometric.nn import GCNConv

    if neighborhood not in ("exact", "induced"):
        raise ValueError(f"neighborhood must be 'exact' or 'induced', got {neighborhood!r}")

    enc_idx = layer_idx * 2
    fc_layer = sae.layers[enc_idx]
    if fc_layer is None:
        raise ValueError(f"encoder layer {layer_idx} is missing")

    post_modules = [
        module for module in fc_layer._fclayer if not isinstance(module, nn.Linear)
    ]

    class GCNEncoderLayer(nn.Module):
        """GCNConv drop-in replacement for SpatialMETA's encoder FCLayer."""

        def __init__(self, fc, edge_index_, num_nodes, post, neighborhood):
            super().__init__()
            self.in_dim = fc.in_dim
            self.out_dim = fc.out_dim
            self.device = fc.device
            self.num_nodes = num_nodes
            self.neighborhood = neighborhood
            self.register_buffer("edge_index", edge_index_)
            # full-graph inputs and their normalized aggregation (exact mode only);
            # non-persistent so checkpoints keep their size and keys
            self.register_buffer("x_in", None, persistent=False)
            self.register_buffer("agg_x", None, persistent=False)
            self.gcn = GCNConv(self.in_dim, self.out_dim, bias=True)
            self.post = post
            self._batch_node_idx = None

        def set_batch_node_idx(self, node_idx):
            self._batch_node_idx = node_idx

        @torch.no_grad()
        def set_full_graph_inputs(self, x_full):
            """Store the full-graph layer inputs and A_hat @ x for exact mini-batches."""
            from torch_geometric.nn.conv.gcn_conv import gcn_norm

            if x_full.size(0) != self.num_nodes:
                raise ValueError(
                    f"expected {self.num_nodes} full-graph rows, got {x_full.size(0)}"
                )
            x_full = x_full.detach().to(self.edge_index.device)
            edge_index, edge_weight = gcn_norm(
                self.edge_index,
                None,
                self.num_nodes,
                self.gcn.improved,
                self.gcn.add_self_loops,
                self.gcn.flow,
                x_full.dtype,
            )
            # flow="source_to_target": node i aggregates over edges (j -> i)
            adj = torch.sparse_coo_tensor(
                torch.stack((edge_index[1], edge_index[0])),
                edge_weight,
                (self.num_nodes, self.num_nodes),
            ).coalesce()
            self.x_in = x_full
            self.agg_x = torch.sparse.mm(adj, x_full)

        def _gcn_edge_index(self, x):
            if x.size(0) == self.num_nodes:
                return self.edge_index
            if self._batch_node_idx is None:
                raise RuntimeError(
                    "GCN mini-batch requires node indices; use n_per_batch=n_obs "
                    "or patch_model_encode_for_gcn()."
                )
            from torch_geometric.utils import subgraph

            node_idx = self._batch_node_idx.to(x.device)
            sub_edge_index, _ = subgraph(
                node_idx,
                self.edge_index,
                relabel_nodes=True,
                num_nodes=self.num_nodes,
            )
            return sub_edge_index

        def _exact_minibatch(self, x):
            if self.agg_x is None:
                raise RuntimeError(
                    "exact neighborhood mode needs prepare_exact_neighborhood(model) "
                    "before mini-batch encoding."
                )
            if self._batch_node_idx is None:
                raise RuntimeError(
                    "GCN mini-batch requires node indices; use patch_model_encode_for_gcn()."
                )
            node_idx = self._batch_node_idx.to(self.agg_x.device)
            # the precomputed aggregation is only valid for the data it was built from
            if not torch.allclose(x, self.x_in[node_idx], rtol=1e-5, atol=1e-6):
                raise RuntimeError(
                    "exact-neighborhood GCN input does not match the stored full-graph "
                    "inputs for these node indices."
                )
            h = self.gcn.lin(self.agg_x[node_idx])
            if self.gcn.bias is not None:
                h = h + self.gcn.bias
            return h

        def forward(self, x, cat_list=None):
            if cat_list is not None:
                raise NotImplementedError(
                    "GCN encoder replacement does not support categorical inputs."
                )
            if self.neighborhood == "exact" and x.size(0) != self.num_nodes:
                return self.post(self._exact_minibatch(x))
            return self.post(self.gcn(x, self._gcn_edge_index(x)))

        def to(self, device):
            super().to(device)
            self.device = device
            return self

    sae.layers[enc_idx] = GCNEncoderLayer(
        fc_layer,
        edge_index,
        num_nodes=num_nodes,
        post=nn.Sequential(*post_modules),
        neighborhood=neighborhood,
    )


def _gcn_encoder_layers(model):
    return [
        encoder.layers[0]
        for encoder in (model.encoder_ST, model.encoder_SM)
        if hasattr(encoder.layers[0], "set_full_graph_inputs")
    ]


@torch.no_grad()
def prepare_exact_neighborhood(model) -> None:
    """Capture each GCN layer's full-graph input and precompute ``A_hat @ x``.

    One full-batch ``encode`` pass (eval mode, forked RNG so the reparameterization
    sample does not shift the training RNG stream) records the exact tensors that
    ``encode()`` feeds the GCN layers (log1p for ST, raw for SM).
    """
    import scipy.sparse

    layers = [layer for layer in _gcn_encoder_layers(model) if layer.neighborhood == "exact"]
    if not layers:
        return

    captured = {}
    hooks = [
        layer.register_forward_pre_hook(
            lambda module, args: captured.__setitem__(module, args[0].detach())
        )
        for layer in layers
    ]
    X = model.X.toarray() if scipy.sparse.issparse(model.X) else np.asarray(model.X)
    X = torch.as_tensor(X, dtype=torch.float32, device=model.device)
    was_training = model.training
    model.eval()
    devices = [torch.device(model.device)] if torch.device(model.device).type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.encode(X)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    for layer in layers:
        layer.set_full_graph_inputs(captured[layer])


def patch_model_encode_for_gcn(model) -> None:
    """Track mini-batch node indices so GCN encoder layers can build subgraphs."""

    class IndexTrackingLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            try:
                for batch in self.loader:
                    model._current_batch_indices = batch[0].detach().cpu()
                    yield batch
            finally:
                model._current_batch_indices = None

        def __len__(self):
            return len(self.loader)

    def _set_gcn_batch_indices(node_idx):
        if node_idx is not None:
            node_idx = torch.as_tensor(node_idx, dtype=torch.long)
        for encoder in (model.encoder_ST, model.encoder_SM):
            layer = encoder.layers[0]
            if hasattr(layer, "set_batch_node_idx"):
                layer.set_batch_node_idx(node_idx)

    _orig_as_dataloader = model.as_dataloader

    def as_dataloader_with_index_tracking(*args, **kwargs):
        return IndexTrackingLoader(_orig_as_dataloader(*args, **kwargs))

    model.as_dataloader = as_dataloader_with_index_tracking

    _orig_encode = model.encode

    def encode_with_gcn_indices(X, *args, **kwargs):
        node_idx = None
        if X.size(0) != model._n_record:
            node_idx = getattr(model, "_current_batch_indices", None)
            if node_idx is None:
                raise RuntimeError(
                    "GCN mini-batch encode requires original node indices. "
                    "Call through model.as_dataloader() or use n_per_batch=model._n_record."
                )
            if len(node_idx) != X.size(0):
                raise RuntimeError(
                    f"GCN batch index length ({len(node_idx)}) does not match "
                    f"input batch size ({X.size(0)})."
                )
        _set_gcn_batch_indices(node_idx)
        try:
            return _orig_encode(X, *args, **kwargs)
        finally:
            _set_gcn_batch_indices(None)

    model.encode = encode_with_gcn_indices


class ConditionalVAESTSM_STtoSM(smt.model.ConditionalVAESTSM):
    """``ConditionalVAESTSM`` with an added ST->SM cross-reconstruction objective.

    The base model only decodes each modality from its own latent
    (``decoder_st(z_st)``, ``decoder_sm(z_sm)``). For RNA-only transfer (e.g. SEA-AD)
    we additionally want ``decoder_sm(z_st)`` -- reconstructing SM from the ST-only
    expert latent ``H['st']['q_mu']`` -- to be *directly* supervised, since that is
    exactly the path used at inference (see
    ``embed_with_spatialmeta.get_st_latent_and_msi_decoding``). This reuses the
    existing ``decoder_sm`` / ``px_sm_*`` modules and adds **no new parameters**, so
    checkpoints remain interchangeable with the base class.
    """

    def _sm_cross_reconstruction_loss(self, X, H, batch_index, reduction):
        z_st = H["st"]["q_mu"]
        if batch_index is not None:
            z_st = torch.hstack([z_st, batch_index])
        px_sm_cross = self.decoder_sm(z_st.to(self.device))
        px_sm_cross_scale = self.px_sm_scale_decoder(px_sm_cross)
        px_sm_cross_rate = self.px_sm_rate_decoder(px_sm_cross)
        px_sm_cross_dropout = self.px_sm_dropout_decoder(px_sm_cross)
        X_SM = X[:, self._type == "SM"]
        if self.reconstruction_method_sm == "zg":
            return LossFunction.zi_gaussian_reconstruction_loss(
                X_SM,
                mean=px_sm_cross_scale,
                variance=px_sm_cross_rate.exp(),
                gate_logits=px_sm_cross_dropout,
                reduction=reduction,
            )
        elif self.reconstruction_method_sm == "mse":
            return nn.MSELoss()(px_sm_cross_scale, X_SM)
        else:  # 'g'
            return LossFunction.gaussian_reconstruction_loss(
                X_SM,
                mean=px_sm_cross_scale,
                variance=px_sm_cross_rate.exp(),
                reduction=reduction,
            )

    def forward(self, X, batch_index=None, reduction="sum"):
        H, Rs, L = super().forward(X, batch_index=batch_index, reduction=reduction)
        L["reconstruction_loss_sm_cross"] = self._sm_cross_reconstruction_loss(
            X, H, batch_index, reduction
        )
        return H, Rs, L

    def fit(
        self,
        max_epoch: int = 35,
        n_per_batch: int = 128,
        mode: Optional[Literal["single", "multi"]] = None,
        sm_cross_weight: Optional[float] = None,
        **kwargs,
    ):
        """Same presets as the base class plus a default ST->SM cross weight.

        ``sm_cross_weight`` overrides the preset ST->SM weight (the presets otherwise
        overwrite any ``reconstruction_sm_cross_weight`` passed in kwargs); 0 disables
        the added cross-reconstruction term while keeping every other preset weight.
        """
        if mode == "single":
            kwargs["reconstruction_st_weight"] = 5
            kwargs["reconstruction_sm_weight"] = 1
            kwargs["reconstruction_st_corr_weight"] = 5
            kwargs["reconstruction_sm_corr_weight"] = 1
            kwargs["reconstruction_sm_cross_weight"] = 1
            kwargs["kl_weight"] = 0.5
        elif mode == "multi":
            kwargs["reconstruction_st_weight"] = 8
            kwargs["reconstruction_sm_weight"] = 2
            kwargs["reconstruction_st_corr_weight"] = 8
            kwargs["reconstruction_sm_corr_weight"] = 2
            kwargs["reconstruction_sm_cross_weight"] = 2
            kwargs["kl_weight"] = 1
            kwargs["mmd_weight"] = 10
        if sm_cross_weight is not None:
            kwargs["reconstruction_sm_cross_weight"] = sm_cross_weight
        return self.fit_core(max_epoch=max_epoch, n_per_batch=n_per_batch, **kwargs)

    def fit_core(
        self,
        max_epoch: int = 35,
        n_per_batch: int = 128,
        reconstruction_reduction: str = "sum",
        kl_weight: float = 1.0,
        reconstruction_st_weight: float = 1.0,
        reconstruction_sm_weight: float = 1.0,
        reconstruction_st_corr_weight: float = 1.0,
        reconstruction_sm_corr_weight: float = 1.0,
        reconstruction_sm_cross_weight: float = 1.0,
        n_epochs_kl_warmup: Union[int, None] = 400,
        optimizer_parameters: Iterable = None,
        weight_decay: float = 1e-6,
        lr: bool = 5e-5,
        random_seed: int = 12,
        kl_loss_reduction: str = "mean",
        mmd_weight: float = 1.0,
        cap_kl_warmup: bool = True,
        epoch_callback: Optional[Callable[[int], bool]] = None,
    ):
        """Copy of ``ConditionalVAESTSM.fit_core`` with the added ST->SM cross term.

        ``cap_kl_warmup=False`` keeps the KL warm-up length even when ``max_epoch`` is
        shorter, so a run stopped at epoch E follows the first E epochs of a longer run.
        ``epoch_callback(epoch)`` runs after every epoch; returning True stops training.
        """
        self.train()
        if n_epochs_kl_warmup:
            if cap_kl_warmup:
                n_epochs_kl_warmup = min(max_epoch, n_epochs_kl_warmup)
            kl_warmup_gradient = kl_weight / n_epochs_kl_warmup
            kl_weight_max = kl_weight
            kl_weight = 0.0

        if optimizer_parameters is None:
            optimizer = optim.AdamW(self.parameters(), lr, weight_decay=weight_decay)
        else:
            optimizer = optim.AdamW(optimizer_parameters, lr, weight_decay=weight_decay)
        pbar = get_tqdm()(range(max_epoch), desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        epoch_reconstruction_loss_st_list = []
        epoch_reconstruction_loss_sm_list = []
        epoch_reconstruction_loss_st_corr_list = []
        epoch_reconstruction_loss_sm_corr_list = []
        epoch_reconstruction_loss_sm_cross_list = []
        epoch_kldiv_loss_list = []
        epoch_total_loss_list = []
        epoch_mmd_loss_list = []

        for epoch in range(1, max_epoch + 1):
            self._trained = True
            pbar.desc = "Epoch {}".format(epoch)
            epoch_total_loss = 0
            epoch_reconstruction_loss_sm = 0
            epoch_reconstruction_loss_st = 0
            epoch_reconstruction_loss_sm_corr = 0
            epoch_reconstruction_loss_st_corr = 0
            epoch_reconstruction_loss_sm_cross = 0
            epoch_kldiv_loss = 0
            epoch_mmd_loss = 0

            X_train = self.as_dataloader(batch_size=n_per_batch, shuffle=True)
            for batch_idx in X_train:
                indices = batch_idx[0].cpu().numpy()
                X_batch = []
                for idx in indices:
                    if scipy.sparse.issparse(self.X):
                        x_row = self.X.getrow(idx).toarray().squeeze()
                    else:
                        x_row = self.X[idx]
                    X_batch.append(x_row)
                X_batch = torch.tensor(np.stack(X_batch), dtype=torch.float32).to(self.device)
                if self.batch_codes is not None:
                    batch_index = [
                        torch.tensor(code[indices], dtype=torch.long).unsqueeze(1).to(self.device)
                        for code in self.batch_codes
                    ]
                    batch_index = torch.hstack(batch_index)
                else:
                    batch_index = None

                H, Rs, L = self.forward(
                    X_batch,
                    batch_index=batch_index,
                    reduction=reconstruction_reduction,
                )

                reconstruction_loss_st = L["reconstruction_loss_st"]
                reconstruction_loss_sm = L["reconstruction_loss_sm"]
                reconstruction_loss_st_corr = L["reconstruction_loss_st_corr"]
                reconstruction_loss_sm_corr = L["reconstruction_loss_sm_corr"]
                reconstruction_loss_sm_cross = L["reconstruction_loss_sm_cross"]
                kldiv_loss = L["kldiv_loss"]
                mmd_loss = L["mmd_loss"]

                avg_reconstruction_loss_st = reconstruction_loss_st.mean() / n_per_batch
                avg_reconstruction_loss_sm = reconstruction_loss_sm.mean() / n_per_batch
                avg_reconstruction_loss_st_corr = reconstruction_loss_st_corr.mean() / n_per_batch
                avg_reconstruction_loss_sm_corr = reconstruction_loss_sm_corr.mean() / n_per_batch
                avg_reconstruction_loss_sm_cross = reconstruction_loss_sm_cross.mean() / n_per_batch
                avg_mmd_loss = mmd_loss.mean() / n_per_batch

                if kl_loss_reduction == "mean":
                    avg_kldiv_loss = kldiv_loss.mean() / n_per_batch
                elif kl_loss_reduction == "sum":
                    avg_kldiv_loss = kldiv_loss.sum() / n_per_batch

                loss = (
                    (avg_reconstruction_loss_sm * reconstruction_sm_weight)
                    + (avg_reconstruction_loss_st * reconstruction_st_weight)
                    + (avg_reconstruction_loss_sm_corr * reconstruction_sm_corr_weight)
                    + (avg_reconstruction_loss_st_corr * reconstruction_st_corr_weight)
                    + (avg_reconstruction_loss_sm_cross * reconstruction_sm_cross_weight)
                    + (avg_kldiv_loss * kl_weight)
                    + (avg_mmd_loss * mmd_weight)
                )

                epoch_reconstruction_loss_sm += avg_reconstruction_loss_sm.item()
                epoch_reconstruction_loss_st += avg_reconstruction_loss_st.item()
                epoch_reconstruction_loss_sm_corr += avg_reconstruction_loss_sm_corr.item()
                epoch_reconstruction_loss_st_corr += avg_reconstruction_loss_st_corr.item()
                epoch_reconstruction_loss_sm_cross += avg_reconstruction_loss_sm_cross.item()
                epoch_mmd_loss += avg_mmd_loss.item()
                epoch_kldiv_loss += avg_kldiv_loss.item()
                epoch_total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pbar.set_postfix(
                {
                    "reconst_sm": "{:.2e}".format(epoch_reconstruction_loss_sm),
                    "reconst_st": "{:.2e}".format(epoch_reconstruction_loss_st),
                    "reconst_sm_corr": "{:.2e}".format(epoch_reconstruction_loss_sm_corr),
                    "reconst_st_corr": "{:.2e}".format(epoch_reconstruction_loss_st_corr),
                    "reconst_sm_cross": "{:.2e}".format(epoch_reconstruction_loss_sm_cross),
                    "kldiv": "{:.2e}".format(epoch_kldiv_loss),
                    "total_loss": "{:.2e}".format(epoch_total_loss),
                    "mmd_loss": "{:.2e}".format(epoch_mmd_loss),
                }
            )

            pbar.update(1)
            epoch_reconstruction_loss_sm_list.append(epoch_reconstruction_loss_sm)
            epoch_reconstruction_loss_st_list.append(epoch_reconstruction_loss_st)
            epoch_reconstruction_loss_sm_corr_list.append(epoch_reconstruction_loss_sm_corr)
            epoch_reconstruction_loss_st_corr_list.append(epoch_reconstruction_loss_st_corr)
            epoch_reconstruction_loss_sm_cross_list.append(epoch_reconstruction_loss_sm_cross)
            epoch_kldiv_loss_list.append(epoch_kldiv_loss)
            epoch_total_loss_list.append(epoch_total_loss)
            epoch_mmd_loss_list.append(epoch_mmd_loss)

            if n_epochs_kl_warmup:
                kl_weight = min(kl_weight + kl_warmup_gradient, kl_weight_max)
            random_seed += 1
            if epoch_callback is not None:
                stop = epoch_callback(epoch)
                self.train()
                if stop:
                    break

        pbar.close()
        self.trained_state_dict = deepcopy(self.state_dict())

        return dict(
            epoch_reconstruction_loss_st_list=epoch_reconstruction_loss_st_list,
            epoch_reconstruction_loss_sm_list=epoch_reconstruction_loss_sm_list,
            epoch_reconstruction_loss_st_corr_list=epoch_reconstruction_loss_st_corr_list,
            epoch_reconstruction_loss_sm_corr_list=epoch_reconstruction_loss_sm_corr_list,
            epoch_reconstruction_loss_sm_cross_list=epoch_reconstruction_loss_sm_cross_list,
            epoch_kldiv_loss_list=epoch_kldiv_loss_list,
            epoch_total_loss_list=epoch_total_loss_list,
            epoch_mmd_loss_list=epoch_mmd_loss_list,
        )


DECODER_MODULE_NAMES = (
    "decoder_st",
    "decoder_sm",
    "px_rna_scale_decoder",
    "px_rna_rate_decoder",
    "px_rna_dropout_decoder",
    "px_sm_scale_decoder",
    "px_sm_rate_decoder",
    "px_sm_dropout_decoder",
)


def _module_has_graph_op(module) -> bool:
    """True if any submodule is a graph (message-passing) layer."""
    from torch_geometric.nn import MessagePassing

    return any(
        isinstance(m, MessagePassing) or type(m).__name__ == "GCNEncoderLayer"
        for m in module.modules()
    )


def copy_decoder_weights(teacher, student):
    """Copy teacher decode-side weights into student, skipping any graph-op module.

    The student's decoder is never trained by ``SpatialJEPA_trainer`` (distillation
    only touches the encoder/latent heads), so without this its MSI decoding runs
    through random-init weights. Decoder modules are plain ``FCLayer``/``nn.Linear``
    today, so all of ``DECODER_MODULE_NAMES`` are copied; any module that grows a
    graph op is skipped (returned in ``skipped``) since its weights are not a safe
    drop-in for the graph-free student.
    """
    copied, skipped = [], []
    for name in DECODER_MODULE_NAMES:
        t = getattr(teacher, name, None)
        s = getattr(student, name, None)
        if t is None or s is None:
            continue
        if _module_has_graph_op(t) or _module_has_graph_op(s):
            skipped.append(name)
            continue
        s.load_state_dict(t.state_dict())
        copied.append(name)
    return copied, skipped


def spatialJEPA_model(
    joint_adata,
    graph_conv=True,
    full_graph=False,
    device="cuda:0",
    batch_keys=None,
    section_key=None,
    graph=None,
    n_neighbors=6,
    graph_seed=0,
    neighborhood="exact",
    dropout_rate=0.1,
):
    """Build a ConditionalVAESTSM, optionally with GCN encoders.

    ``joint_adata`` must carry spot coordinates in ``obsm['spatial']`` when
    ``graph_conv=True`` (full-graph teacher needs them to build the kNN graph;
    the identity-graph student only needs ``n_obs``).

    For horizontal (multi-section) integration, pass ``batch_keys`` (a list of
    ``obs`` columns identifying the section/batch) to enable decoder batch
    conditioning + the MMD alignment loss, and ``section_key`` so the full-graph
    teacher builds a block-diagonal spatial graph (no cross-section edges).

    ``graph`` ("spatial", "identity" or "permuted") overrides ``full_graph`` when
    given; "permuted" is the spatial kNN graph with node labels shuffled within each
    section (``graph_seed``). ``neighborhood`` selects mini-batch message passing for
    non-identity graphs (see :func:`replace_fc_encoder_with_gcn`); identity graphs are
    unaffected by it. ``dropout_rate`` is the encoders' hidden-layer dropout (SpatialMETA
    default 0.1); it has no parameters, so checkpoints rebuild at the default.
    """

    model = ConditionalVAESTSM_STtoSM(
        joint_adata,
        device=device,
        reconstruction_method_sm='g',
        reconstruction_method_st='zinb',
        batch_keys=batch_keys,
        dropout_rate=dropout_rate,
    )

    if graph is None:
        graph = "spatial" if full_graph else "identity"
    if graph not in ("spatial", "identity", "permuted"):
        raise ValueError(f"graph must be 'spatial', 'identity' or 'permuted', got {graph!r}")
    model._spatialjepa_meta = dict(
        graph_conv=graph_conv,
        full_graph=graph != "identity",
        graph=graph,
        n_neighbors=n_neighbors,
        graph_seed=graph_seed,
        neighborhood=neighborhood,
        section_key=section_key,
    )

    if graph_conv:
        if graph != "identity" and "spatial" not in joint_adata.obsm:
            raise ValueError(
                f"graph={graph!r} requires joint_adata.obsm['spatial']; set it before "
                "calling spatialJEPA_model() for teacher/full-graph models."
            )

        section_labels = (
            joint_adata.obs[section_key].to_numpy()
            if section_key is not None
            else None
        )
        if graph == "spatial":
            edge_index = build_spatial_edge_index(
                joint_adata.obsm["spatial"],
                n_neighbors=n_neighbors,
                device=str(model.device),
                section_labels=section_labels,
            )
        elif graph == "permuted":
            edge_index = build_permuted_spatial_edge_index(
                joint_adata.obsm["spatial"],
                n_neighbors=n_neighbors,
                device=str(model.device),
                section_labels=section_labels,
                seed=graph_seed,
            )
        else:
            edge_index = build_identity_edge_index(
                joint_adata.n_obs,
                device=str(model.device),
            )

        # SAE.layers = ModuleList([FCLayer(in->128), None]) for encode_only stacks=[128]
        num_nodes = joint_adata.n_obs
        layer_neighborhood = neighborhood if graph != "identity" else "induced"
        replace_fc_encoder_with_gcn(
            model.encoder_ST, edge_index, num_nodes, layer_idx=0, neighborhood=layer_neighborhood
        )
        replace_fc_encoder_with_gcn(
            model.encoder_SM, edge_index, num_nodes, layer_idx=0, neighborhood=layer_neighborhood
        )
        patch_model_encode_for_gcn(model)
        model.to(model.device)
        prepare_exact_neighborhood(model)

    return model


def save_spatialjepa_model(model, path, *, full_graph: bool):
    """Persist a SpatialJEPA model plus the metadata needed to rebuild it."""
    meta = getattr(model, "_spatialjepa_meta", {})
    torch.save(
        {
            "state_dict": model.state_dict(),
            "full_graph": full_graph,
            "graph_conv": True,
            "graph": meta.get("graph", "spatial" if full_graph else "identity"),
            "n_neighbors": meta.get("n_neighbors", 6),
            "graph_seed": meta.get("graph_seed", 0),
            "neighborhood": meta.get("neighborhood", "induced"),
            "section_key": meta.get("section_key"),
            "reconstruction_method_sm": model.reconstruction_method_sm,
            "reconstruction_method_st": model.reconstruction_method_st,
            "hidden_stacks": model.hidden_stacks,
            "n_latent": model.n_latent,
            "batch_keys": getattr(model, "batch_keys", None),
        },
        path,
    )


def rebuild_spatialjepa_model(state_dict, meta, template_joint_adata, device="cuda:0"):
    """Rebuild a SpatialJEPA model on a (possibly new) ``template_joint_adata``.

    ``meta`` is a checkpoint dict (or the same keys). The ``edge_index`` buffers carry
    the *training* graph (sized to the training ``n_obs``); they are dropped so the
    freshly built, template-sized graph is kept instead. All learned weights load by
    name via ``strict=False``. Checkpoints written before the ``neighborhood`` key
    existed were trained with induced-subgraph mini-batches and rebuild that way.
    """
    model = spatialJEPA_model(
        template_joint_adata,
        graph_conv=meta["graph_conv"],
        full_graph=meta["full_graph"],
        device=device,
        batch_keys=meta.get("batch_keys"),
        section_key=meta.get("section_key"),
        graph=meta.get("graph"),
        n_neighbors=meta.get("n_neighbors", 6),
        graph_seed=meta.get("graph_seed", 0),
        neighborhood=meta.get("neighborhood", "induced"),
    )
    state = {k: v for k, v in state_dict.items() if not k.endswith("edge_index")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    leftover = [k for k in (list(missing) + list(unexpected)) if not k.endswith("edge_index")]
    if leftover:
        raise RuntimeError(f"Unexpected missing/unexpected keys when rebuilding: {leftover}")
    model.eval()
    return model


def load_spatialjepa_model(checkpoint_path, template_joint_adata, device="cuda:0"):
    """Rebuild a saved SpatialJEPA model on a (possibly new) ``template_joint_adata``."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    return rebuild_spatialjepa_model(ckpt["state_dict"], ckpt, template_joint_adata, device=device)


# ---------------------------------------------------------------------------
# Teacher-student training (moved from train_rna_msi.py)
# ---------------------------------------------------------------------------

def _optimizer_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


class _EpochStepLRState:
    def __init__(self, label, optimizer, steps_per_epoch):
        self.label = label
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.optimizer_step_count = 0
        self.epoch_lr_list = []
        self.scheduler = None


@contextmanager
def _scheduled_adamw_epochs(label: str, steps_per_epoch: int, step_size: int, gamma: float):
    """Temporarily add epoch-level StepLR to AdamW optimizers created inside fit()."""
    if steps_per_epoch <= 0:
        raise ValueError("steps_per_epoch must be positive for LR scheduling")

    original_adamw = optim.AdamW
    states = []

    def scheduled_adamw(*args, **kwargs):
        optimizer = original_adamw(*args, **kwargs)
        state = _EpochStepLRState(label=label, optimizer=optimizer, steps_per_epoch=steps_per_epoch)
        state.scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        scheduler_wrapped_step = optimizer.step

        def step_with_epoch_scheduler(*step_args, **step_kwargs):
            result = scheduler_wrapped_step(*step_args, **step_kwargs)
            state.optimizer_step_count += 1
            if state.optimizer_step_count % state.steps_per_epoch == 0:
                state.epoch_lr_list.append(_optimizer_lr(optimizer))
                state.scheduler.step()
            return result

        step_with_epoch_scheduler._with_counter = getattr(
            scheduler_wrapped_step,
            "_with_counter",
            False,
        )
        optimizer.step = step_with_epoch_scheduler
        states.append(state)
        return optimizer

    optim.AdamW = scheduled_adamw
    try:
        yield states
    finally:
        optim.AdamW = original_adamw


@dataclass
class StudentSpec:
    """An additional student distilled from the same teacher batches.

    ``mask_sm_input`` zeroes the student's SM (MSI) input columns, giving an RNA-only
    student; ``distill_targets`` defaults to the trainer's targets when None.
    """

    name: str
    model: object
    distill_targets: Optional[tuple] = None
    mask_sm_input: bool = False
    optimizer: object = None
    lr_scheduler: object = None
    distill_loss: list = field(default_factory=list)
    epoch_distill_loss: list = field(default_factory=list)
    epoch_lr: list = field(default_factory=list)
    current_epoch_loss: list = field(default_factory=list)


class SpatialJEPA_trainer:
    """Train a full-graph teacher while distilling its batch embeddings to a student.

    ``extra_students`` (list of :class:`StudentSpec`) are each stepped once on every
    teacher batch, after the primary student, so several students can share one
    teacher run (e.g. an RNA-only student alongside the paired-input one).

    ``distill_schedule`` sets when the students learn:
      "batch"  one student step per teacher mini-batch, against that batch's teacher
               embeddings from the teacher's forward pass (before its update).
      "epoch"  no student steps during a teacher epoch; once the epoch's last teacher
               update is done, one student epoch (same number of steps, fresh shuffle)
               against targets from the end-of-epoch teacher. ``epoch_targets``="train"
               keeps the batch scheme's kind of target (train-mode dropout and batch
               statistics, with BatchNorm running stats left untouched); "eval" uses
               the deterministic eval-mode teacher.
    ``student_seed`` (optional) gives the students their own torch RNG stream, so the
    teacher's dropout and shuffling draws, and hence the teacher itself, do not depend
    on the schedule.
    """

    DISTILL_TARGETS = (
        (("q_mu",), 1.0),
        (("st", "q_mu"), 0.5),
        (("sm", "q_mu"), 0.5),
    )

    def __init__(
        self,
        teacher_model,
        student_model,
        n_per_batch=128,
        distill_targets=DISTILL_TARGETS,
        student_lr=1e-3,
        student_weight_decay=1e-4,
        student_optimizer=None,
        extra_students=None,
        distill_schedule="batch",
        epoch_targets="train",
        student_seed=None,
    ):
        if distill_schedule not in ("batch", "epoch"):
            raise ValueError(f"distill_schedule must be 'batch' or 'epoch', got {distill_schedule!r}")
        if epoch_targets not in ("train", "eval"):
            raise ValueError(f"epoch_targets must be 'train' or 'eval', got {epoch_targets!r}")
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.n_per_batch = n_per_batch
        self.distill_targets = distill_targets
        self.distill_schedule = distill_schedule
        self.epoch_targets = epoch_targets
        self.student_seed = student_seed
        self._student_rng_state = None
        self._epoch_rng = np.random.default_rng(0 if student_seed is None else student_seed)
        self._pending_epoch_distill = False
        self.student_optimizer = student_optimizer or optim.Adam(
            self.student_model.parameters(),
            lr=student_lr,
            weight_decay=student_weight_decay,
        )
        self.student_lr_scheduler = None
        self.extra_students = list(extra_students or [])
        for spec in self.extra_students:
            if spec.distill_targets is None:
                spec.distill_targets = self.distill_targets
            if spec.optimizer is None:
                spec.optimizer = optim.Adam(
                    spec.model.parameters(),
                    lr=student_lr,
                    weight_decay=student_weight_decay,
                )

        self.batches_per_epoch = len(
            self.teacher_model.as_dataloader(batch_size=n_per_batch, shuffle=True)
        )
        self.epoch_H = []
        self.batch_H = []
        self.current_epoch_H = []
        self.student_distill_loss = []
        self.epoch_student_distill_loss = []
        self.epoch_student_lr_list = []

    def get_expression_batch(self, model, indices, mask_sm=False):
        """Fetch a model input batch by original observation indices.

        ``mask_sm`` zeroes the SM (MSI) columns, as for an RNA-only input.
        """
        from scipy.sparse import issparse

        indices = torch.as_tensor(indices, dtype=torch.long).detach().cpu()
        indices_np = indices.numpy()
        if issparse(model.X):
            X_batch = model.X[indices_np].toarray()
        else:
            X_batch = model.X[indices_np]
        X_batch = torch.as_tensor(np.asarray(X_batch), dtype=torch.float32, device=model.device)
        if mask_sm:
            sm_cols = torch.as_tensor(np.asarray(model._type) == "SM", device=model.device)
            X_batch[:, sm_cols] = 0.0
        return X_batch

    def encode_indices(self, model, indices, mask_sm=False):
        """Encode the same observations used by the teacher batch."""
        indices = torch.as_tensor(indices, dtype=torch.long).detach().cpu()
        had_current_indices = hasattr(model, "_current_batch_indices")
        old_current_indices = getattr(model, "_current_batch_indices", None)
        model._current_batch_indices = indices
        try:
            X_batch = self.get_expression_batch(model, indices, mask_sm=mask_sm)
            return model.encode(X_batch)
        finally:
            if had_current_indices:
                model._current_batch_indices = old_current_indices
            else:
                delattr(model, "_current_batch_indices")

    @staticmethod
    def get_nested(mapping, path):
        value = mapping
        for key in path:
            value = value[key]
        return value

    @staticmethod
    def detach_H(H, keys=("q_mu",), nested_keys=("q_mu",), device=None):
        def detach_tensor(x):
            x = x.detach()
            return x if device is None else x.to(device)

        out = {}
        for k, v in H.items():
            if k in ("st", "sm"):
                out[k] = {
                    sk: detach_tensor(sv)
                    for sk, sv in v.items()
                    if sk in nested_keys and torch.is_tensor(sv)
                }
            elif k in keys and torch.is_tensor(v):
                out[k] = detach_tensor(v)
        return out

    @staticmethod
    def zscore(x, eps=1e-6):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mean) / std

    def h_distillation_loss(self, H_student, H_teacher, normalize=True, distill_targets=None):
        loss = None
        for path, weight in (distill_targets or self.distill_targets):
            student_value = self.get_nested(H_student, path)
            teacher_value = self.get_nested(H_teacher, path).to(student_value.device)

            if normalize:
                student_value = self.zscore(student_value)
                teacher_value = self.zscore(teacher_value)

            term = weight * torch.nn.functional.mse_loss(student_value, teacher_value)
            loss = term if loss is None else loss + term

        return loss

    def _distillation_step(
        self, model, optimizer, H_teacher, indices, K=1, distill_targets=None, mask_sm=False
    ):
        model.train()
        last_loss = None

        for _ in range(K):
            optimizer.zero_grad(set_to_none=True)
            H_student = self.encode_indices(model, indices, mask_sm=mask_sm)
            distil_loss = self.h_distillation_loss(
                H_student, H_teacher, distill_targets=distill_targets
            )
            distil_loss.backward()
            optimizer.step()
            last_loss = distil_loss.detach()

        return float(last_loss.cpu())

    def jepa_distillation_step(self, H_teacher, indices, K=1):
        return self._distillation_step(
            self.student_model, self.student_optimizer, H_teacher, indices, K=K
        )

    def extra_students_step(self, H_teacher, indices):
        for spec in self.extra_students:
            loss = self._distillation_step(
                spec.model,
                spec.optimizer,
                H_teacher,
                indices,
                distill_targets=spec.distill_targets,
                mask_sm=spec.mask_sm_input,
            )
            spec.distill_loss.append(loss)
            spec.current_epoch_loss.append(loss)

    @contextmanager
    def _student_rng(self):
        """Run student work on its own torch RNG stream when ``student_seed`` is set."""
        if self.student_seed is None:
            yield
            return
        device = torch.device(self.student_model.device)
        devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if self._student_rng_state is None:
                torch.default_generator.manual_seed(self.student_seed)
                if devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(self.student_seed)
            else:
                torch.set_rng_state(self._student_rng_state[0])
                if devices:
                    torch.cuda.set_rng_state(self._student_rng_state[1], device)
            try:
                yield
            finally:
                self._student_rng_state = (
                    torch.get_rng_state(),
                    torch.cuda.get_rng_state(device) if devices else None,
                )

    def _end_student_epoch(self, losses):
        """Per-epoch student bookkeeping and LR scheduler steps."""
        self.epoch_student_distill_loss.append(float(np.mean(losses)))
        self.epoch_student_lr_list.append(_optimizer_lr(self.student_optimizer))
        if self.student_lr_scheduler is not None:
            self.student_lr_scheduler.step()
        for spec in self.extra_students:
            spec.epoch_distill_loss.append(float(np.mean(spec.current_epoch_loss)))
            spec.epoch_lr.append(_optimizer_lr(spec.optimizer))
            if spec.lr_scheduler is not None:
                spec.lr_scheduler.step()
            spec.current_epoch_loss.clear()

    @torch.no_grad()
    def _teacher_targets(self, indices):
        """End-of-epoch teacher embeddings for one student batch (no teacher side effects)."""
        teacher = self.teacher_model
        was_training = teacher.training
        batch_norms = [m for m in teacher.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        momenta = [m.momentum for m in batch_norms]
        try:
            if self.epoch_targets == "train":
                teacher.train()
                for m in batch_norms:
                    m.momentum = 0.0      # batch statistics, running stats unchanged
            else:
                teacher.eval()
            return self.detach_H(self.encode_indices(teacher, indices))
        finally:
            for m, momentum in zip(batch_norms, momenta):
                m.momentum = momentum
            teacher.train(was_training)

    def distill_epoch_end(self):
        """One student epoch against the teacher as it stands after its latest epoch."""
        n_obs = self.teacher_model.X.shape[0]
        order = self._epoch_rng.permutation(n_obs)
        batches = [order[i:i + self.n_per_batch] for i in range(0, n_obs, self.n_per_batch)]
        if len(batches) > 1 and len(batches[-1]) == 1:   # BatchNorm cannot take one row
            batches[-2] = np.concatenate([batches[-2], batches.pop()])
        losses = []
        with self._student_rng():
            for idx in batches:
                H_teacher = self._teacher_targets(idx)
                losses.append(self.jepa_distillation_step(H_teacher, idx, K=1))
                self.extra_students_step(H_teacher, idx)
        self.student_distill_loss.extend(losses)
        self._end_student_epoch(losses)
        self._pending_epoch_distill = False

    def capture_teacher_outputs(self, outputs):
        H, Rs, L = outputs
        indices = getattr(self.teacher_model, "_current_batch_indices", None)
        if indices is None:
            raise RuntimeError(
                "Teacher batch indices were not available during distillation. "
                "Make sure patch_model_encode_for_gcn(teacher_model) has been applied."
            )

        indices = indices.detach().cpu().clone()
        if self.distill_schedule == "batch":
            H_teacher = self.detach_H(H)
            with self._student_rng():
                distil_loss = self.jepa_distillation_step(H_teacher, indices, K=1)
                self.extra_students_step(H_teacher, indices)
            self.student_distill_loss.append(distil_loss)
        else:
            distil_loss = float("nan")    # the student learns at the end of the epoch

        batch_record = dict(
            indices=indices,
            H=self.detach_H(H, device="cpu"),
            student_distill_loss=distil_loss,
        )
        self.batch_H.append(batch_record)
        self.current_epoch_H.append(batch_record)

        if len(self.current_epoch_H) == self.batches_per_epoch:
            self.epoch_H.append(self.current_epoch_H.copy())
            if self.distill_schedule == "batch":
                self._end_student_epoch([r["student_distill_loss"] for r in self.current_epoch_H])
            else:
                # this batch's teacher update has not happened yet, so the student epoch
                # runs just before the next teacher forward (or after fit returns)
                self._pending_epoch_distill = True
            self.current_epoch_H.clear()

    def fit_teacher(self, **fit_kwargs):
        teacher_forward = self.teacher_model.forward

        def teacher_forward_with_distillation(*args, **kwargs):
            if self._pending_epoch_distill:
                self.distill_epoch_end()
            outputs = teacher_forward(*args, **kwargs)
            self.capture_teacher_outputs(outputs)
            return outputs

        self.teacher_model.forward = teacher_forward_with_distillation
        try:
            loss_dict = self.teacher_model.fit(
                n_per_batch=self.n_per_batch,
                **fit_kwargs,
            )
        finally:
            self.teacher_model.forward = teacher_forward
        if self._pending_epoch_distill:
            self.distill_epoch_end()

        loss_dict["epoch_student_distill_loss_list"] = self.epoch_student_distill_loss
        loss_dict["epoch_student_lr_list"] = self.epoch_student_lr_list
        for spec in self.extra_students:
            loss_dict[f"epoch_{spec.name}_distill_loss_list"] = spec.epoch_distill_loss
            loss_dict[f"epoch_{spec.name}_lr_list"] = spec.epoch_lr
        return loss_dict

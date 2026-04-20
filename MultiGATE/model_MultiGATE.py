from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

RHO_MASK_MODE_NONE = 0
RHO_MASK_MODE_FIXED = 1
RHO_MASK_MODE_TRAINABLE_MASKED = 2

def _load_legacy_tf_mgate():
    from .model_MultiGATE_legacy_tf import MGATE as legacy_cls

    return legacy_cls

def decorr_loss_correlation(
    z: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Stable decorrelation loss using the off-diagonal of the correlation matrix.

    Args:
        z: Tensor of shape [batch, dim]
        eps: Numerical stability term

    Returns:
        Scalar loss
    """
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, unbiased=False, keepdim=True) + eps)

    b = z.shape[0]
    corr = (z.T @ z) / b

    upper = torch.triu(corr, diagonal=1)
    num_offdiag = corr.shape[0] * (corr.shape[0] - 1) // 2
    loss = upper.pow(2).sum() / num_offdiag
    return loss

class LegacyTFMGATE(object):
    def __new__(cls, *args, **kwargs):
        legacy_cls = _load_legacy_tf_mgate()
        return legacy_cls(*args, **kwargs)


class MGATE(nn.Module):

    def __init__(
        self,
        hidden_dims1,
        hidden_dims2,
        spot_num,
        temp=1.0,
        nonlinear=True,
        weight_decay=0.0001,
        vgp_anchor_mode="spot",
        skip_gp_attention=True,
        linear_etm_decoder=True,
        etm_emb_dim: Optional[int]=None,
        rho_rna_mask: Optional[torch.Tensor]=None,
        rho_atac_mask: Optional[torch.Tensor]=None,
        rho_mask_mode: str="trainable_masked",
    ):
        super(MGATE, self).__init__()
        self.n_layers = len(hidden_dims1) - 1
        if self.n_layers < 1:
            raise ValueError("hidden_dims must define at least one encoder layer")
        if vgp_anchor_mode not in {"spot", "feature"}:
            raise ValueError("vgp_anchor_mode must be one of {'spot', 'feature'}")

        self.nonlinear = nonlinear
        self.hidden_dims1 = hidden_dims1
        self.hidden_dims2 = hidden_dims2
        self.temp = temp
        self.spot_num = int(spot_num)
        self.vgp_anchor_mode = vgp_anchor_mode
        self.skip_gp_attention = skip_gp_attention

        self.W1 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dims1[i], hidden_dims1[i + 1]))
            for i in range(self.n_layers)
        ])
        self.W2 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dims2[i], hidden_dims2[i + 1]))
            for i in range(self.n_layers)
        ])

        self.v1_0 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dims1[i + 1], 1))
            for i in range(self.n_layers - 1)
        ])
        self.v1_1 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dims1[i + 1], 1))
            for i in range(self.n_layers - 1)
        ])
        self.v2_0 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dims2[i + 1], 1))
            for i in range(self.n_layers - 1)
        ])
        self.v2_1 = nn.ParameterList([
            nn.Parameter(torch.empty(hidden_dims2[i + 1], 1))
            for i in range(self.n_layers - 1)
        ])

        if self.vgp_anchor_mode == "feature":
            feat_num = hidden_dims1[0] + hidden_dims2[0]
            self.vgp0 = nn.Parameter(torch.empty(feat_num, 1))
            self.vgp1 = nn.Parameter(torch.empty(feat_num, 1))
        else:
            self.vgp0 = nn.Parameter(torch.empty(self.spot_num, 1))
            self.vgp1 = nn.Parameter(torch.empty(self.spot_num, 1))

        emb_dim1 = hidden_dims1[-1]
        emb_dim2 = hidden_dims2[-1]
        self.W_i = nn.Parameter(torch.empty(emb_dim1, emb_dim1))
        self.W_t = nn.Parameter(torch.empty(emb_dim2, emb_dim2))
        self.logit_scale = nn.Parameter(torch.tensor(float(temp), dtype=torch.float32))

        self.linear_etm_decoder = linear_etm_decoder
        self.etm_emb_dim = None
        self.rho_mask_mode = "none"
        self.register_buffer("rho_is_fixed_mask", torch.tensor(0, dtype=torch.uint8), persistent=True)
        self.register_buffer("rho_mask_mode_code", torch.tensor(RHO_MASK_MODE_NONE, dtype=torch.uint8), persistent=True)
        if linear_etm_decoder:
            if rho_mask_mode not in {"fixed", "trainable_masked"}:
                raise ValueError("rho_mask_mode must be one of {'fixed', 'trainable_masked'}.")
            if (rho_rna_mask is None) ^ (rho_atac_mask is None):
                raise ValueError(
                    "Both rho_rna_mask and rho_atac_mask must be provided together."
                )

            if rho_rna_mask is not None:
                rho_rna_tensor = torch.as_tensor(rho_rna_mask, dtype=torch.float32)
                rho_atac_tensor = torch.as_tensor(rho_atac_mask, dtype=torch.float32)

                if rho_rna_tensor.ndim != 2 or rho_atac_tensor.ndim != 2:
                    raise ValueError("rho masks must be rank-2 tensors of shape (n_pathways, n_features).")
                if rho_rna_tensor.shape[1] != hidden_dims1[0]:
                    raise ValueError(
                        "rho_rna_mask has {} genes but model expects {} features.".format(
                            rho_rna_tensor.shape[1],
                            hidden_dims1[0],
                        )
                    )
                if rho_atac_tensor.shape[1] != hidden_dims2[0]:
                    raise ValueError(
                        "rho_atac_mask has {} peaks but model expects {} features.".format(
                            rho_atac_tensor.shape[1],
                            hidden_dims2[0],
                        )
                    )
                if rho_rna_tensor.shape[0] != rho_atac_tensor.shape[0]:
                    raise ValueError(
                        "rho mask pathway dimensions differ: RNA {} vs ATAC {}.".format(
                            rho_rna_tensor.shape[0],
                            rho_atac_tensor.shape[0],
                        )
                    )

                inferred_etm_emb_dim = int(rho_rna_tensor.shape[0])
                if etm_emb_dim is not None and int(etm_emb_dim) != inferred_etm_emb_dim:
                    raise ValueError(
                        "Provided etm_emb_dim={} does not match mask pathway count={}.".format(
                            int(etm_emb_dim),
                            inferred_etm_emb_dim,
                        )
                    )
                self.etm_emb_dim = inferred_etm_emb_dim
                self.alpha = nn.Parameter(torch.empty(emb_dim1, self.etm_emb_dim))
                self.rho_mask_mode = rho_mask_mode
                if rho_mask_mode == "fixed":
                    self.register_buffer("rho_rna", rho_rna_tensor.clone().detach(), persistent=True)
                    self.register_buffer("rho_atac", rho_atac_tensor.clone().detach(), persistent=True)
                    self.rho_is_fixed_mask.fill_(1)
                    self.rho_mask_mode_code.fill_(RHO_MASK_MODE_FIXED)
                else:
                    self.rho_rna = nn.Parameter(torch.empty_like(rho_rna_tensor))
                    self.rho_atac = nn.Parameter(torch.empty_like(rho_atac_tensor))
                    self.register_buffer("rho_rna_mask", rho_rna_tensor.clone().detach(), persistent=True)
                    self.register_buffer("rho_atac_mask", rho_atac_tensor.clone().detach(), persistent=True)
                    self.rho_mask_mode_code.fill_(RHO_MASK_MODE_TRAINABLE_MASKED)
            else:
                if rho_mask_mode != "fixed":
                    raise ValueError(
                        "rho_mask_mode='trainable_masked' requires both rho_rna_mask and rho_atac_mask."
                    )
                self.etm_emb_dim = int(etm_emb_dim) if etm_emb_dim is not None else 20
                self.alpha = nn.Parameter(torch.empty(emb_dim1, self.etm_emb_dim))
                self.rho_rna = nn.Parameter(torch.empty(self.etm_emb_dim, hidden_dims1[0]))
                self.rho_atac = nn.Parameter(torch.empty(self.etm_emb_dim, hidden_dims2[0]))

        self.bn_rna = nn.LayerNorm(emb_dim1)
        self.bn_atac = nn.LayerNorm(emb_dim2)
        #self.bn_rna = nn.BatchNorm1d(emb_dim1)
        #self.bn_atac = nn.BatchNorm1d(emb_dim2)
        #self.bn_rna = nn.Identity()
        #self.bn_atac = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.W1:
            nn.init.xavier_uniform_(weight)
        for weight in self.W2:
            nn.init.xavier_uniform_(weight)

        for vec in self.v1_0:
            nn.init.xavier_uniform_(vec)
        for vec in self.v1_1:
            nn.init.xavier_uniform_(vec)
        for vec in self.v2_0:
            nn.init.xavier_uniform_(vec)
        for vec in self.v2_1:
            nn.init.xavier_uniform_(vec)

        nn.init.xavier_uniform_(self.vgp0)
        nn.init.xavier_uniform_(self.vgp1)
        nn.init.xavier_uniform_(self.W_i)
        nn.init.xavier_uniform_(self.W_t)

        if self.linear_etm_decoder:
            nn.init.xavier_uniform_(self.alpha)
            if isinstance(self.rho_rna, nn.Parameter):
                if self.rho_mask_mode == "trainable_masked":
                    self._masked_xavier_uniform_(self.rho_rna, self.rho_rna_mask)
                else:
                    nn.init.xavier_uniform_(self.rho_rna)
            if isinstance(self.rho_atac, nn.Parameter):
                if self.rho_mask_mode == "trainable_masked":
                    self._masked_xavier_uniform_(self.rho_atac, self.rho_atac_mask)
                else:
                    nn.init.xavier_uniform_(self.rho_atac)

    def _effective_rho(self, rho, rho_mask=None):
        if rho_mask is None:
            return rho
        return rho * rho_mask

    def _masked_xavier_uniform_(self, tensor, mask):
        with torch.no_grad():
            fan_in = mask.sum(1).float().mean().clamp_min(1.0)
            fan_out = mask.sum(0).float().mean().clamp_min(1.0)
            bound = torch.sqrt(6.0 / (fan_in + fan_out))   # gain = 1
            tensor.data.uniform_(-bound, bound)
            tensor.data.mul_(mask)

    def forward(self, A, prune_A, GP, X1, X2):
        del prune_A
        self.C1 = {}
        self.C2 = {}
        self.Cgp = {}

        # Encoder
        if not self.skip_gp_attention:
            H = torch.cat([X1.transpose(0, 1), X2.transpose(0, 1)], dim=0)
            if self.vgp_anchor_mode == "feature":
                self.Cgp[0] = self._gp_attention_layer(GP, self.vgp0, self.vgp1)  # att_fg of Eq. (3), feature-anchored
            else:
                if X1.shape[0] != self.vgp0.shape[0]:
                    raise ValueError(
                        "Spot-anchored vgp expects {} cells, but got {}. "
                        "Instantiate MGATE with matching spot_num for this dataset."
                        .format(self.vgp0.shape[0], X1.shape[0])
                    )
                self.Cgp[0] = self.graph_attention_layer(GP, H, self.vgp0, self.vgp1)  # att_fg of Eq. (3), spot-anchored
            H = torch.sparse.mm(self.Cgp[0], H)
            H = F.relu(H) # after relu, H is output of Eq. (1) in the paper, i.e. ~X^T_(f)

            split_point = X1.shape[1]
            H1 = H[:split_point, :].transpose(0, 1).contiguous()
            H2 = H[split_point:, :].transpose(0, 1).contiguous()

        else:
            H1 = X1
            H2 = X2

        for layer in range(self.n_layers):
            H1 = self.__encoder(A, H1, self.W1, self.C1, self.v1_0, self.v1_1, layer) # output of Eq. (5) for first modality (RNA)
            H2 = self.__encoder(A, H2, self.W2, self.C2, self.v2_0, self.v2_1, layer) # output of Eq. (5) for second modality (ATAC)
            if self.nonlinear and layer != self.n_layers - 1:
                H1 = F.elu(H1)
                H2 = F.elu(H2)

        self.H1 = H1 # output of Eq. (8) for first modality (RNA)
        self.H2 = H2 # output of Eq. (8) for second modality (ATAC) 

        # Decoder
        if not self.linear_etm_decoder:
            for layer in range(self.n_layers - 1, -1, -1): # layer index decreases from n_layers - 1 to 0
                H1 = self.__decoder(H1, self.W1, self.C1, layer) # Eq. (9) for first modality (RNA)
                H2 = self.__decoder(H2, self.W2, self.C2, layer) # Eq. (9) for second modality (ATAC)
                if self.nonlinear and layer != 0:
                    H1 = F.elu(H1)
                    H2 = F.elu(H2)
        else:
            H = 0.5 * (H1 + H2)
            H = F.softmax(H, dim=1)
            H1 = self.__linear_etm_decoder(
                H,
                self.alpha,
                self._effective_rho(self.rho_rna, getattr(self, "rho_rna_mask", None)),
            )
            H2 = self.__linear_etm_decoder(
                H,
                self.alpha,
                self._effective_rho(self.rho_atac, getattr(self, "rho_atac_mask", None)),
            )

        if not self.skip_gp_attention:
            H = torch.cat([H1.transpose(0, 1), H2.transpose(0, 1)], dim=0)
            H = torch.sparse.mm(self.Cgp[0], H) # Eq. (11)
            H = F.elu(H)

            H1_dec = H[:split_point, :].transpose(0, 1).contiguous()
            H2_dec = H[split_point:, :].transpose(0, 1).contiguous()

            X1_ = H1_dec
            X2_ = H2_dec
        else:
            X1_ = H1
            X2_ = H2

        # CLIP-style alignment loss
        rna_proj = self.bn_rna(torch.matmul(self.H1, self.W_i))
        atac_proj = self.bn_atac(torch.matmul(self.H2, self.W_t))
        RNA_e = F.normalize(rna_proj, p=2, dim=1)
        ATAC_e = F.normalize(atac_proj, p=2, dim=1)

        self.H1 = RNA_e
        self.H2 = ATAC_e

        logits = torch.matmul(RNA_e, ATAC_e.transpose(0, 1)) * torch.exp(self.logit_scale)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_rna = F.cross_entropy(logits, labels, reduction='none')
        loss_atac = F.cross_entropy(logits.transpose(0, 1), labels, reduction='none')
        clip_loss = ((loss_rna + loss_atac) / 2.0).mean()

        # Reconstruction losses
        features_loss1 = torch.sqrt(torch.sum(torch.pow(X1 - X1_, 2)))
        features_loss2 = torch.sqrt(torch.sum(torch.pow(X2 - X2_, 2)))

        # decorrelation loss
        rna_decorr_loss = decorr_loss_correlation(RNA_e)
        atac_decorr_loss = decorr_loss_correlation(ATAC_e)
        decorr_loss = (rna_decorr_loss + atac_decorr_loss) / 2.0 * 100.0 * 0.0

        # total loss
        self.loss = features_loss1 + features_loss2 + clip_loss + decorr_loss
        self.loss_rna = features_loss1
        self.loss_atac = features_loss2
        self.clip_loss = clip_loss

        self.Att_l1 = self.C1
        self.Att_l2 = self.C2
        self.Att_lgp = self.Cgp

        return (
            self.loss,
            self.loss_rna,
            self.loss_atac,
            decorr_loss,
            self.clip_loss,
            self.H1,
            self.H2,
            self.Att_l1,
            self.Att_l2,
            self.Att_lgp,
            X1_,
            X2_,
        )

    def __encoder(self, A, H, W, C, v0, v1, layer):
        H = torch.matmul(H, W[layer])
        if layer == self.n_layers - 1:
            return H

        C[layer] = self.graph_attention_layer(A, H, v0[layer], v1[layer]) # att_ij of Eq. (7)
        return torch.sparse.mm(C[layer], H)

    def __decoder(self, H, W, C, layer):
        H = torch.matmul(H, W[layer].transpose(0, 1))
        if layer == 0:
            return H # Eq. (10)

        return torch.sparse.mm(C[layer - 1], H)

    def __linear_etm_decoder(self, H, alpha, rho):
        x = torch.matmul(H, alpha)
        x = torch.matmul(x, rho)
        x = F.elu(x)
        return x

    def graph_attention_layer(self, A, M, v0, v1):
        A = A.coalesce()
        indices = A.indices()
        row = indices[0]
        col = indices[1]

        f1 = torch.matmul(M, v0).squeeze(-1)
        f2 = torch.matmul(M, v1).squeeze(-1) 
        logits = f1[row] + f2[col]
        e_fg = torch.sigmoid(logits) # Eqs. (2) or (6) in the paper

        weighted_logits = torch.log(torch.clamp(A.values(), min=1e-12)) * e_fg
        unnormalized_attentions = torch.sparse_coo_tensor(
            indices,
            weighted_logits,
            A.shape,
            device=A.device,
        ).coalesce()

        attentions = torch.sparse.softmax(unnormalized_attentions, dim=1)  # att_fg of Eq. (3), or att_ij of Eq. (7)
        return attentions.coalesce()

    def _gp_attention_layer(self, A, v0, v1):
        """GP attention using feature-anchored keys (transferable across cell counts)."""
        A = A.coalesce() # prior feature-feature adjacency matrix, i.e. A_fg in Eq. (4)
        indices = A.indices()
        row = indices[0]
        col = indices[1]

        f1 = v0.squeeze(-1)  # (n_features,) — direct per-feature attention key
        f2 = v1.squeeze(-1)
        logits = f1[row] + f2[col]
        e_fg = torch.sigmoid(logits) # Eq. (2) in the paper

        weighted_logits = torch.log(torch.clamp(A.values(), min=1e-12)) * e_fg
        unnormalized_attentions = torch.sparse_coo_tensor(
            indices,
            weighted_logits,
            A.shape,
            device=A.device,
        ).coalesce()

        return torch.sparse.softmax(unnormalized_attentions, dim=1).coalesce()

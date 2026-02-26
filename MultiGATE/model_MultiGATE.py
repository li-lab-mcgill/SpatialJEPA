import torch
import torch.nn as nn
import torch.nn.functional as F

def _load_legacy_tf_mgate():
    from .model_MultiGATE_legacy_tf import MGATE as legacy_cls

    return legacy_cls


class LegacyTFMGATE(object):
    def __new__(cls, *args, **kwargs):
        legacy_cls = _load_legacy_tf_mgate()
        return legacy_cls(*args, **kwargs)


class MGATE(nn.Module):

    def __init__(self, hidden_dims1, hidden_dims2, spot_num, temp=1.0, nonlinear=True, weight_decay=0.0001):
        super(MGATE, self).__init__()
        self.n_layers = len(hidden_dims1) - 1
        if self.n_layers < 1:
            raise ValueError("hidden_dims must define at least one encoder layer")

        self.nonlinear = nonlinear
        self.weight_decay = weight_decay
        self.hidden_dims1 = hidden_dims1
        self.hidden_dims2 = hidden_dims2
        self.temp = temp

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

        self.vgp0 = nn.Parameter(torch.empty(spot_num, 1))
        self.vgp1 = nn.Parameter(torch.empty(spot_num, 1))

        emb_dim1 = hidden_dims1[-1]
        emb_dim2 = hidden_dims2[-1]
        self.W_i = nn.Parameter(torch.empty(emb_dim1, emb_dim1))
        self.W_t = nn.Parameter(torch.empty(emb_dim2, emb_dim2))
        self.logit_scale = nn.Parameter(torch.tensor(float(temp), dtype=torch.float32))

        self.bn_rna = nn.BatchNorm1d(emb_dim1)
        self.bn_atac = nn.BatchNorm1d(emb_dim2)

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

    def forward(self, A, prune_A, GP, X1, X2):
        del prune_A
        self.C1 = {}
        self.C2 = {}
        self.Cgp = {}

        # Encoder
        H = torch.cat([X1.transpose(0, 1), X2.transpose(0, 1)], dim=0)
        self.Cgp[0] = self.graph_attention_layer(GP, H, self.vgp0, self.vgp1)
        H = torch.sparse.mm(self.Cgp[0], H)
        H = F.relu(H)

        split_point = X1.shape[1]
        H1 = H[:split_point, :].transpose(0, 1).contiguous()
        H2 = H[split_point:, :].transpose(0, 1).contiguous()

        for layer in range(self.n_layers):
            H1 = self.__encoder(A, H1, self.W1, self.C1, self.v1_0, self.v1_1, layer)
            H2 = self.__encoder(A, H2, self.W2, self.C2, self.v2_0, self.v2_1, layer)
            if self.nonlinear and layer != self.n_layers - 1:
                H1 = F.elu(H1)
                H2 = F.elu(H2)

        self.H1 = H1
        self.H2 = H2

        # Decoder
        for layer in range(self.n_layers - 1, -1, -1):
            H1 = self.__decoder(H1, self.W1, self.C1, layer)
            H2 = self.__decoder(H2, self.W2, self.C2, layer)
            if self.nonlinear and layer != 0:
                H1 = F.elu(H1)
                H2 = F.elu(H2)

        H = torch.cat([H1.transpose(0, 1), H2.transpose(0, 1)], dim=0)
        H = torch.sparse.mm(self.Cgp[0], H)
        H = F.elu(H)

        H1_dec = H[:split_point, :].transpose(0, 1).contiguous()
        H2_dec = H[split_point:, :].transpose(0, 1).contiguous()

        X1_ = H1_dec
        X2_ = H2_dec

        # CLIP-style alignment loss
        rna_proj = self.bn_rna(torch.matmul(self.H1, self.W_i))
        atac_proj = self.bn_atac(torch.matmul(self.H2, self.W_t))
        RNA_e = F.normalize(rna_proj, p=2, dim=1)
        ATAC_e = F.normalize(atac_proj, p=2, dim=1)

        logits = torch.matmul(RNA_e, ATAC_e.transpose(0, 1)) * torch.exp(self.logit_scale)
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_rna = F.cross_entropy(logits, labels, reduction='none')
        loss_atac = F.cross_entropy(logits.transpose(0, 1), labels, reduction='none')
        clip_loss = ((loss_rna + loss_atac) / 2.0).mean()

        # Reconstruction losses
        features_loss1 = torch.sqrt(torch.sum(torch.pow(X1 - X1_, 2)))
        features_loss2 = torch.sqrt(torch.sum(torch.pow(X2 - X2_, 2)))

        # Manual weight decay to match previous TensorFlow behavior
        weight_decay_loss = self._weight_decay_penalty()

        self.loss = features_loss1 + features_loss2 + weight_decay_loss + clip_loss
        self.loss_rna = features_loss1
        self.loss_atac = features_loss2
        self.weight_decay_loss = weight_decay_loss
        self.clip_loss = clip_loss

        self.Att_l1 = self.C1
        self.Att_l2 = self.C2
        self.Att_lgp = self.Cgp

        return (
            self.loss,
            self.loss_rna,
            self.loss_atac,
            self.weight_decay_loss,
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

        C[layer] = self.graph_attention_layer(A, H, v0[layer], v1[layer])
        return torch.sparse.mm(C[layer], H)

    def __decoder(self, H, W, C, layer):
        H = torch.matmul(H, W[layer].transpose(0, 1))
        if layer == 0:
            return H

        return torch.sparse.mm(C[layer - 1], H)

    def _weight_decay_penalty(self):
        penalty = self.W_i.new_tensor(0.0)
        for weight in self.W1:
            penalty = penalty + 0.5 * torch.sum(weight * weight)
        for weight in self.W2:
            penalty = penalty + 0.5 * torch.sum(weight * weight)
        penalty = penalty + 0.5 * torch.sum(self.W_i * self.W_i)
        penalty = penalty + 0.5 * torch.sum(self.W_t * self.W_t)
        return penalty * self.weight_decay

    def graph_attention_layer(self, A, M, v0, v1):
        A = A.coalesce()
        indices = A.indices()
        row = indices[0]
        col = indices[1]

        f1 = torch.matmul(M, v0).squeeze(-1)
        f2 = torch.matmul(M, v1).squeeze(-1)
        logits = f1[row] + f2[col]

        weighted_logits = torch.log(torch.clamp(A.values(), min=1e-12)) * torch.sigmoid(logits)
        unnormalized_attentions = torch.sparse_coo_tensor(
            indices,
            weighted_logits,
            A.shape,
            device=A.device,
        ).coalesce()

        attentions = torch.sparse.softmax(unnormalized_attentions, dim=1)
        return attentions.coalesce()

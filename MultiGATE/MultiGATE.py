import numpy as np
import scipy.sparse as sp
import torch
from tqdm import tqdm

from .model_MultiGATE import MGATE


class MultiGATE(object):

    def __init__(
        self,
        hidden_dims1,
        hidden_dims2,
        spot_num,
        temp,
        n_epochs=500,
        lr=0.0001,
        gradient_clipping=5,
        nonlinear=True,
        weight_decay=0.0001,
        verbose=False,
        random_seed=2020,
        config=None,
    ):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        self.loss_list_rna = []
        self.loss_list_atac = []
        self.loss_list = []
        self.loss_list_clip = []
        self.weight_decay_loss_list = []

        self.lr = lr
        self.n_epochs = n_epochs
        self.gradient_clipping = gradient_clipping
        self.verbose = verbose
        self.config = config

        self.device = self._resolve_device(config)
        self.mgate = MGATE(
            hidden_dims1=hidden_dims1,
            hidden_dims2=hidden_dims2,
            spot_num=spot_num,
            temp=temp,
            nonlinear=nonlinear,
            weight_decay=weight_decay,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.mgate.parameters(), lr=self.lr)

    def _resolve_device(self, config):
        if isinstance(config, dict) and config.get("device") is not None:
            return torch.device(config["device"])
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _as_dense_tensor(self, x):
        values = x.values if hasattr(x, "values") else np.asarray(x)
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _as_sparse_tensor(self, graph):
        if isinstance(graph, torch.Tensor):
            if not graph.is_sparse:
                raise TypeError("Expected sparse tensor for graph input")
            return graph.coalesce().to(self.device)

        if sp.isspmatrix(graph):
            graph = graph.tocoo()
            indices = np.vstack((graph.row, graph.col))
            values = graph.data
            shape = graph.shape
        elif isinstance(graph, tuple) and len(graph) == 3:
            indices, values, shape = graph
            indices = np.asarray(indices)
            if indices.ndim != 2:
                raise ValueError("Graph indices must be 2D")
            if indices.shape[0] == 2:
                pass
            elif indices.shape[1] == 2:
                indices = indices.T
            else:
                raise ValueError("Graph indices should have shape (2, E) or (E, 2)")
        else:
            raise TypeError("Unsupported graph format: {}".format(type(graph)))

        indices_t = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        values_t = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        sparse_t = torch.sparse_coo_tensor(indices_t, values_t, torch.Size(shape), device=self.device)
        return sparse_t.coalesce()

    def _prepare_inputs(self, A, prune_A, GP, X1, X2):
        return (
            self._as_sparse_tensor(A),
            self._as_sparse_tensor(prune_A),
            self._as_sparse_tensor(GP),
            self._as_dense_tensor(X1),
            self._as_dense_tensor(X2),
        )

    def __call__(self, A, prune_A, GP, X1, X2):
        A_t, prune_A_t, GP_t, X1_t, X2_t = self._prepare_inputs(A, prune_A, GP, X1, X2)

        with tqdm(total=self.n_epochs, desc="Epoch Progress", unit="epoch") as pbar:
            for epoch in range(self.n_epochs):
                loss = self.run_epoch(epoch, A_t, prune_A_t, GP_t, X1_t, X2_t)
                pbar.update(1)
                if self.verbose:
                    tqdm.write("Epoch: {}, Loss: {:.4f}".format(epoch, loss))

    def run_epoch(self, epoch, A, prune_A, GP, X1, X2):
        del epoch
        if not (isinstance(A, torch.Tensor) and isinstance(X1, torch.Tensor)):
            A, prune_A, GP, X1, X2 = self._prepare_inputs(A, prune_A, GP, X1, X2)

        self.mgate.train()
        self.optimizer.zero_grad()

        outputs = self.mgate(A, prune_A, GP, X1, X2)
        loss, loss_rna, loss_atac, weight_decay_loss, clip_loss = outputs[:5]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mgate.parameters(), self.gradient_clipping)
        self.optimizer.step()

        loss_scalar = float(loss.detach().cpu().item())
        self.loss_list.append(loss_scalar)
        self.loss_list_atac.append(float(loss_atac.detach().cpu().item()))
        self.loss_list_rna.append(float(loss_rna.detach().cpu().item()))
        self.loss_list_clip.append(float(clip_loss.detach().cpu().item()))
        self.weight_decay_loss_list.append(float(weight_decay_loss.detach().cpu().item()))

        return loss_scalar

    def infer(self, A, prune_A, GP, X1, X2):
        A_t, prune_A_t, GP_t, X1_t, X2_t = self._prepare_inputs(A, prune_A, GP, X1, X2)

        self.mgate.eval()
        with torch.no_grad():
            H1, H2, C1, C2, Cgp, ReX1, ReX2 = self.mgate(A_t, prune_A_t, GP_t, X1_t, X2_t)[5:]

        return (
            H1.detach().cpu().numpy(),
            H2.detach().cpu().numpy(),
            self.Conbine_Atten_l(C1),
            self.Conbine_Atten_l(C2),
            self.Conbine_Atten_l(Cgp),
            self.loss_list,
            ReX1.detach().cpu().numpy(),
            ReX2.detach().cpu().numpy(),
        )

    def Conbine_Atten_l(self, input_att):
        attentions = []
        for layer in sorted(input_att):
            tensor = input_att[layer].coalesce()
            idx = tensor.indices().detach().cpu().numpy()
            values = tensor.values().detach().cpu().numpy()
            shape = tuple(tensor.shape)
            attentions.append(sp.coo_matrix((values, (idx[0], idx[1])), shape=shape))
        return attentions

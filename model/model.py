import torch
from torch import nn
from torch.nn import functional as F

from model.maps_builder import MapsBuilder
from model.laplacian_builder import LaplacianBuilder, SparseLaplacianBuilder
from model.decoders import DotProductDecoder, BilinearDecoder, MLPDecoder
from model.preprocessor import Preprocessor
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data


class Diffusion(nn.Module):
    """
    Sheaf Diffusion для линк-прогнозирования с интерфейсом, совместимым с train():
      - encode(x, train_pos_edge_index) -> z
      - decode(z, edge_index) -> logits по рёбрам
      - recon_loss(z, pos_edge_index) -> скаляр
      - test(z, pos_edge_index, neg_edge_index) -> (auc, ap)
    Математика sheaf-Лапласиана сохранена; убран NetworkX.
    """

    def __init__(self, input_dim, hidden_chanels, maps_dim, device, n_layers=2, decoder='dot_product', dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_chanels * maps_dim
        self.input_dim = input_dim
        self.dropout = dropout
        self.n_layers = n_layers
        self.maps_dim = maps_dim

        self.preprocessor = None
        if self.input_dim == None:
            self.input_dim = self.hidden_dim
            self.preprocessor = Preprocessor(
                hidden_dim=self.hidden_dim, device=device).to(device)

        self.first_linear = self.first_linear = nn.Identity() if self.preprocessor else nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.ELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ELU()
        )

        self.maps_builders = nn.ModuleList()
        self.middle_linear = nn.ModuleList()
        self.last_linear = nn.Linear(
            self.hidden_dim, self.hidden_dim, bias=True)

        for _ in range(n_layers):
            self.maps_builders.append(
                MapsBuilder(self.hidden_dim, maps_dim)
            )
            self.middle_linear.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ELU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ELU()
                )
            )

        self.laplacian_builder = SparseLaplacianBuilder(device)
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(n_layers)
        ])
        self.act = nn.ELU()

        # Декодер для линк-прогнозирования
        match decoder:
            case "dot_product":
                self.decoder = DotProductDecoder()
            case "bilinear":
                self.decoder = BilinearDecoder(self.hidden_dim)
            case "mlp":
                self.decoder = MLPDecoder(
                    self.hidden_dim, decoder_dropout=self.dropout)
            case _:
                # по умолчанию оставим dot-product
                self.decoder = DotProductDecoder()

        self.norm = nn.LayerNorm(self.hidden_dim)

    # ---------- ВСПОМОГАТЕЛЬНОЕ: сформировать ориентированные рёбра (i->j) и (j->i) ----------
    @staticmethod
    def _make_oriented_pairs(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Из неориентированных рёбер делаем ориентированные пары [2, 2E]:
        первая половина - (i->j) для i<j, вторая - (j->i) в том же порядке.
        """
        row, col = edge_index
        # выбираем верхнетреугольные ребра (i<j) как уникальные
        mask = row < col
        row_u = row[mask]
        col_u = col[mask]
        # конкатенируем (i->j) и (j->i)
        oriented = torch.cat([torch.stack([row_u, col_u], dim=0),
                              torch.stack([col_u, row_u], dim=0)], dim=1)
        return oriented

    # ---------- API совместимый с train() ----------
    def encode(self, data: Data) -> torch.Tensor:
        """
        Вычисляет эмбеддинги узлов с использованием sheaf-Лапласиана,
        построенного по train графу.
        """
        if self.preprocessor:
            h = self.preprocessor(data.x_dict)
        else:
            h = self.first_linear(data.x)   # [n, hidden]

        pos_edge_index = data.train_pos_edge_index
        num_nodes = h.size(0)

        # строим ориентированный edge_index из train_pos_edge_index
        oriented_edge_index = self._make_oriented_pairs(
            pos_edge_index, num_nodes)

        for layer in range(self.n_layers):
            x_maps = F.dropout(h, p=self.dropout,
                               training=self.training)
            maps = self.maps_builders[layer](
                x_maps, oriented_edge_index)  # [2E, d, d]
            L = self.laplacian_builder(
                maps, oriented_edge_index, num_nodes)  # [(n d),(n d)]

            dx = self.middle_linear[layer](h)  # [n, hidden]
            d = self.maps_dim
            # умножаем L на dx, учитывая блочную структуру
            # dx = torch.matmul(L, dx.reshape(num_nodes * d, -1)
            #   ).reshape(-1, self.hidden_dim)
            dx = torch.sparse.mm(L, dx.reshape(
                num_nodes * d, -1)).reshape(-1, self.hidden_dim)
            h = h - self.alpha[layer] * dx

        h = self.last_linear(h)
        return h  # z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Возвращает логиты по заданным рёбрам edge_index (вектор длины num_edges).
        """
        return self.decoder(z, edge_index)

    def recon_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor) -> torch.Tensor:
        """
        BPR-подобная лог-вероятность для позитивных/негативных рёбер.
        """
        # позитивные
        pos_logit = self.decode(z, pos_edge_index)
        pos_loss = -torch.log(torch.sigmoid(pos_logit) + 1e-15).mean()

        # негативные (семплируем столько же)
        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=z.size(0),
            num_neg_samples=pos_edge_index.size(1),
            force_undirected=True
        )
        neg_logit = self.decode(z, neg_edge_index)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_logit) + 1e-15).mean()

        return pos_loss + neg_loss

    @torch.no_grad()
    def test(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        """
        Считает AUC и AP по заданным наборам рёбер.
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        pos = torch.sigmoid(self.decode(z, pos_edge_index)).cpu().numpy()
        neg = torch.sigmoid(self.decode(z, neg_edge_index)).cpu().numpy()
        y = torch.cat([
            torch.ones(len(pos)),
            torch.zeros(len(neg))
        ]).cpu().numpy()
        scores = torch.tensor(
            list(pos) + list(neg)
        ).cpu().numpy()

        # sklearn ждёт ndarray
        from numpy import concatenate
        scores = concatenate([pos, neg], axis=0)

        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        return float(auc), float(ap)

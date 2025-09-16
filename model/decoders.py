import torch
from torch import nn
from torch.nn import functional as F


class MLPDecoder(nn.Module):
    """
    МЛП-декодер: принимает эмбеддинги узлов z и список рёбер edge_index,
    возвращает логиты для этих рёбер (вектор длины num_edges).
    """

    def __init__(self, hidden_dim, decoder_dropout):
        super().__init__()
        self.decoder_dropout = decoder_dropout
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # edge_index: [2, num_edges]
        row, col = edge_index
        # [num_edges, 2*hidden_dim]
        edges = torch.cat([x[row], x[col]], dim=-1)
        edges = F.elu(self.linear1(edges))
        edges = F.dropout(edges, p=self.decoder_dropout,
                          training=self.training)
        edges = self.linear2(edges).squeeze(-1)     # [num_edges]
        return edges


class DotProductDecoder(nn.Module):
    """
    Скалярное произведение по парам узлов из edge_index.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        return (x[row] * x[col]).sum(dim=1)


class BilinearDecoder(nn.Module):
    """
    Билинейный декодер: z_i^T W z_j по рёбрам из edge_index.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        xW = x @ self.W  # [n, h]
        return (xW[row] * x[col]).sum(dim=1)

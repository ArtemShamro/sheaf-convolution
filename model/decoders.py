import torch
from torch import nn
from torch.nn import functional as F


class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, decoder_dropout):
        super().__init__()
        self.decoder_dropout = decoder_dropout
        self.linear1 = nn.Linear(2 * hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.linear1.weight)
        self.linear2 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x: torch.Tensor):
        row, col = torch.triu_indices(x.shape[0], x.shape[0], offset=1)

        edges = torch.cat([x[row], x[col]], dim=-1)
        edges = F.elu(self.linear1(edges))
        edges = F.dropout(edges, p=self.decoder_dropout)
        edges = self.linear2(edges)

        adj_mat = torch.zeros(x.shape[0], x.shape[0], device=x.device)
        adj_mat[row, col] = edges.squeeze()

        adj_mat = adj_mat + adj_mat.T
        return adj_mat

# class MLPDecoder(nn.Module):
#     def __init__(self, in_dim, dropout=0.0):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim * 2, in_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(in_dim, 1)
#         )
    
#     def forward(self, x, edge_index, node_types):
#         # Filter user-movie edges
#         valid = ((node_types[edge_index[0]] == 0) & (node_types[edge_index[1]] == 1)) | \
#                 ((node_types[edge_index[0]] == 1) & (node_types[edge_index[1]] == 0))
#         edge_index = edge_index[:, valid]
#         x_row = x[edge_index[0]]
#         x_col = x[edge_index[1]]
#         probs = self.mlp(torch.cat([x_row, x_col], dim=1)).squeeze()
#         return probs, valid

class DotProductDecoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        adj_logits = torch.mm(x, x.t())  # [num_nodes, num_nodes]
        return adj_logits  # Вероятности рёбер


class BilinearDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

    def forward(self, x):
        adj_logits = x @ self.W @ x.t()  # [num_nodes, num_nodes]
        return adj_logits

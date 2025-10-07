import torch
from torch import nn


class MapsBuilder(nn.Module):
    """
    Генератор карт R_ij из признаков узлов (через MLP), как было,
    только теперь всегда работаем по edge_index (любого направления).
    """

    def __init__(self, features_dim, maps_dim):
        super().__init__()
        self.maps_dim = maps_dim

        self.mlp = nn.Sequential(
            nn.Linear(features_dim * 2, features_dim * 2),
            nn.ELU(),
            nn.Linear(features_dim * 2, maps_dim ** 2)
        )

    def forward(self, x, edge_index):
        # edge_index: [2, num_edges]
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)
        maps = self.mlp(torch.cat([x_row, x_col], dim=1))  # [num_edges, d*d]
        maps = torch.tanh(maps)
        # [num_edges, d, d]
        return maps.reshape(-1, self.maps_dim, self.maps_dim)

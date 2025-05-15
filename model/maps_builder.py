import torch
from torch import nn


class MapsBuilder(nn.Module):
    def __init__(self, features_dim, maps_dim):
        super().__init__()
        self.maps_dim = maps_dim

        self.mlp = nn.Sequential(
            nn.Linear(features_dim * 2, features_dim * 2),
            nn.ELU(),
            nn.Linear(features_dim * 2, maps_dim ** 2)
        )
        nn.init.xavier_uniform_(self.mlp[-1].weight)

        # self.linear = nn.Linear(features_dim * 2, maps_dim ** 2, bias=False)
        # nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index):
        row, col = edge_index
        x_row = torch.index_select(x, dim=0, index=row)
        x_col = torch.index_select(x, dim=0, index=col)

        maps = self.mlp(torch.cat([x_row, x_col], dim=1))

        # maps = self.linear(torch.cat([x_row, x_col], dim=1))
        maps = torch.tanh(maps)
        return maps.reshape(-1, self.maps_dim, self.maps_dim)

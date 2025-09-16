import torch.nn as nn
import torch
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data, HeteroData


class Preprocessor(nn.Module):
    def __init__(self, hidden_dim: int, device: str):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.encoders = nn.ModuleDict()

    def build_encoders(self, x_dict):
        for node_type, x in x_dict.items():
            in_dim = x.size(1)
            self.encoders[node_type] = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ELU(),
            ).to(self.device)

    def forward(self, x_dict):
        if not self.encoders:
            self.build_encoders(x_dict)

        out = []
        for node_type, x in x_dict.items():
            x = x.to(self.device)
            out.append(self.encoders[node_type](x))
        return torch.cat(out, dim=0)  # объединяем пользователей и фильмы

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import networkx as nx
from config.model_config import ModelDiffusionConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import networkx as nx


class VGAE(nn.Module):
    def __init__(self, config: ModelDiffusionConfig, hidden_dim1: int = 32, hidden_dim2: int = 16):
        """
        Variational Graph Auto-Encoder (VGAE) для восстановления рёбер.

        Args:
            input_dim (int): Размерность входных эмбеддингов вершин.
            hidden_dim1 (int): Размерность первого скрытого слоя (default: 32).
            hidden_dim2 (int): Размерность второго скрытого слоя (default: 16).
        """
        super().__init__()
        # Энкодер: два GCN слоя для mu и logvar
        self.conv1 = GCNConv(config.input_dim, hidden_dim1)
        self.conv_mu = GCNConv(hidden_dim1, hidden_dim2)  # Для среднего (mu)
        self.conv_logvar = GCNConv(
            hidden_dim1, hidden_dim2)  # Для log(variance)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Энкодер: вычисляет mu и logvar для латентных представлений.

        Args:
            x (torch.Tensor): Эмбеддинги вершин, форма (num_nodes, input_dim).
            edge_index (torch.Tensor): Индексы рёбер, форма (2, num_edges).

        Returns:
            mu (torch.Tensor): Средние латентных распределений, форма (num_nodes, hidden_dim2).
            logvar (torch.Tensor): Логарифмы дисперсий, форма (num_nodes, hidden_dim2).
        """
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Репараметризация: сэмплирует z из нормального распределения N(mu, sigma).

        Args:
            mu (torch.Tensor): Средние, форма (num_nodes, hidden_dim2).
            logvar (torch.Tensor): Логарифмы дисперсий, форма (num_nodes, hidden_dim2).

        Returns:
            z (torch.Tensor): Сэмплированные латентные представления, форма (num_nodes, hidden_dim2).
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Декодер: вычисляет логиты рёбер через скалярное произведение.

        Args:
            z (torch.Tensor): Латентные представления, форма (num_nodes, hidden_dim2).

        Returns:
            logits (torch.Tensor): Логиты рёбер, форма (num_nodes, num_nodes).
        """
        return torch.matmul(z, z.t())  # Скалярное произведение: z_i^T z_j

    def forward(self, x: torch.Tensor, G: nx.Graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Прямой проход модели.

        Args:
            x (torch.Tensor): Эмбеддинги вершин, форма (num_nodes, input_dim).
            G (nx.Graph): Граф в формате NetworkX.

        Returns:
            logits (torch.Tensor): Логиты рёбер, форма (num_nodes, num_nodes).
            mu (torch.Tensor): Средние латентных распределений, форма (num_nodes, hidden_dim2).
            logvar (torch.Tensor): Логарифмы дисперсий, форма (num_nodes, hidden_dim2).
        """
        # Преобразуем nx.Graph в edge_index
        edge_index = torch.tensor(
            list(G.edges()), dtype=torch.long, device=x.device).t()
        # Убедимся, что граф неориентированный
        edge_index = to_undirected(edge_index)

        # Энкодер
        mu, logvar = self.encode(x, edge_index)
        # Репараметризация
        z = self.reparameterize(mu, logvar)
        # Декодер
        logits = self.decode(z)

        return logits, mu, logvar

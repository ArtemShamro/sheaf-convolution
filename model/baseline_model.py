import torch
import torch.nn.functional as F
import torch.nn as nn
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
from model.decoders import BilinearDecoder
from config.model_config import ModelDiffusionConfig


class GAE(nn.Module):
    def __init__(self, config: ModelDiffusionConfig, hidden_dim1: int = 32, hidden_dim2: int = 16):
        """
        Graph Auto-Encoder (GAE) для восстановления рёбер.

        Args:
            config (ModelDiffusionConfig): Конфигурация модели, содержащая input_dim.
            hidden_dim1 (int): Размерность первого скрытого слоя (default: 32).
            hidden_dim2 (int): Размерность второго скрытого слоя (default: 16).
        """
        super().__init__()
        # Энкодер: два GCN слоя
        self.conv1 = GCNConv(config.input_dim, hidden_dim1)
        # Для скрытых представлений
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Энкодер: вычисляет скрытые представления вершин.

        Args:
            x (torch.Tensor): Эмбеддинги вершин, форма (num_nodes, input_dim).
            edge_index (torch.Tensor): Индексы рёбер, форма (2, num_edges).

        Returns:
            z (torch.Tensor): Скрытые представления, форма (num_nodes, hidden_dim2).
        """
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Декодер: вычисляет логиты рёбер через скалярное произведение.

        Args:
            z (torch.Tensor): Скрытые представления, форма (num_nodes, hidden_dim2).

        Returns:
            logits (torch.Tensor): Логиты unwitting рёбер, форма (num_nodes, num_nodes).
        """
        return torch.matmul(z, z.t())  # Скалярное произведение: z_i^T z_j
        # return self.decoder(z)

    def forward(self, x: torch.Tensor, G: nx.Graph) -> torch.Tensor:
        """
        Прямой проход модели.

        Args:
            x (torch.Tensor): Эмбеддинги вершин, форма (num_nodes, input_dim).
            G (nx.Graph): Граф в формате NetworkX.

        Returns:
            logits (torch.Tensor): Логиты рёбер, форма (num_nodes, num_nodes).
        """
        # Преобразуем nx.Graph в edge_index
        edge_index = torch.tensor(
            list(G.edges()), dtype=torch.long, device=x.device).t()
        # Убедимся, что граф неориентированный
        edge_index = to_undirected(edge_index)

        # Энкодер
        z = self.encode(x, edge_index)
        # Декодер
        logits = self.decode(z)

        return logits

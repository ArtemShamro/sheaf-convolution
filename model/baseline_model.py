import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score

from model.preprocessor import Preprocessor


class GAE(nn.Module):
    def __init__(self, input_dim, device,  hidden_dim1: int = 32, hidden_dim2: int = 16, dropout=0.2):
        """
        Graph Auto-Encoder (GAE) для линк-прогнозирования.

        Args:
            hidden_dim1 (int): Размерность первого скрытого слоя (default: 32).
            hidden_dim2 (int): Размерность второго скрытого слоя (default: 16).
        """
        super().__init__()
        self.preprocessor, self.first_linear = None, None
        self.dropout = dropout

        if input_dim is None:
            input_dim = hidden_dim1
            self.preprocessor = Preprocessor(
                hidden_dim=hidden_dim1, device=device).to(device)
        else:
            self.first_linear = self.first_linear = nn.Identity() if self.preprocessor else nn.Sequential(
                nn.Linear(input_dim, hidden_dim1 * 2),
                nn.ELU(),
                # nn.Dropout(self.dropout),
                nn.Linear(hidden_dim1 * 2, hidden_dim1),
                nn.ELU()
            )

        self.conv1 = GCNConv(hidden_dim1, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)

        self.last_linear = nn.Linear(
            hidden_dim2, hidden_dim2, bias=True)

    def encode(self, data) -> torch.Tensor:
        """Энкодер: вычисляет скрытые представления вершин."""
        if self.preprocessor:
            x = self.preprocessor(data.x_dict)
        else:
            x = self.first_linear(data.x)  # type: ignore

        x = F.relu(self.conv1(x, data.train_pos_edge_index))
        z = self.conv2(x, data.train_pos_edge_index)

        z = self.last_linear(z)

        return z

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Декодер: скалярное произведение для рёбер."""
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)

    def recon_loss(self, z: torch.Tensor, pos_edge_index: torch.Tensor) -> torch.Tensor:
        """Функция потерь: бинарная кросс-энтропия по позитивным и негативным рёбрам."""
        pos_loss = - \
            torch.log(torch.sigmoid(self.decode(
                z, pos_edge_index)) + 1e-15).mean()

        neg_edge_index = negative_sampling(
            edge_index=pos_edge_index,
            num_nodes=z.size(0),
            num_neg_samples=pos_edge_index.size(1),
        )
        neg_loss = -torch.log(
            1 - torch.sigmoid(self.decode(z, neg_edge_index)) + 1e-15
        ).mean()

        return pos_loss + neg_loss

    @torch.no_grad()
    def test(self, z: torch.Tensor, pos_edge_index: torch.Tensor, neg_edge_index: torch.Tensor):
        """Вычисление AUC и AP."""
        pos_y = torch.ones(pos_edge_index.size(1))
        neg_y = torch.zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y]).cpu().numpy()

        preds = torch.cat([
            torch.sigmoid(self.decode(z, pos_edge_index)),
            torch.sigmoid(self.decode(z, neg_edge_index))
        ]).cpu().numpy()

        return roc_auc_score(y, preds), average_precision_score(y, preds)


import torch
from torch import nn

from utils import build_masks


class LaplacianBuilder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    # def forward(self, adj_mat, degrees, maps, edge_index):
    #     d = maps.shape[1]
    #     num_edges = maps.shape[0] // 2
    #     num_nodes = adj_mat.shape[0]
    #     maps_dim = maps.shape[-1]

    #     L = torch.zeros(num_nodes, num_nodes, maps_dim,
    #                     maps_dim, device=self.device)

    #     mask_diag, mask_ul, mask_triu, mask_tril = build_masks(adj_mat)

    #     # diag elements
    #     adj_mat_maps = torch.zeros(
    #         num_nodes, num_nodes, maps_dim, maps_dim, device=self.device)
    #     adj_mat_maps[mask_ul] = torch.bmm(torch.transpose(maps, 1, 2), maps)
    #     maps_diag = adj_mat_maps.sum(dim=1)

    #     # Triu elements
    #     maps_from = maps[:num_edges]
    #     maps_to = maps[num_edges:]
    #     maps_triu = torch.bmm(torch.transpose(maps_from, 1, 2), maps_to)

    #     # Normalization
    #     row, col = edge_index[:, :num_edges]
    #     degrees_inv_sqrt = (degrees * d + 1).pow(-0.5).reshape(-1, 1, 1)
    #     left_norm = degrees_inv_sqrt[row]
    #     right_norm = degrees_inv_sqrt[col]
    #     maps_triu = left_norm * maps_triu * right_norm
    #     maps_diag = degrees_inv_sqrt * maps_diag * degrees_inv_sqrt

    #     # Tril elements
    #     maps_tril = torch.transpose(maps_triu, 1, 2)

    #     L[mask_triu] = -maps_triu
    #     L[mask_tril] = -maps_tril
    #     L[mask_diag] = maps_diag

    #     L = L.permute(0, 2, 1, 3).reshape(
    #         num_nodes * maps_dim, num_nodes * maps_dim)

    #     return L
    def forward(self, adj_mat, degrees, maps, edge_index):
        """
        adj_mat:   [n, n] (используем только n)
        degrees:   [n]    (не нужен для матричной нормализации)
        maps:      [2E, d, d] (ориентированные карты: первая половина и обратные)
        edge_index:[2, 2E]    (ориентированный)
        """
        device = self.device
        n = adj_mat.size(0)
        d = maps.size(1)
        twoE = maps.size(0)
        assert twoE % 2 == 0, "maps должен содержать пары ориентированных рёбер (2E)."
        E = twoE // 2

        # --- 1) Диагональные блоки D_i = sum_{e: i->*} R_e^T R_e ---
        # Складываем по ВСЕМ ориентированным рёбрам исходящим из i (row_all)
        row_all = edge_index[0]                  # [2E]
        diag_contrib = torch.bmm(maps.transpose(1, 2), maps)  # [2E, d, d]

        maps_diag = torch.zeros(n, d, d, device=device)
        # index_add_ для 3D: делаем через вью
        maps_diag_flat = maps_diag.view(n, -1)
        maps_diag_flat.index_add_(0, row_all, diag_contrib.reshape(twoE, -1))
        maps_diag = maps_diag_flat.view(n, d, d)

        # --- 2) Вне диагональные блоки для верхнего треугольника ---
        # Берём первую половину пар (i->j) как "triu"-набор, вторую половину считаем обратной (j->i)
        row, col = edge_index[:, :E]             # [E], [E]
        left_maps = maps[:E]                    # R_{ij}
        right_maps = maps[E:]                    # R_{ji}
        # Блоки L_{ij} = - R_{ij}^T R_{ji} до нормализации
        maps_triu = torch.bmm(left_maps.transpose(1, 2),
                              right_maps)  # [E, d, d]

        # --- 3) Матричная нормализация (normalized sheaf Laplacian) ---
        # augmentation + джиттер (как в эталонной реализации)
        if self.training:
            eps = torch.empty(d, device=device).uniform_(-1e-3, 1e-3)
            I_aug = torch.diag(1.0 + eps).unsqueeze(0)  # [1,d,d]
        else:
            I_aug = torch.eye(d, device=device).unsqueeze(0)

        to_inv = maps_diag + I_aug                 # [n,d,d], SPD
        evals, evecs = torch.linalg.eigh(to_inv)   # стабильный симм. корень
        inv_sqrt = evecs @ torch.diag_embed(evals.clamp_min(
            1e-8).pow(-0.5)) @ evecs.transpose(-1, -2)  # [n,d,d]

        # Нормируем off-diag и diag
        left_norm = inv_sqrt[row]                 # [E,d,d]
        right_norm = inv_sqrt[col]                 # [E,d,d]
        maps_triu = left_norm @ maps_triu @ right_norm
        maps_diag = inv_sqrt @ maps_diag @ inv_sqrt

        # (опционально) ограничим значения как в исходной реализации
        maps_triu = maps_triu.clamp(min=-1, max=1)
        maps_diag = maps_diag.clamp(min=-1, max=1)

        # --- 4) Собираем полный плотный L \in R^{(n d) \times (n d)} ---
        L = torch.zeros(n, n, d, d, device=device)

        # Диагональ
        idx = torch.arange(n, device=device)
        L[idx, idx] = maps_diag

        # Верхний/нижний треугольник
        L[row, col] = -maps_triu
        L[col, row] = -maps_triu.transpose(1, 2)   # симметричная часть

        # Разворачиваем блоки: (i,d) x (j,d)
        L = L.permute(0, 2, 1, 3).reshape(n * d, n * d)
        return L

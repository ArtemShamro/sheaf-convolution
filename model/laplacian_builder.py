import torch
from torch import nn


class LaplacianBuilder(nn.Module):
    """
    Сборка нормализованного sheaf-Лапласиана.
    Ожидает ориентированный edge_index размера [2, 2E], где первая половина — (i->j),
    вторая — соответствующие обратные (j->i) в том же порядке.
    """

    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, maps, edge_index, num_nodes: int):
        """
        maps:       [2E, d, d]   (первые E — R_ij, вторые E — R_ji)
        edge_index: [2, 2E]      (ориентированный)
        num_nodes:  int
        """
        device = self.device
        n = num_nodes
        d = maps.size(1)
        twoE = maps.size(0)
        assert twoE % 2 == 0, "maps должен содержать пары ориентированных рёбер (2E)."
        E = twoE // 2

        # 1) Диагональные блоки D_i = sum_{e: i->*} R_e^T R_e
        row_all = edge_index[0]                                  # [2E]
        diag_contrib = torch.bmm(maps.transpose(1, 2), maps)     # [2E, d, d]

        maps_diag = torch.zeros(n, d, d, device=device)
        maps_diag_flat = maps_diag.view(n, -1)
        maps_diag_flat.index_add_(0, row_all, diag_contrib.reshape(twoE, -1))
        maps_diag = maps_diag_flat.view(n, d, d)

        # 2) Внедиагональные блоки для верхнего треугольника
        row, col = edge_index[:, :E]              # [E], [E]
        left_maps = maps[:E]                      # R_ij
        right_maps = maps[E:]                     # R_ji
        maps_triu = torch.bmm(left_maps.transpose(1, 2),
                              right_maps)  # [E, d, d]

        # 3) Матричная нормализация
        if self.training:
            eps = torch.empty(d, device=device).uniform_(-1e-3, 1e-3)
            I_aug = torch.diag(1.0 + eps).unsqueeze(0)
        else:
            I_aug = torch.eye(d, device=device).unsqueeze(0)

        to_inv = maps_diag + I_aug
        evals, evecs = torch.linalg.eigh(to_inv)
        inv_sqrt = evecs @ torch.diag_embed(
            evals.clamp_min(1e-8).pow(-0.5)) @ evecs.transpose(-1, -2)

        left_norm = inv_sqrt[row]      # [E,d,d]
        right_norm = inv_sqrt[col]     # [E,d,d]
        maps_triu = left_norm @ maps_triu @ right_norm
        maps_diag = inv_sqrt @ maps_diag @ inv_sqrt

        maps_triu = maps_triu.clamp(min=-1, max=1)
        maps_diag = maps_diag.clamp(min=-1, max=1)

        # 4) Собираем плотный L \in R^{(n d) \times (n d)} из блоков
        L = torch.zeros(n, n, d, d, device=device)

        idx = torch.arange(n, device=device)
        L[idx, idx] = maps_diag
        L[row, col] = -maps_triu
        L[col, row] = -maps_triu.transpose(1, 2)

        L = L.permute(0, 2, 1, 3).reshape(n * d, n * d)
        return L

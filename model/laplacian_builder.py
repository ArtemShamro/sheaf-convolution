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


class SparseLaplacianBuilder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, maps, edge_index, num_nodes: int):
        """
        maps:       [2E, d, d] (первые E — R_ij, вторые E — R_ji)
        edge_index: [2, 2E]    (ориентированный; вторые E — обратные)
        num_nodes:  int
        """
        device = self.device
        n = num_nodes
        d = maps.size(1)
        twoE = maps.size(0)
        assert twoE % 2 == 0, "maps должен содержать пары ориентированных рёбер (2E)."
        E = twoE // 2

        # ----- Диагональные блоки: D_i = sum_{e: i->*} R_e^T R_e -----
        row_all = edge_index[0]                                     # [2E]
        diag_contrib = torch.bmm(maps.transpose(
            1, 2), maps)        # [2E, d, d]
        maps_diag = torch.zeros(n, d, d, device=device)
        maps_diag.view(n, -1).index_add_(0, row_all,
                                         diag_contrib.reshape(twoE, -1))
        maps_diag = maps_diag  # [n, d, d]

        # ----- Внедиагональные блоки (верхний треугольник): -R_ij^T R_ji -----
        row, col = edge_index[:, :E]       # [E], [E]
        left_maps = maps[:E]              # [E, d, d] (R_ij)
        right_maps = maps[E:]              # [E, d, d] (R_ji)
        maps_triu = torch.bmm(left_maps.transpose(1, 2),
                              right_maps)  # [E, d, d]

        # ----- Матричная нормализация (твоя логика сохранена) -----
        if self.training:
            eps = torch.empty(d, device=device).uniform_(-1e-3, 1e-3)
            I_aug = torch.diag(1.0 + eps).unsqueeze(0)
        else:
            I_aug = torch.eye(d, device=device).unsqueeze(0)

        to_inv = maps_diag + I_aug
        # [n,d], [n,d,d]
        inv_sqrt = self.matrix_inv_sqrt_newton_schulz(to_inv)

        # [E,d,d]
        left_norm = inv_sqrt[row]
        # [E,d,d]
        right_norm = inv_sqrt[col]
        maps_triu = (left_norm @ maps_triu @
                     right_norm).clamp(min=-1, max=1)    # [E,d,d]
        maps_diag = (inv_sqrt  @ maps_diag @
                     inv_sqrt).clamp(min=-1, max=1)     # [n,d,d]

        # ================== ВЕКТОРИЗОВАННАЯ СБОРКА CSR ==================
        # Подготавливаем индексы в блоке d×d
        a = torch.arange(d, device=device)
        b = torch.arange(d, device=device)
        A, B = torch.meshgrid(a, b, indexing="ij")  # [d, d]

        # ---- Диагональные элементы ----
        diag_rows = (torch.arange(n, device=device)[
                     :, None, None] * d + A).reshape(-1)
        diag_cols = (torch.arange(n, device=device)[
                     :, None, None] * d + B).reshape(-1)
        diag_vals = maps_diag.reshape(-1)

        # ---- Внедиагональные блоки (i,j) и (j,i) ----
        tri_rows_ij = (row[:, None, None] * d + A).reshape(-1)
        tri_cols_ij = (col[:, None, None] * d + B).reshape(-1)
        tri_vals_ij = (-maps_triu).reshape(-1)

        tri_rows_ji = (col[:, None, None] * d + A).reshape(-1)
        tri_cols_ji = (row[:, None, None] * d + B).reshape(-1)
        tri_vals_ji = (-maps_triu.transpose(1, 2)).reshape(-1)

        # ---- Склеиваем всё в один список ненулевых элементов ----
        rows = torch.cat([diag_rows, tri_rows_ij, tri_rows_ji], dim=0)
        cols = torch.cat([diag_cols, tri_cols_ij, tri_cols_ji], dim=0)
        vals = torch.cat([diag_vals, tri_vals_ij, tri_vals_ji], dim=0)

        # ---- Преобразуем в CSR ----
        size = n * d

        # сортировка по строкам
        perm = torch.argsort(rows)
        rows = rows[perm]
        cols = cols[perm]
        vals = vals[perm]

        # crow_indices: накопленная сумма количества элементов в строках
        # row_counts = torch.bincount(rows, minlength=size)
        row_counts = torch.zeros(size, device=device, dtype=torch.long)
        row_counts.index_add_(0, rows, torch.ones_like(
            rows, device=device, dtype=torch.long))
        crow_indices = torch.cat([
            torch.zeros(1, device=device, dtype=torch.long),
            torch.cumsum(row_counts, dim=0)
        ])

        # создаём CSR Laplacian
        L = torch.sparse_csr_tensor(crow_indices, cols, vals,
                                    size=(size, size), device=device)

        return L

    def batched_sym_matrix_pow(self, A, p):
        evals, evecs = torch.linalg.eigh(A)
        return evecs @ torch.diag_embed(evals.clamp_min(1e-8).pow(p)) @ evecs.transpose(-1, -2)

    def matrix_inv_sqrt_newton_schulz(self, A, num_iters=5):
        normA = A.mul(A).sum(dim=(-2, -1), keepdim=True).sqrt()
        Y = A / normA
        I = torch.eye(A.size(-1), device=A.device).expand_as(A)
        Z = torch.eye(A.size(-1), device=A.device).expand_as(A)
        for _ in range(num_iters):
            T = 0.5 * (3*I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
        return Z / normA.sqrt()


import torch
from torch import nn

from utils import build_masks


class LaplacianBuilder(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, adj_mat, degrees, maps, edge_index):
        d = maps.shape[1]
        num_edges = maps.shape[0] // 2
        num_nodes = adj_mat.shape[0]
        maps_dim = maps.shape[-1]

        L = torch.zeros(num_nodes, num_nodes, maps_dim,
                        maps_dim, device=self.device)

        mask_diag, mask_ul, mask_triu, mask_tril = build_masks(adj_mat)

        # diag elements
        adj_mat_maps = torch.zeros(
            num_nodes, num_nodes, maps_dim, maps_dim, device=self.device)
        adj_mat_maps[mask_ul] = torch.bmm(torch.transpose(maps, 1, 2), maps)
        maps_diag = adj_mat_maps.sum(dim=1)

        # Triu elements
        maps_from = maps[:num_edges]
        maps_to = maps[num_edges:]
        maps_triu = torch.bmm(torch.transpose(maps_from, 1, 2), maps_to)

        # Normalization
        row, col = edge_index[:, :num_edges]
        degrees_inv_sqrt = (degrees * d + 1).pow(-0.5).reshape(-1, 1, 1)
        left_norm = degrees_inv_sqrt[row]
        right_norm = degrees_inv_sqrt[col]
        maps_triu = left_norm * maps_triu * right_norm
        maps_diag = degrees_inv_sqrt * maps_diag * degrees_inv_sqrt

        # Tril elements
        maps_tril = torch.transpose(maps_triu, 1, 2)

        L[mask_triu] = -maps_triu
        L[mask_tril] = -maps_tril
        L[mask_diag] = maps_diag

        L = L.permute(0, 2, 1, 3).reshape(
            num_nodes * maps_dim, num_nodes * maps_dim)

        return L

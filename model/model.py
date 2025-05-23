from config.model_config import ModelDiffusionConfig
import torch
from torch import nn
from torch.nn import functional as F
import networkx as nx

from model.maps_builder import MapsBuilder
from model.laplacian_builder import LaplacianBuilder
from model.decoders import MLPDecoder, DotProductDecoder, BilinearDecoder
from utils import build_masks, get_adj_mat, get_degrees_and_edges

import wandb


class Diffusion(nn.Module):
    def __init__(self, config: ModelDiffusionConfig):
        """
        input_dim - num features
        output_dim - num classes
        """
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_chanels * config.maps_dim

        # input dim -> hidden_dim
        self.first_linear = nn.Linear(
            config.input_dim, self.hidden_dim, bias=True)
        nn.init.uniform_(self.first_linear.weight, a=-1, b=1)

        self.maps_builders = nn.ModuleList()
        self.middle_linear = nn.ModuleList()

        for layer in range(config.n_layers):
            self.maps_builders.append(
                MapsBuilder(self.hidden_dim, config.maps_dim)
            )
            self.middle_linear.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ELU(),
                    # nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                )
            )

        self.laplacian_builder = LaplacianBuilder(config.device)
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(0.5)) for _ in range(config.n_layers)
        ])
        self.act = nn.ELU()

        match config.task, config.decoder:
            case "edges_prediction", "mlp":
                self.decoder = MLPDecoder(
                    self.hidden_dim, config.decoder_dropout)
            case "edges_prediction", "dot_product":
                self.decoder = DotProductDecoder()
            case "edges_prediction", "bilinear":
                self.decoder = BilinearDecoder(self.hidden_dim)
            case "node_classification":
                self.decoder = nn.Sequential(
                    nn.Linear(self.hidden_dim, config.output_dim),
                    nn.ELU(),
                    nn.Linear(config.output_dim, config.output_dim)
                )

        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, G: nx.Graph):

        adj_mat = get_adj_mat(G, self.config.device)
        degrees, edge_index = get_degrees_and_edges(G, self.config.device)

        num_nodes = x.shape[0]
        x = self.first_linear(x)

        # ADDED
        x = self.norm(x)

        x = self.act(x)

        for layer in range(self.config.n_layers):
            x_maps = F.dropout(x, p=self.config.dropout)
            maps = self.maps_builders[layer](x_maps, edge_index)
            laplacian = self.laplacian_builder(
                adj_mat, degrees, maps, edge_index)

            # print(torch.norm(maps, dim=(1, 2)).mean())

            dx = self.middle_linear[layer](x)

            # Добавлено
            dx = nn.LayerNorm(self.hidden_dim, device=self.config.device)(dx)

            dx = torch.matmul(laplacian, dx.reshape(
                num_nodes * self.config.maps_dim, -1)).reshape(-1, self.hidden_dim)
            x = x - self.alpha[layer] * dx

        # x = nn.LayerNorm(self.hidden_dim, device=self.config.device)(x)

        # print(x.detach())
        # print("x mean:", x.mean().item())
        # print("x std:", x.std().item())
        # print("x min:", x.min().item())
        # print("x max:", x.max().item())
        # print("x L2 norm:", torch.norm(x, p=2, dim=1).detach())
        # wandb.log({"x_distribution": wandb.Histogram(
        #     x.cpu().detach().flatten(), num_bins=50)})
        x = self.decoder(x)
        return x

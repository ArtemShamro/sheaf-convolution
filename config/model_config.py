import torch
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelDiffusionConfig():
    
    
    input_dim: int
    hidden_chanels: int
    maps_dim: int
    output_dim: int
    device: torch.device
    A: torch.Tensor
    degrees: torch.Tensor
    edge_index: torch.Tensor


    type: Literal["diffusion", "VGAE"] = "diffusion"
    task: Literal["edges_prediction",
                  "node_classification"] = "edges_prediction"
    decoder: str = "dot_product"
    n_epochs: int = 300
    weight_decay: float = 1e-5
    dropout: float = 0
    decoder_dropout: float = 0
    n_layers: int = 2

import torch
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class ModelDiffusionConfig():
    type: Literal["diffusion", "VGAE"] = "diffusion"
    task: Literal["edges_prediction",
                  "node_classification"] = "edges_prediction"
    decoder: str = "dot_product"
    n_epochs: int = 300
    weight_decay: float = 1e-5
    dropout: float = 0
    decoder_dropout: float = 0
    n_layers: int = 2
    input_dim: Optional[int] = None
    hidden_chanels: Optional[int] = None
    maps_dim: Optional[int] = None
    output_dim: Optional[int] = None
    device: Optional[torch.device] = None
    A: Optional[torch.tensor] = None
    degrees: Optional[torch.tensor] = None
    edge_index: Optional[torch.tensor] = None

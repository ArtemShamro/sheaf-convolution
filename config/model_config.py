import torch
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class ModelDiffusionConfig:
    # === из YAML ===
    hidden_chanels: int = 64   # <-- fix naming
    maps_dim: int = 32
    output_dim: Optional[int] = None
    device_str: str = "cpu"     # "cpu" | "cuda:0" в YAML
    type: Literal["diffusion", "VGAE"] = "diffusion"
    task: Literal["edges_prediction", "node_classification"] = "edges_prediction"
    decoder: str = "dot_product"
    n_epochs: int = 300
    weight_decay: float = 1e-5
    dropout: float = 0.0
    decoder_dropout: float = 0.0
    n_layers: int = 2

    # === рантайм-поля (не из YAML) ===
    input_dim: int = field(init=False, repr=False)
    device: torch.device = field(init=False, repr=False)
    A: Optional[torch.Tensor] = field(default=None, repr=False)
    degrees: Optional[torch.Tensor] = field(default=None, repr=False)
    edge_index: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        self.device = torch.device(self.device_str)

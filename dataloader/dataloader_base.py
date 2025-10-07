# dataset_base.py
from abc import ABC, abstractmethod
from typing import Tuple
import torch


class BaseDataset(ABC):
    """Базовый класс для всех датасетов."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.data = None

    @abstractmethod
    def get_data(self) -> Tuple[torch.Tensor, int]:
        """Возвращает (data, num_features)."""
        pass

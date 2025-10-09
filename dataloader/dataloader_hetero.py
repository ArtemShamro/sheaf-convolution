# dataset_hetero.py
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.utils import train_test_split_edges
from dataloader import BaseDataset


class HeteroGraphDataset(BaseDataset):
    """Загрузка гетерофильных графов из Yandex Research."""

    def __init__(self, name: str, val_ratio=0.5, test_ratio=0.15, device: str = "cuda", model='GAE', **kwargs):
        super().__init__(device)
        self.name = name
        self.model = model
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio

    def get_data(self):
        dataset = HeterophilousGraphDataset(
            root=f"/tmp/{self.name}",
            name=self.name,
            transform=T.NormalizeFeatures()
        )
        data = dataset[0]

        # Разделяем рёбра на train/val/test
        data = train_test_split_edges(
            data, val_ratio=self.val_ratio, test_ratio=self.test_ratio)

        # Переносим на устройство
        for key, value in data:
            if isinstance(value, torch.Tensor):
                data[key] = value.to(self.device, non_blocking=True)
        # data.x = torch.randn(data.num_nodes, 32, device=data.x.device)
        if self.model == "Diffusion":
            data = self._add_random_noise(data)
            print("Random noise added")
        self.data = data
        return data, dataset.num_node_features

    def _add_random_noise(self, data, scale=0.2):
        noise = torch.randn_like(data.x) * scale
        data.x = data.x + noise
        return data

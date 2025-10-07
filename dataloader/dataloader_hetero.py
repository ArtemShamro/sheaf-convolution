# dataset_hetero.py
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset
from torch_geometric.utils import train_test_split_edges
from dataloader import BaseDataset


class HeteroGraphDataset(BaseDataset):
    """Загрузка гетерофильных графов из Yandex Research."""

    def __init__(self, name: str, val_ratio=0.5, test_ration=0.15, device: str = "cpu", **kwargs):
        super().__init__(device)
        self.name = name
        self.test_ratio = test_ration
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
        data.x = data.x.to(self.device)
        data.train_pos_edge_index = data.train_pos_edge_index.to(self.device)
        data.val_pos_edge_index = data.val_pos_edge_index.to(self.device)
        data.test_pos_edge_index = data.test_pos_edge_index.to(self.device)

        if data.val_neg_edge_index is not None:
            data.val_neg_edge_index = data.val_neg_edge_index.to(self.device)
        if data.test_neg_edge_index is not None:
            data.test_neg_edge_index = data.test_neg_edge_index.to(self.device)
        self.data = data
        return data, dataset.num_node_features

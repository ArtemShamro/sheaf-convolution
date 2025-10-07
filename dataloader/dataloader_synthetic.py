# dataset_synthetic.py
import os
import pickle
import hashlib
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, train_test_split_edges
from dataloader.dataloader_base import BaseDataset
from utils import get_mask_edge_prediction


class SyntheticDataset(BaseDataset):
    """Генерация синтетического графа (без изменения математики)."""

    def __init__(
        self,
        ndata: int = 1000,
        dimx: int = 4,
        test_ratio: float = 0.15,
        val_ratio: float = 0.05,
        s_threshold: float = 0.2,
        nproj: int = 4,
        neg_ratio: int = 1,
        device: str = "cpu",
        seed=None,
        **kwargs
    ):
        super().__init__(device)
        self.ndata = ndata
        self.dimx = dimx
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.s_threshold = s_threshold
        self.nproj = nproj
        self.neg_ratio = neg_ratio
        self.seed = seed

    # ------------------------ Основной API ------------------------
    def get_data(self):
        """Возвращает синтетический граф (Data, num_node_features)."""
        G_train, data_x, adj_mat, _, _ = self._generate_synthetic_data()

        # --- Полный граф по матрице смежности ---
        if isinstance(adj_mat, torch.Tensor):
            adj_np = adj_mat.cpu().numpy()
        else:
            adj_np = adj_mat
        G_full = nx.from_numpy_array(adj_np)

        # --- Конвертация в PyG Data ---
        data = from_networkx(G_full)
        data.x = data_x.clone()

        # --- Train/Val/Test split ---
        data = train_test_split_edges(
            data, val_ratio=self.val_ratio, test_ratio=self.test_ratio
        )

        # --- Перенос на устройство ---
        data.x = data.x.to(self.device)
        data.train_pos_edge_index = data.train_pos_edge_index.to(self.device)
        data.val_pos_edge_index = data.val_pos_edge_index.to(self.device)
        data.test_pos_edge_index = data.test_pos_edge_index.to(self.device)
        if data.val_neg_edge_index is not None:
            data.val_neg_edge_index = data.val_neg_edge_index.to(self.device)
        if data.test_neg_edge_index is not None:
            data.test_neg_edge_index = data.test_neg_edge_index.to(self.device)

        self.data = data

        num_node_features = int(data.x.size(1))
        return data, num_node_features

    # ------------------------ Реализация генерации ------------------------
    def _generate_synthetic_data(self):
        """Полностью повторяет generate_synthetic_data из synthetic.py."""
        save_dir = "./datasets/synthetic"
        os.makedirs(save_dir, exist_ok=True)

        params = {
            'ndata': self.ndata,
            'dimx': self.dimx,
            'test_size': self.test_ratio,
            's_threshold': self.s_threshold,
            'nproj': self.nproj,
            'seed': self.seed
        }
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        save_path = os.path.join(save_dir, f"graph_{param_hash}.pkl")

        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
            return data['G'], data['data_x'], data['adj_mat'], data['train_mask'], data['test_mask']

        # --- Генерация нового графа ---
        data_x, _, adj_mat, _ = self._generate_graph(
            self.ndata, self.dimx, self.s_threshold, self.nproj, nvec=2
        )
        adj_mat -= np.diag(np.ones(self.ndata))
        G = nx.from_numpy_array(adj_mat)
        adj_mat = torch.tensor(adj_mat)

        G_train, train_mask, val_mask, test_mask = get_mask_edge_prediction(
            G,
            test_size=self.test_ratio,
            val_size=self.val_ratio,
            neg_ratio=self.neg_ratio,
            device=self.device,
        )

        data = {
            'G': G_train,
            'data_x': data_x,
            'adj_mat': adj_mat,
            'train_mask': train_mask,
            'test_mask': test_mask,
            'val_mask': val_mask,
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)

        return G_train, data_x, adj_mat, train_mask, test_mask

    # ------------------------ Вспомогательные методы ------------------------
    @staticmethod
    def _generate_data(ndata, dimy):
        ydata = torch.normal(0.0, 1.0, (ndata, dimy))
        return F.normalize(ydata, dim=1).numpy()

    @staticmethod
    def _perform_orthogonalization(vmat):
        smat = vmat.copy()
        nvec, dimy = smat.shape
        for idx0 in range(nvec):
            for idx1 in range(idx0):
                smat[idx0, :] = smat[idx0, :] - \
                    np.dot(smat[idx0, :], smat[idx1, :]) * smat[idx1, :]
            smat[idx0, :] = smat[idx0, :] / \
                np.sqrt(np.dot(smat[idx0, :], smat[idx0, :]))
        return smat

    @staticmethod
    def _generate_projection_matrix(nvec, dimx):
        return SyntheticDataset._perform_orthogonalization(
            np.random.normal(0.0, 1.0, (nvec, dimx))
        )

    @staticmethod
    def _compute_projection(xdata, smat):
        return np.dot(xdata, np.dot(np.transpose(smat), smat))

    @staticmethod
    def _generate_graph(ndata, dimx, s_threshold=0.2, nproj=4, nvec=2):
        print('Generating synthetic graph...')
        xdata = SyntheticDataset._generate_data(ndata, dimx)
        wmat = np.zeros((ndata, ndata))
        proj_mat_data = np.zeros((nproj, nvec, dimx))
        xproj = np.zeros((nproj, ndata, dimx))

        for idx in range(nproj):
            proj_mat_data[idx, :, :] = SyntheticDataset._generate_projection_matrix(
                nvec, dimx)
            xproj[idx, :, :] = SyntheticDataset._compute_projection(
                xdata, proj_mat_data[idx, :, :])

        for idx0 in tqdm(range(ndata)):
            for idx1 in range(ndata):
                for idx in range(nproj):
                    dx = xproj[idx, idx1, :] - xproj[idx, idx0, :]
                    s = np.sqrt(np.sum(dx * dx))
                    if s < s_threshold:
                        wmat[idx0, idx1] = 1.0

        return (
            torch.from_numpy(xdata).float(),
            torch.from_numpy(proj_mat_data).float(),
            wmat,
            torch.from_numpy(xproj).float(),
        )

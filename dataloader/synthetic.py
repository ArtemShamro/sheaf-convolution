from scipy.sparse import csr_matrix
import requests
from scipy.sparse.csgraph import shortest_path
from networkx import adjacency_matrix
import networkx as nx
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from typing import Tuple
from utils import get_mask_edge_prediction
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import hashlib
import os

import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, train_test_split_edges


def generate_synthetic_pyg(
    ndata: int,
    dimx: int,
    device: str = "cpu",
    test_size: float = 0.15,
    val_size: float = 0.05,
    s_threshold: float = 0.2,
    nproj: int = 4,
    # оставляем для совместимости сигнатуры (не используется train_test_split_edges)
    neg_ratio: int = 1,
):
    """
    Генерация синтетического графа с тем же ИНТЕРФЕЙСОМ возврата, что и generate_planetoid_data:
      -> return data, num_node_features

    Математику построения графа НЕ меняем: используем твою generate_synthetic_data,
    берём исходную (полную) смежность adj_mat и по ней строим PyG Data,
    затем делаем стандартный сплит рёбер через train_test_split_edges.
    """

    # 1) Сгенерировать/загрузить граф твоим исходным кодом (математика остаётся прежней)
    #    ВОЗВРАЩАЕТ: G_train (train-граф), data_x (признаки), adj_mat (ПОЛНАЯ смежность), ...
    G_train, data_x, adj_mat, _train_mask, _test_mask = generate_synthetic_data(
        ndata=ndata,
        dimx=dimx,
        test_size=test_size,
        val_size=val_size,
        neg_ratio=neg_ratio,
        s_threshold=s_threshold,
        nproj=nproj,
        device=device,
    )

    # 2) ВАЖНО: для корректного сплита используем ПОЛНУЮ смежность (до масок),
    #    чтобы train/val/test разбивались так же, как в Planetoid-пайплайне.
    if isinstance(adj_mat, torch.Tensor):
        adj_np = adj_mat.cpu().numpy()
    else:
        adj_np = adj_mat
    G_full = nx.from_numpy_array(adj_np)  # неориентированный граф

    # 3) Переводим в PyG Data, добавляем признаки
    # создаёт data.edge_index и data.num_nodes
    data = from_networkx(G_full)
    data.x = data_x.clone()               # [num_nodes, dimx]

    # 4) Делаем стандартный сплит рёбер (как в твоём образце для Planetoid)
    data = train_test_split_edges(
        data, val_ratio=val_size, test_ratio=test_size)

    # 5) Переносим всё нужное на устройство (один-в-один как в образце)
    data.x = data.x.to(device)
    data.train_pos_edge_index = data.train_pos_edge_index.to(device)
    data.val_pos_edge_index = data.val_pos_edge_index.to(device)
    data.test_pos_edge_index = data.test_pos_edge_index.to(device)
    if data.val_neg_edge_index is not None:
        data.val_neg_edge_index = data.val_neg_edge_index.to(device)
    if data.test_neg_edge_index is not None:
        data.test_neg_edge_index = data.test_neg_edge_index.to(device)

    num_node_features = int(data.x.size(1))
    return data, num_node_features


def generate_synthetic_data(ndata, dimx, test_size=0.15, val_size=0.05, neg_ratio=1, s_threshold=0.2, nproj=4, device='cpu'):
    """
    Generate or load synthetic graph data, saving it to ./datasets/synthetic.

    Args:
        ndata (int): Number of nodes
        dimx (int): Feature dimension
        task (str): Task type (e.g., 'classification')
        test_size (float): Proportion of edges for test set
        s_threshold (float): Similarity threshold for graph generation
        nproj (int): Number of projections for graph generation

    Returns:
        G: NetworkX graph
        data_x: Node features
        data_y: Node labels
        train_mask: Training mask
        test_mask: Test mask
    """
    # Create directory if it doesn't exist
    save_dir = "./datasets/synthetic"
    os.makedirs(save_dir, exist_ok=True)

    # Create unique filename based on parameters
    params = {
        'ndata': ndata,
        'dimx': dimx,
        'test_size': test_size,
        's_threshold': s_threshold,
        'nproj': nproj
    }
    param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    save_path = os.path.join(save_dir, f"graph_{param_hash}.pkl")

    # Check if data exists
    if os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        return data['G'], data['data_x'], data['adj_mat'], data['train_mask'], data['test_mask']

    # Generate new data
    data_x, _, adj_mat, _ = generate_graph(
        ndata, dimx, s_threshold, nproj, nvec=2)
    adj_mat -= np.diag(np.ones(ndata))
    G = nx.from_numpy_array(adj_mat)
    adj_mat = torch.tensor(adj_mat)
    G_train, train_mask, val_mask, test_mask = get_mask_edge_prediction(
        G, test_size=test_size, val_size=val_size, neg_ratio=neg_ratio, device=device)

    # Save data
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


def generate_data(ndata, dimy):
    ydata = torch.normal(0.0, 1.0, (ndata, dimy))
    return F.normalize(ydata, dim=1).numpy()


def perform_orthogonalization(vmat):
    """ Gram–Schmidt process for 2 #nvec random vectors"""
    smat = vmat.copy()
    nvec, dimy = smat.shape
    for idx0 in range(nvec):
        for idx1 in range(idx0):
            smat[idx0, :] = smat[idx0, :] - \
                np.dot(smat[idx0, :], smat[idx1, :]) * smat[idx1, :]
        smat[idx0, :] = smat[idx0, :] / \
            np.sqrt(np.dot(smat[idx0, :], smat[idx0, :]))
    return smat


def generate_projection_matrix(nvec, dimx):
    return perform_orthogonalization(np.random.normal(0.0, 1.0, (nvec, dimx)))


def compute_projection(xdata, smat):
    return np.dot(xdata, np.dot(np.transpose(smat), smat))


def compute_distance_matrix(ydata):
    ndata, dimy = ydata.shape
    smat = np.zeros((ndata, ndata))
    for idx0 in range(ndata):
        for idx1 in range(ndata):
            dy = ydata[idx1, :] - ydata[idx0, :]
            smat[idx0, idx1] = np.sqrt(np.sum(dy * dy))
    return smat


def generate_graph(ndata, dimx, s_threshold=0.2, nproj=4, nvec=2):
    print('Generating graph...')
    xdata = generate_data(ndata, dimx)

    # adjacency mat
    wmat = np.zeros((ndata, ndata))

    proj_mat_data = np.zeros((nproj, nvec, dimx))

    xproj = np.zeros((nproj, ndata, dimx))

    for idx in range(nproj):
        proj_mat_data[idx, :, :] = generate_projection_matrix(nvec, dimx)
        xproj[idx, :, :] = compute_projection(xdata, proj_mat_data[idx, :, :])

    for idx0 in tqdm(range(ndata)):
        for idx1 in range(ndata):
            for idx in range(nproj):
                dx = xproj[idx, idx1, :] - xproj[idx, idx0, :]
                s = np.sqrt(np.sum(dx * dx))
                if s < s_threshold:
                    wmat[idx0, idx1] = 1.0

    return (torch.from_numpy(xdata).float(),
            torch.from_numpy(proj_mat_data).float(),
            wmat,
            torch.from_numpy(xproj).float())

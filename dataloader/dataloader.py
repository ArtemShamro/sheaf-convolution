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


def generate_dataset(name: str = 'Simple', task: str = 'classification', test_size: float = 0.2,
                     ndata: int = 1000, dimx: int = 10) -> nx.Graph:
    """
    task - 'classification' / 'edges_prediction'
    """
    match name:
        case 'Simple':
            return generate_simple_data()
        case 'Synthetic':
            return generate_synthetic_data(ndata, dimx, task, test_size)
        case 'Cora':
            return generate_cora_data(task, test_size)
        case _:
            raise ValueError(f'Unknown data name: {name}')


def generate_simple_data():
    G_sim = nx.erdos_renyi_graph(n=4, p=0.5)
    data_x, data_y = generate_features_and_labels(4, 2, num_classes=2)
    mask = torch.ones_like(torch.tensor(nx.to_numpy_array(G_sim))).bool()
    return G_sim, data_x, data_y, mask, mask


def generate_synthetic_data(ndata, dimx, task, test_size=0.2, s_threshold=0.2, nproj=4):
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
        'task': task,
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
        return data['G'], data['data_x'], data['data_y'], data['train_mask'], data['test_mask']

    # Generate new data
    data_x, _, adj_mat, _ = generate_graph(
        ndata, dimx, s_threshold, nproj, nvec=2)
    adj_mat -= np.diag(np.ones(ndata))
    G = nx.from_numpy_array(adj_mat)
    _, data_y = generate_features_and_labels(ndata, 1, num_classes=1)
    train_mask, test_mask = get_mask_edge_prediction(G, test_size=test_size)

    # Save data
    data = {
        'G': G,
        'data_x': data_x,
        'data_y': data_y,
        'train_mask': train_mask,
        'test_mask': test_mask,
        'params': params
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    return G, data_x, data_y, train_mask, test_mask


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


def generate_cora_data(task: str = 'classification', test_size: float = 0.2) -> Tuple[nx.Graph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    PL = Planetoid(root="./datasets/Planetoid/", name="Cora",
                   transform=T.NormalizeFeatures())
    graph = PL[0]

    # # Additional normalization steps
    # # 1. Normalize features to unit norm
    # features = graph.x
    # features = torch.nn.functional.normalize(features, p=2, dim=1)

    # # 2. Scale features to [-1, 1] range
    # features = 2 * (features - features.min()) / \
    #     (features.max() - features.min()) - 1

    # # 3. Add small epsilon to avoid zero values
    # features = features + 1e-6

    # # Update graph features
    # graph.x = features

    G = to_networkx(graph, to_undirected=True)
    if task == 'classification':
        return G, graph.x, graph.y, ~graph.test_mask, graph.test_mask
    train_mask, test_mask = get_mask_edge_prediction(G, test_size=test_size)
    return G, graph.x, graph.y, train_mask, test_mask


def generate_features_and_labels(num_nodes, dim_features, num_classes=2):
    data_x = torch.rand(size=(num_nodes, dim_features),
                        dtype=torch.float) * 2 - 1
    data_x = torch.nn.functional.normalize(data_x, dim=1)
    data_y = torch.randint(0, num_classes, size=(num_nodes, 1))
    return data_x, data_y

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
from collections import defaultdict



def generate_dataset(name: str = 'Simple', test_size: float = 0.2,
                     ndata: int = 1000, dimx: int = 10, device='cpu') -> nx.Graph:
    match name:
        case 'Synthetic':
            return generate_synthetic_data(ndata, dimx, test_size)
        case 'Cora' | 'Citeseer' | 'PubMed':
            return generate_planetoid_data(dataset=name, test_size=test_size)
        case 'movielens':
            return generate_movielens_data(test_size, device=device)
        case 'movielens1M':
            return generate_movielens_1m_data(test_size, device=device)
        case 'PPI':
            return generate_ppi_data(test_size, device=device)
        case _:
            raise ValueError(f'Unknown data name: {name}')


def generate_simple_data():
    G_sim = nx.erdos_renyi_graph(n=4, p=0.5)
    data_x, data_y = generate_features_and_labels(4, 2, num_classes=2)
    mask = torch.ones_like(torch.tensor(nx.to_numpy_array(G_sim))).bool()
    return G_sim, data_x, data_y, mask, mask


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


def generate_planetoid_data(dataset: str = 'Cora', test_size: float = 0.1, val_size: float = 0.05, device: str = 'cpu', neg_ratio: float = 1.0):
    dataset = Planetoid(root="./datasets/Planetoid/",
                        name=dataset, transform=T.NormalizeFeatures())
    graph = dataset[0]
    G = to_networkx(graph, to_undirected=True)
    adj_mat = torch.tensor(adjacency_matrix(G).toarray())

    G_train, train_mask, val_mask, test_mask = get_mask_edge_prediction(
        G, test_size=test_size, val_size=val_size, neg_ratio=neg_ratio, device=device)

    return G_train, graph.x.to(device), adj_mat, train_mask, test_mask


def generate_features_and_labels(num_nodes, dim_features, num_classes=2):
    data_x = torch.rand(size=(num_nodes, dim_features),
                        dtype=torch.float) * 2 - 1
    data_x = torch.nn.functional.normalize(data_x, dim=1)
    data_y = torch.randint(0, num_classes, size=(num_nodes, 1))
    return data_x, data_y

from torch_geometric.datasets import MovieLens100K
def generate_movielens_data(test_size: float = 0.1, 
                           val_size: float = 0.05, 
                           device: str = 'cpu', 
                           neg_ratio: float = 1.0, 
                           data_dir: str = "./data"):
    """
    Loads and processes the MovieLens100K dataset from PyTorch Geometric,
    transforming it into a homogeneous graph. Handles different feature dimensions
    for users and movies with separate normalization. Saves the graph to disk with
    edge weights and loads it if already created.

    Parameters:
    -----------
    test_size : float
        Fraction of edges for test set. Default: 0.1
    val_size : float
        Fraction of edges for validation set. Default: 0.05
    device : str
        Device to place tensors ('cpu' or 'cuda'). Default: 'cpu'
    neg_ratio : float
        Ratio of negative to positive edges for masks. Default: 1.0
    data_dir : str
        Directory to store dataset and graph. Default: "./data"

    Returns:
    --------
    G_train : networkx.Graph
        Training graph with train edges
    node_features : torch.Tensor
        Node features tensor
    adj_mat : torch.Tensor
        Dense adjacency matrix of the graph
    train_mask : torch.Tensor
        Boolean mask for training edges
    test_mask : torch.Tensor
        Boolean mask for test edges
    """
    # File name for saved graph
    graph_dir = os.path.join(data_dir, "graphs")
    graph_file = "movielens100k.npz"
    graph_path = os.path.join(graph_dir, graph_file)

    # Check if graph exists
    if os.path.exists(graph_path):
        print(f"Loading existing graph from {graph_path}...")
        try:
            data = np.load(graph_path)
            node_features = torch.tensor(data['node_features'], dtype=torch.float)
            labels = torch.tensor(data['node_labels'], dtype=torch.long)
            edges = torch.tensor(data['edges'], dtype=torch.long)
            edge_weights = torch.tensor(data['edge_weights'], dtype=torch.float)
        except Exception as e:
            raise RuntimeError(f"Failed to load graph {graph_path}: {e}")
    else:
        # Load MovieLens100K
        print(f"Loading MovieLens100K dataset...")
        dataset_pyg = MovieLens100K(root=data_dir)
        data = dataset_pyg[0]  # HeteroData object

        # Extract user and movie features
        user_features = data['user'].x
        movie_features = data['movie'].x
        edge_index = data['user', 'rates', 'movie'].edge_index
        edge_label = data['user', 'rates', 'movie'].edge_label

        # Normalize features separately
        user_features = user_features / (user_features.norm(dim=1, keepdim=True) + 1e-10)
        movie_features = movie_features / (movie_features.norm(dim=1, keepdim=True) + 1e-10)

        # Align feature dimensions
        max_dim = max(user_features.shape[1], movie_features.shape[1])
        if user_features.shape[1] < max_dim:
            user_features = torch.cat([user_features, torch.zeros(user_features.shape[0], max_dim - user_features.shape[1])], dim=1)
        if movie_features.shape[1] < max_dim:
            movie_features = torch.cat([movie_features, torch.zeros(movie_features.shape[0], max_dim - movie_features.shape[1])], dim=1)

        # Add type indicator (0 for users, 1 for movies)
        user_type = torch.zeros(user_features.shape[0], 1)
        movie_type = torch.ones(movie_features.shape[0], 1)
        user_features = torch.cat([user_features, user_type], dim=1)
        movie_features = torch.cat([movie_features, movie_type], dim=1)

        # Combine features
        num_users = user_features.shape[0]
        num_movies = movie_features.shape[0]
        node_features = torch.cat([user_features, movie_features], dim=0)

        # Create labels (0 for users, 1 for movies)
        labels = torch.cat([torch.zeros(num_users, dtype=torch.long), torch.ones(num_movies, dtype=torch.long)])

        # Create edges with renumbered indices
        edges = edge_index.t().clone()
        edges[:, 1] += num_users  # Shift movie indices

        # Create edge weights (ratings)
        edge_weights = edge_label.clone()

        # Add reverse edges for undirected graph
        edges_reverse = edges[:, [1, 0]]
        edge_weights_reverse = edge_weights.clone()
        edges = torch.cat([edges, edges_reverse], dim=0)
        edge_weights = torch.cat([edge_weights, edge_weights_reverse], dim=0)

        # Remove duplicate edges, averaging weights
        unique_edges, indices = torch.unique(edges, dim=0, return_inverse=True)
        edge_weights = torch.zeros(unique_edges.shape[0], dtype=torch.float)
        for i in range(unique_edges.shape[0]):
            mask = indices == i
            edge_weights[i] = edge_weights[mask].mean()
        edges = unique_edges

        # Save graph
        os.makedirs(graph_dir, exist_ok=True)
        print(f"Saving graph to {graph_path}...")
        np.savez(graph_path, 
                 node_features=node_features.numpy(),
                 node_labels=labels.numpy(),
                 edges=edges.numpy(),
                 edge_weights=edge_weights.numpy())

    # Create NetworkX graph from edges
    num_nodes = node_features.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges.tolist())

    # Create dense adjacency matrix
    adj_mat = torch.tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float).to(device)

    # Normalize features (additional normalization for consistency)
    node_features = node_features / (node_features.sum(dim=1, keepdim=True) + 1e-10)
    node_features = node_features.to(device)

    # Generate edge masks
    G_train, train_mask, val_mask, test_mask = get_mask_edge_prediction(
        G, test_size=test_size, val_size=val_size, neg_ratio=neg_ratio, device=device)

    return G_train, node_features, adj_mat, train_mask, test_mask


from torch_geometric.datasets import MovieLens1M

def generate_movielens_1m_data(test_size: float = 0.1, 
                               val_size: float = 0.05, 
                               device: str = 'cpu', 
                               neg_ratio: float = 1.0, 
                               data_dir: str = "./data"):
    """
    Loads and processes the MovieLens1M dataset from PyTorch Geometric,
    transforming it into a homogeneous graph. Handles different feature dimensions
    for users and movies with separate normalization. Saves the graph to disk with
    edge weights and loads it if already created.

    Parameters:
    -----------
    test_size : float
        Fraction of edges for test set. Default: 0.1
    val_size : float
        Fraction of edges for validation set. Default: 0.05
    device : str
        Device to place tensors ('cpu' or 'cuda'). Default: 'cpu'
    neg_ratio : float
        Ratio of negative to positive edges for masks. Default: 1.0
    data_dir : str
        Directory to store dataset and graph. Default: "./data"

    Returns:
    --------
    G_train : networkx.Graph
        Training graph with train edges
    node_features : torch.Tensor
        Node features tensor
    adj_mat : torch.Tensor
        Dense adjacency matrix of the graph
    train_mask : torch.Tensor
        Boolean mask for training edges
    test_mask : torch.Tensor
        Boolean mask for test edges
    """
    # File name for saved graph
    graph_dir = os.path.join(data_dir, "graphs")
    graph_file = "movielens1m.npz"
    graph_path = os.path.join(graph_dir, graph_file)

    # Check if graph exists
    if os.path.exists(graph_path):
        print(f"Loading existing graph from {graph_path}...")
        try:
            data = np.load(graph_path)
            node_features = torch.tensor(data['node_features'], dtype=torch.float)
            labels = torch.tensor(data['node_labels'], dtype=torch.long)
            edges = torch.tensor(data['edges'], dtype=torch.long)
            edge_weights = torch.tensor(data['edge_weights'], dtype=torch.float)
        except Exception as e:
            raise RuntimeError(f"Failed to load graph {graph_path}: {e}")
    else:
        # Load MovieLens1M
        print(f"Loading MovieLens1M dataset...")
        dataset_pyg = MovieLens1M(root=data_dir)
        data = dataset_pyg[0]  # HeteroData object

        # Extract user and movie features
        user_features = data['user'].x
        movie_features = data['movie'].x
        edge_index = data['user', 'rates', 'movie'].edge_index
        edge_label = data['user', 'rates', 'movie'].edge_label

        # Normalize features separately
        user_features = user_features / (user_features.norm(dim=1, keepdim=True) + 1e-10)
        movie_features = movie_features / (movie_features.norm(dim=1, keepdim=True) + 1e-10)

        # Align feature dimensions
        max_dim = max(user_features.shape[1], movie_features.shape[1])
        if user_features.shape[1] < max_dim:
            user_features = torch.cat([user_features, torch.zeros(user_features.shape[0], max_dim - user_features.shape[1])], dim=1)
        if movie_features.shape[1] < max_dim:
            movie_features = torch.cat([movie_features, torch.zeros(movie_features.shape[0], max_dim - movie_features.shape[1])], dim=1)

        # Add type indicator (0 for users, 1 for movies)
        user_type = torch.zeros(user_features.shape[0], 1)
        movie_type = torch.ones(movie_features.shape[0], 1)
        user_features = torch.cat([user_features, user_type], dim=1)
        movie_features = torch.cat([movie_features, movie_type], dim=1)

        # Combine features
        num_users = user_features.shape[0]
        num_movies = movie_features.shape[0]
        node_features = torch.cat([user_features, movie_features], dim=0)

        # Create labels (0 for users, 1 for movies)
        labels = torch.cat([torch.zeros(num_users, dtype=torch.long), torch.ones(num_movies, dtype=torch.long)])

        # Create edges with renumbered indices
        edges = edge_index.t().clone()
        edges[:, 1] += num_users  # Shift movie indices

        # Create edge weights (ratings)
        edge_weights = edge_label.clone()

        # Add reverse edges for undirected graph
        edges_reverse = edges[:, [1, 0]]
        edge_weights_reverse = edge_weights.clone()
        edges = torch.cat([edges, edges_reverse], dim=0)
        edge_weights = torch.cat([edge_weights, edge_weights_reverse], dim=0)

        # Remove duplicate edges, averaging weights
        unique_edges, indices = torch.unique(edges, dim=0, return_inverse=True)
        edge_weights = torch.zeros(unique_edges.shape[0], dtype=torch.float)
        for i in range(unique_edges.shape[0]):
            mask = indices == i
            edge_weights[i] = edge_weights[mask].mean()
        edges = unique_edges

        # Save graph
        os.makedirs(graph_dir, exist_ok=True)
        print(f"Saving graph to {graph_path}...")
        np.savez(graph_path, 
                 node_features=node_features.numpy(),
                 node_labels=labels.numpy(),
                 edges=edges.numpy(),
                 edge_weights=edge_weights.numpy())

    # Create NetworkX graph from edges
    num_nodes = node_features.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges.tolist())

    # Create dense adjacency matrix
    adj_mat = torch.tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float).to(device)

    # Normalize features (additional normalization for GAE compatibility)
    node_features = node_features / (node_features.sum(dim=1, keepdim=True) + 1e-10)
    node_features = node_features.to(device)

    # Generate edge masks
    G_train, train_mask, val_mask, test_mask = get_mask_edge_prediction(
        G, test_size=test_size, val_size=val_size, neg_ratio=neg_ratio, device=device)

    return G_train, node_features, adj_mat, train_mask, test_mask


from torch_geometric.datasets import PPI
from sklearn.cluster import KMeans

def generate_ppi_data(test_size: float = 0.1, 
                      val_size: float = 0.05, 
                      device: str = 'cpu', 
                      neg_ratio: float = 1.0, 
                      data_dir: str = "./data"):
    """
    Loads and processes the PPI dataset (first training graph) from PyTorch Geometric,
    transforming it into a homogeneous graph. Normalizes node features and adds a node type
    indicator based on label clustering. Saves the graph to disk and loads it if already created.

    Parameters:
    -----------
    test_size : float
        Fraction of edges for test set. Default: 0.1
    val_size : float
        Fraction of edges for validation set. Default: 0.05
    device : str
        Device to place tensors ('cpu' or 'cuda'). Default: 'cpu'
    neg_ratio : float
        Ratio of negative to positive edges for masks. Default: 1.0
    data_dir : str
        Directory to store dataset and graph. Default: "./data"

    Returns:
    --------
    G_train : networkx.Graph
        Training graph with train edges
    node_features : torch.Tensor
        Node features tensor
    adj_mat : torch.Tensor
        Dense adjacency matrix of the graph
    train_mask : torch.Tensor
        Boolean mask for training edges
    test_mask : torch.Tensor
        Boolean mask for test edges
    """
    # File name for saved graph
    graph_dir = os.path.join(data_dir, "graphs")
    graph_file = "ppi.npz"
    graph_path = os.path.join(graph_dir, graph_file)

    # Check if graph exists
    if os.path.exists(graph_path):
        print(f"Loading existing graph from {graph_path}...")
        try:
            data = np.load(graph_path)
            node_features = torch.tensor(data['node_features'], dtype=torch.float)
            labels = torch.tensor(data['node_labels'], dtype=torch.long)
            edges = torch.tensor(data['edges'], dtype=torch.long)
        except Exception as e:
            raise RuntimeError(f"Failed to load graph {graph_path}: {e}")
    else:
        # Load PPI (first training graph)
        print(f"Loading PPI dataset...")
        dataset_pyg = PPI(root=data_dir, split='train')
        data = dataset_pyg[0]  # First training graph

        # Extract node features
        node_features = data.x  # Shape: [num_nodes, 50]

        # Normalize features
        node_features = node_features / (node_features.norm(dim=1, keepdim=True) + 1e-10)

        # Create node type indicator using KMeans clustering on labels
        labels_binary = data.y.cpu().numpy()  # Shape: [num_nodes, 121]
        kmeans = KMeans(n_clusters=5, random_state=42)
        node_types = kmeans.fit_predict(labels_binary)
        node_types = torch.tensor(node_types, dtype=torch.float).reshape(-1, 1)
        
        # Combine features with type indicator
        node_features = torch.cat([node_features, node_types], dim=1)

        # Create labels (node types from clustering)
        labels = torch.tensor(node_types, dtype=torch.long).squeeze()

        # Create edges
        edges = data.edge_index.t().clone()

        # Add reverse edges for undirected graph
        edges_reverse = edges[:, [1, 0]]
        edges = torch.cat([edges, edges_reverse], dim=0).unique(dim=0)

        # Save graph
        os.makedirs(graph_dir, exist_ok=True)
        print(f"Saving graph to {graph_path}...")
        np.savez(graph_path, 
                 node_features=node_features.numpy(),
                 node_labels=labels.numpy(),
                 edges=edges.numpy())

    # Create NetworkX graph from edges
    num_nodes = node_features.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges.tolist())

    # Create dense adjacency matrix
    adj_mat = torch.tensor(nx.adjacency_matrix(G).toarray(), dtype=torch.float).to(device)

    # Normalize features (additional normalization for GAE compatibility)
    node_features = node_features / (node_features.sum(dim=1, keepdim=True) + 1e-10)
    node_features = node_features.to(device)

    # Generate edge masks
    G_train, train_mask, val_mask, test_mask = get_mask_edge_prediction(
        G, test_size=test_size, val_size=val_size, neg_ratio=neg_ratio, device=device)

    return G_train, node_features, adj_mat, train_mask, test_mask
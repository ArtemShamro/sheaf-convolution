from torch_geometric.utils import train_test_split_edges
from torch_geometric.datasets import Planetoid
from dataloader.synthetic import generate_synthetic_pyg
import torch_geometric.transforms as T
from dataloader.other import generate_movielens100k, generate_heterophilous_graph


def generate_dataset(name: str = 'Cora', device: str = 'cpu'):
    """
    name : str ['Cora' 'Synthetic' 'Movielens100K']
    """
    match name:
        case 'Cora' | 'Citeseer' | 'PubMed':
            return generate_planetoid_data(name=name, device=device)
        case 'Synthetic':
            return generate_synthetic_pyg(1000, 4, device=device)
        case 'Movielens100K':
            return generate_movielens100k(device=device)
        case 'roman-empire' | 'amazon-ratings' | 'minesweeper' | 'tolokers' | 'questions':
            return generate_heterophilous_graph(name=name, device=device)
        case _:
            raise ValueError(f'Unknown data name: {name}')


def generate_planetoid_data(name: str = 'Cora', device: str = 'cpu'):
    dataset = Planetoid(root=f"/tmp/{name}",
                        name=name, transform=T.NormalizeFeatures())
    data = dataset[0]

    data = train_test_split_edges(data)

    data.x = data.x.to(device)
    data.train_pos_edge_index = data.train_pos_edge_index.to(device)
    data.val_pos_edge_index = data.val_pos_edge_index.to(device)
    data.test_pos_edge_index = data.test_pos_edge_index.to(device)

    if data.val_neg_edge_index is not None:
        data.val_neg_edge_index = data.val_neg_edge_index.to(device)
    if data.test_neg_edge_index is not None:
        data.test_neg_edge_index = data.test_neg_edge_index.to(device)

    return data, dataset.num_node_features

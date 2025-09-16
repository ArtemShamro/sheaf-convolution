import torch
from torch_geometric.datasets import MovieLens100K
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges, to_undirected
import torch_geometric.transforms as T
from torch_geometric.datasets import HeterophilousGraphDataset


def generate_movielens100k(device: str = "cpu"):
    dataset = MovieLens100K(root="/tmp/MovieLens100K",
                            transform=T.NormalizeFeatures())
    hetero_data = dataset[0]  # HeteroData

    num_users = hetero_data['user'].x.size(0)
    num_movies = hetero_data['movie'].x.size(0)

    # сдвигаем индексы фильмов, чтобы они не пересекались с пользователями
    edge_index = hetero_data['user', 'rates', 'movie'].edge_index.clone()
    edge_index[1, :] += num_users

    # делаем неориентированный граф
    edge_index = to_undirected(edge_index)

    # создаём Data только с edge_index, без x
    data = Data(edge_index=edge_index, num_nodes=num_users + num_movies)

    # объединяем признаки в dict (передадим в Preprocessor)
    data.x_dict = {
        'user': hetero_data['user'].x.to(device),
        'movie': hetero_data['movie'].x.to(device),
    }

    # train/val/test сплит
    data = train_test_split_edges(data)

    # переносим на устройство
    data.train_pos_edge_index = data.train_pos_edge_index.to(device)
    data.val_pos_edge_index = data.val_pos_edge_index.to(device)
    data.test_pos_edge_index = data.test_pos_edge_index.to(device)
    if data.val_neg_edge_index is not None:
        data.val_neg_edge_index = data.val_neg_edge_index.to(device)
    if data.test_neg_edge_index is not None:
        data.test_neg_edge_index = data.test_neg_edge_index.to(device)

    return data, None


def generate_heterophilous_graph(name: str, device: str = "cpu"):
    """
    Загружает один из датасетов:
    ['roman-empire', 'amazon-ratings', 'minesweeper', 'tolokers', 'questions'].
    Возвращает (Data, dataset) в стиле generate_planetoid_data.
    """
    dataset = HeterophilousGraphDataset(
        root=f"/tmp/{name}",
        name=name,
        transform=T.NormalizeFeatures()
    )
    data = dataset[0]

    # Разделяем рёбра на train/val/test
    data = train_test_split_edges(data)

    # Переносим на устройство
    data.x = data.x.to(device)
    data.train_pos_edge_index = data.train_pos_edge_index.to(device)
    data.val_pos_edge_index = data.val_pos_edge_index.to(device)
    data.test_pos_edge_index = data.test_pos_edge_index.to(device)

    if data.val_neg_edge_index is not None:
        data.val_neg_edge_index = data.val_neg_edge_index.to(device)
    if data.test_neg_edge_index is not None:
        data.test_neg_edge_index = data.test_neg_edge_index.to(device)

    return data, dataset.num_node_features

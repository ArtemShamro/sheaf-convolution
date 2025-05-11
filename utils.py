import torch.nn.functional as F
import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_masks(adj_mat):
    mask_diag = torch.eye(adj_mat.shape[0]).bool()
    mask_ul = adj_mat.clone().detach().bool()
    mask_triu = torch.triu(adj_mat.clone().detach()).bool()
    mask_tril = torch.tril(adj_mat.clone().detach()).bool()
    return mask_diag, mask_ul, mask_triu, mask_tril


def get_adj_mat(G: nx.Graph, device='cpu'):
    adj_matrix = nx.to_numpy_array(G)
    adj_matrix = torch.tensor(adj_matrix).float().to(device)
    return adj_matrix


def get_degrees_and_edges(G: nx.Graph, device='cpu'):
    degrees = np.array(list(dict(G.degree()).values()))
    degrees = torch.tensor(degrees).float().to(device)
    edge_index = torch.tensor(
        list(G.edges) + [(v, u) for (u, v) in list(G.edges)]).T.to(device)
    return degrees, edge_index


def draw_results(train_accuracies, test_accuracies, train_losses, test_losses, grad_norm):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    axes[0].plot(train_accuracies, label="train_accuracies")
    axes[0].plot(test_accuracies, label="test_accuracies")
    axes[0].legend()

    axes[1].plot(train_losses, label="train_losses")
    axes[1].plot(test_losses, label="test_losses")
    axes[1].legend()

    for name, norms in grad_norm.items():
        axes[2].plot(norms, label=name)
    axes[2].legend(fontsize='small', loc='upper right', ncol=2)
    axes[2].set_title("Gradient Norms by Layer")
    plt.tight_layout()
    plt.show(block=False)


def draw_roc_auc(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(np.array(y_true), np.array(y_scores))
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    plt.savefig('roc_auc_curve.png')
    plt.close()


def get_mask_node_classification(G: nx.Graph, device: torch.device):
    num_nodes = np.array(G.nodes).size
    all_nodes = np.array(G.nodes)
    train_idxs = np.random.choice(all_nodes, int(num_nodes*0.7), replace=False)
    train_vert_mask = np.zeros(num_nodes)
    train_vert_mask[train_idxs] = 1
    train_vert_mask = train_vert_mask.astype(bool)
    test_vert_mask = ~train_vert_mask
    return torch.tensor(train_vert_mask).to(device), torch.tensor(test_vert_mask).to(device)


def draw_graph(G: nx.Graph, data_y=None, data_x: np.array = None):
    plt.figure(figsize=(16, 8))
    pos = nx.spring_layout(G, seed=42)
    if data_y is not None:
        class_colors = ['c', 'y', 'blue', 'red', 'brown', 'black', 'green']
        node_colors = [class_colors[data_y[node].item()] for node in G.nodes]
        nx.draw(G, pos, with_labels=True, node_color=node_colors,
                edge_color='black', node_size=800, font_size=12)
    else:
        nx.draw(G, pos, with_labels=True, edge_color='black',
                node_size=800, font_size=12)
    if data_x is not None:
        for node, (x, y) in pos.items():
            plt.text(x, y+0.1, str(data_x[node]), fontsize=9,
                     ha='center', bbox=dict(facecolor='white', alpha=0.6))

    plt.show(block=False)


def get_mask_edge_prediction(G: nx.Graph, train_size: float = None, test_size: float = None, device: str = 'cpu'):
    assert train_size is not None or test_size is not None, "Either train_size or test_size must be specified"
    test_size = test_size if test_size is not None else 1 - train_size
    train_size = train_size if train_size is not None else 1 - test_size

    assert train_size + test_size == 1, "Train and test sizes must sum to 1"

    # Все ребра
    edges = list(G.edges())

    # Перемешал
    random.shuffle(edges)

    # Выбрал случайных 0.8 доля
    num_test_edges = int(len(edges) * test_size)

    test_edges = edges[:num_test_edges]

    # Negative Sampling
    all_possible_edges = set((i, j) for i in G.nodes()
                             for j in G.nodes() if i < j)
    existing_edges = set(G.edges())
    # Все отстутсвующе ребра (i, j)
    negative_edges = list(all_possible_edges - existing_edges)

    random.shuffle(negative_edges)  # Shuffle
    # Выбрал столько же сколько и positive ребер в test set
    test_negative_edges = negative_edges[:len(test_edges)]

    test_mask_edges = test_edges + test_negative_edges

    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes)
    G_train.add_edges_from(test_mask_edges)
    test_mask = torch.tensor(nx.to_numpy_array(G_train)).bool().to(device)
    train_mask = ~test_mask

    return train_mask, test_mask


class CustomBCELoss(nn.Module):
    def __init__(self, print_loss=False):
        super().__init__()
        self.print_loss = print_loss

    def forward(self, preds, targets, mask=None):
        # default_mask = torch.ones_like(targets) - torch.eye(targets.shape[0], device=targets.device)
        # mask = default_mask.bool() if mask is None else (mask * default_mask).bool()

        pos_weight = (targets.numel() - targets.sum()) / targets.sum()
        # predictions, targets = predictions[mask], targets[mask]

        preds = preds.flatten()
        targets = targets.flatten()

        pos_mask = (targets == 1)
        zeros = torch.zeros_like(preds)
        # Вычисляем бинарную кросс-энтропию по формуле
        loss_pos = pos_weight * targets * (F.softplus(-preds))
        loss_neg = (1 - targets) * (F.softplus(preds))
        loss = loss_pos + loss_neg

        loss = loss_pos + loss_neg

        if self.print_loss:
            print("NEG  WEIGHTS: ", pos_weight)
            print("PREDICTIONS: ", preds)
            print("TARGETS: ", targets)
            print("LOSS_pos: ", loss[pos_mask])
            print("LOSS_neg: ", loss[~pos_mask])
            print("")

        # Возвращаем среднее значение потерь
        return loss.mean()

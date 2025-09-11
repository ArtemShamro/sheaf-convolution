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


def get_adj_mat(G: nx.Graph, device : torch.device):
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


def get_mask_edge_prediction(G: nx.Graph, test_size: float = None, val_size: float = None, neg_ratio=1, device: str = 'cpu'):
    num_nodes = G.number_of_nodes()
    edges = list(G.edges())
    num_edges = len(edges)
    random.shuffle(edges)
    num_test_edges = int(num_edges * test_size)
    num_val_edges = int(num_edges * val_size)

    test_edges = edges[:num_test_edges]
    val_edges = edges[num_test_edges:num_test_edges + num_val_edges]

    all_possible_edges = set((i, j) for i in range(num_nodes)
                             for j in range(i + 1, num_nodes))
    existing_edges = set((min(u, v), max(u, v)) for u, v in G.edges())
    negative_edges = list(all_possible_edges - existing_edges)
    random.shuffle(negative_edges)

    num_test_neg = int(num_test_edges * neg_ratio)
    num_val_neg = int(num_val_edges * neg_ratio)

    test_negative_edges = negative_edges[:num_test_neg]
    val_negative_edges = negative_edges[num_test_neg:num_test_neg + num_val_neg]

    train_mask = torch.ones((num_nodes, num_nodes),
                            dtype=torch.bool, device=device)
    val_mask = torch.zeros((num_nodes, num_nodes),
                           dtype=torch.bool, device=device)
    test_mask = torch.zeros((num_nodes, num_nodes),
                            dtype=torch.bool, device=device)

    val_edges_array = np.array(val_edges).T
    val_neg_edges_array = np.array(val_negative_edges).T
    val_mask[val_edges_array[0], val_edges_array[1]] = True
    val_mask[val_edges_array[1], val_edges_array[0]] = True
    val_mask[val_neg_edges_array[0], val_neg_edges_array[1]] = True
    val_mask[val_neg_edges_array[1], val_neg_edges_array[0]] = True

    test_edges_array = np.array(test_edges).T
    test_neg_edges_array = np.array(test_negative_edges).T
    test_mask[test_edges_array[0], test_edges_array[1]] = True
    test_mask[test_edges_array[1], test_edges_array[0]] = True
    test_mask[test_neg_edges_array[0], test_neg_edges_array[1]] = True
    test_mask[test_neg_edges_array[1], test_neg_edges_array[0]] = True

    train_mask[val_edges_array[0], val_edges_array[1]] = False
    train_mask[val_edges_array[1], val_edges_array[0]] = False
    train_mask[val_neg_edges_array[0], val_neg_edges_array[1]] = False
    train_mask[val_neg_edges_array[1], val_neg_edges_array[0]] = False
    train_mask[test_edges_array[0], test_edges_array[1]] = False
    train_mask[test_edges_array[1], test_edges_array[0]] = False
    train_mask[test_neg_edges_array[0], test_neg_edges_array[1]] = False
    train_mask[test_neg_edges_array[1], test_neg_edges_array[0]] = False

    train_mask.fill_diagonal_(False)

    G_train = G.copy()
    G_train.remove_edges_from(test_edges + val_edges)

    return G_train, train_mask, val_mask, test_mask


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

        # Вычисляем бинарную кросс-энтропию
        loss_pos = pos_weight * targets * (F.softplus(-preds))
        loss_neg = (1 - targets) * (F.softplus(preds))
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

# class CustomBCELoss(nn.Module):
#     def __init__(self, print_loss=False):
#         super().__init__()
#         self.print_loss = print_loss

#     def forward(self, preds, targets, mask=None):
#         if mask is not None:
#             preds = preds[mask]
#             targets = targets[mask]

#         # Ограничиваем pos_weight для стабильности
#         pos_weight = min((targets.numel() - targets.sum()) /
#                          targets.sum(), 50.0)
#         pos_weight = torch.tensor(pos_weight, device=preds.device)

#         # Используем BCEWithLogitsLoss
#         criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#         loss = criterion(preds.flatten(), targets.flatten())

#         if self.print_loss:
#             print("NEG WEIGHTS:", pos_weight.item())
#             print("PREDICTIONS:", preds.flatten()[:10])
#             print("TARGETS:", targets.flatten()[:10])
#             print("LOSS:", loss.item())
#             print("")

#         return loss


class VGAELoss(nn.Module):
    def __init__(self, print_loss: bool = False):
        """
        Функция потерь для VGAE: ELBO = BCE (реконструкция рёбер) - KL-дивергенция.

        Args:
            print_loss (bool): Если True, выводит значения BCE, KL и общего лосса для отладки.
        """
        super().__init__()
        self.print_loss = print_loss

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Вычисляет ELBO: -BCE - KL-дивергенция.

        Args:
            logits (torch.Tensor): Логиты рёбер, форма (num_nodes, num_nodes).
            labels (torch.Tensor): Целевая матрица смежности, форма (num_nodes, num_nodes).
            mu (torch.Tensor): Средние латентных распределений, форма (num_nodes, hidden_dim2).
            logvar (torch.Tensor): Логарифмы дисперсий, форма (num_nodes, hidden_dim2).
            mask (torch.Tensor): Маска для рёбер/non-edges, форма (num_nodes, num_nodes), опционально.

        Returns:
            loss (torch.Tensor): Скалярное значение ELBO.
        """
        # BCE для реконструкции рёбер
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction='sum')

        # KL-дивергенция: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # ELBO = -BCE - KL (минимизируем -ELBO)
        loss = bce_loss + kl_div

        if self.print_loss:
            print("BCE Loss:", bce_loss.item())
            print("KL Divergence:", kl_div.item())
            print("Total Loss:", loss.item())
            print("Logits mean:", logits.mean().item())
            print("Logits std:", logits.std().item())
            print("Mu std:", mu.std().item())
            print("Logvar std:", logvar.std().item())
            print("")

        return loss

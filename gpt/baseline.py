"""
Baseline: Graph Auto-Encoder (GAE) for Link Prediction
======================================================

Использует GCN-энкодер + dot-product декодер.
Метрики: AUC/AP и confusion matrix (Train/Val/Test) с подбором порога по F1.
"""

import os
import random
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix, f1_score


# -------------------
# Encoder / Decoder
# -------------------

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act(h)
            h = self.dropout(h)
        return h


class DotDecoder(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize

    def forward(self, z: Tensor, edge_index: Tensor) -> Tensor:
        u, v = edge_index
        zu, zv = z[u], z[v]
        if self.normalize:
            zu = F.normalize(zu, p=2, dim=-1)
            zv = F.normalize(zv, p=2, dim=-1)
        return (zu * zv).sum(dim=-1)  # logits


# -------------------
# Utils
# -------------------

def negative_edges(num_nodes, num_neg, pos_edge_index, device):
    neg = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method='sparse'
    )
    return neg.to(device)


def evaluate(encoder, decoder, data, edge_index_ref, device, return_raw=False, add_neg=False, neg_ratio=1.0):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(data.x, edge_index_ref)

        if add_neg:
            # сэмплируем негативы
            pos_idx = data.edge_label_index[:, data.edge_label == 1]
            num_pos = pos_idx.size(1)
            num_neg = int(neg_ratio * num_pos)
            neg_idx = negative_edges(data.num_nodes, num_neg, pos_idx, device)

            edge_idx = torch.cat([pos_idx, neg_idx], dim=1)
            y = np.concatenate([np.ones(num_pos), np.zeros(num_neg)])
        else:
            edge_idx = data.edge_label_index
            y = data.edge_label.cpu().numpy()

        logits = decoder(z, edge_idx)
        s = torch.sigmoid(logits).cpu().numpy()

        # если только один класс — безопасно вернуть NaN
        if len(np.unique(y)) < 2:
            return float("nan"), float("nan"), (y, s) if return_raw else None

        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
    raw = (y, s) if return_raw else None
    return float(auc), float(ap), raw


def choose_threshold_f1(y, s):
    prec, rec, thr = precision_recall_curve(y, s)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1s))])


def report_confusion(y, s, thr, name):
    y_pred = (s >= thr).astype(int)
    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"Confusion matrix ({name}, thr={thr:.3f}):\n{cm}\nF1={f1:.4f}")


# -------------------
# Training loop
# -------------------

def train_epoch(encoder, decoder, data, opt, device, neg_ratio=1.0):
    encoder.train()
    decoder.train()
    opt.zero_grad()

    z = encoder(data.x, data.edge_index)

    pos_idx = data.edge_label_index[:, data.edge_label == 1]
    num_pos = pos_idx.size(1)
    num_neg = int(neg_ratio * num_pos)
    neg_idx = negative_edges(data.num_nodes, num_neg, pos_idx, device)

    pos_logits = decoder(z, pos_idx)
    neg_logits = decoder(z, neg_idx)

    y = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
    logits = torch.cat([pos_logits, neg_logits])
    loss = F.binary_cross_entropy_with_logits(logits, y)

    loss.backward()
    opt.step()
    return float(loss.item())


# -------------------
# Main
# -------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--wd', type=float, default=5e-5)
    parser.add_argument('--neg-ratio', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    dataset = Planetoid(root=os.path.join(
        'data', 'Planetoid'), name=args.dataset)
    data = dataset[0]
    data = ToUndirected()(data)
    splitter = RandomLinkSplit(
        is_undirected=True, num_val=0.05, num_test=0.1, add_negative_train_samples=False)
    train_data, val_data, test_data = splitter(data)
    for d in (train_data, val_data, test_data):
        d.x = d.x.to(device)
        d.edge_index = d.edge_index.to(
            device) if d.edge_index is not None else None
        d.edge_label_index = d.edge_label_index.to(device)
        d.edge_label = d.edge_label.to(device)
        d.num_nodes = data.num_nodes

    encoder = GCNEncoder(in_dim=train_data.x.size(-1),
                         hidden_dim=args.hidden_dim, layers=args.layers).to(device)
    decoder = DotDecoder().to(device)
    opt = torch.optim.Adam(list(encoder.parameters(
    )) + list(decoder.parameters()), lr=args.lr, weight_decay=args.wd)

    best_val = -1
    best_state = None

    for epoch in range(1, args.epochs+1):
        loss = train_epoch(encoder, decoder, train_data,
                           opt, device, args.neg_ratio)
        val_auc, val_ap, _ = evaluate(
            encoder, decoder, val_data, train_data.edge_index, device)
        if val_auc > best_val:
            best_val = val_auc
            best_state = {
                'encoder': {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                'decoder': {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
            }
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[Epoch {epoch:03d}] loss={loss:.4f}  val_auc={val_auc:.4f}  val_ap={val_ap:.4f}")

    if best_state is not None:
        encoder.load_state_dict(best_state['encoder'])
        decoder.load_state_dict(best_state['decoder'])

    # Threshold by F1 on validation
    _, _, raw_val = evaluate(encoder, decoder, val_data,
                             train_data.edge_index, device, return_raw=True)
    yv, sv = raw_val
    thr = choose_threshold_f1(yv, sv)

    train_auc, train_ap, raw_train = evaluate(
        encoder, decoder, train_data, train_data.edge_index, device,
        return_raw=True, add_neg=True, neg_ratio=args.neg_ratio
    )

    yt, st = raw_train
    val_auc, val_ap, _ = evaluate(
        encoder, decoder, val_data, train_data.edge_index, device)
    test_auc, test_ap, raw_test = evaluate(
        encoder, decoder, test_data, train_data.edge_index, device, return_raw=True)
    yte, ste = raw_test

    report_confusion(yt, st, thr, "Train")
    report_confusion(yv, sv, thr, "Val")
    report_confusion(yte, ste, thr, "Test")

    print(f"\\nFinal Train: AUC={train_auc: .4f}  AP={train_ap: .4f}")
    print(f"Final Val:   AUC={val_auc: .4f}  AP={val_ap: .4f}")
    print(f"Final Test:  AUC={test_auc: .4f}  AP={test_ap: .4f}")


if __name__ == "__main__":
    main()

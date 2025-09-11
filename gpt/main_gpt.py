from __future__ import annotations
"""
Sheaf Diffusion (Connection Laplacian) — Link Prediction & Node Classification (Cora)
====================================================================================

This is a compact, practical reference that implements a *connection-style* sheaf
diffusion layer and supports two tasks:
- `--task linkpred`  (edge recovery)
- `--task nodeclf`   (node classification like in Cora benchmarks)

Key design choices for node classification (closer to paper setups):
- Connection-style normalized update with learnable transports on directed edges.
- Strong dropout (0.5), weight decay (5e-4), Adam lr=0.01, StepLR.
- MLP classifier head on top of embeddings.

Run examples
------------
# Link prediction (as before)
python3 main_gpt.py --task linkpred --dataset Cora --epochs 60 --hidden-dim 32 --layers 3

# Node classification (recommended settings)
python3 main_gpt.py --task nodeclf --dataset Cora --epochs 200 --hidden-dim 64 --layers 3

Dependencies: PyTorch + PyTorch Geometric, scikit-learn.
"""

import argparse
import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torch_geometric.data import Data
    from torch_geometric.utils import to_undirected
    from torch_geometric.transforms import ToUndirected
    from torch_geometric.datasets import Planetoid
    from torch_geometric.utils import negative_sampling
    from torch_geometric.transforms import RandomLinkSplit
except Exception as e:
    raise SystemExit(
        "PyTorch Geometric is required. Install with pip.\n"
        f"Import error: {e}"
    )

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix, f1_score,
)

# -----------------------------------------------------------------------------
# Sheaf Connection Diffusion Layer (normalized)
# -----------------------------------------------------------------------------


class SheafConnectionLayer(nn.Module):
    """Connection-style sheaf diffusion with degree normalization.

    For each *directed* edge e=(j->i), learn a transport T_e ∈ R^{d×d} and
    conductance w_e ≥ 0 (softplus). Update:

        x_i' = x_i + α * ( 1/deg(i) ) * Σ_{j∈N(i)} w_{j→i} ( T_{j→i} x_j − x_i )

    This stabilizes training on small graphs (Cora) and mimics a connection
    Laplacian step.
    """

    def __init__(self, d: int, num_directed_edges: int, alpha: float = 0.1, dropout: float = 0.5):
        super().__init__()
        self.d = d
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=True)
        self.T = nn.Parameter(torch.empty(num_directed_edges, d, d))
        self.raw_w = nn.Parameter(torch.empty(num_directed_edges))
        self.reset_parameters()
        self.drop = nn.Dropout(dropout)

    def reset_parameters(self):
        with torch.no_grad():
            self.T.zero_()
            for e in range(self.T.size(0)):
                self.T[e].copy_(torch.eye(self.d))
            self.T.add_(0.01 * torch.randn_like(self.T))
            self.raw_w.fill_(math.log(math.exp(0.1) - 1.0)
                             )  # softplus^{-1}(0.1)

    def forward(self, x: Tensor, edge_index: Tensor, deg: Tensor) -> Tensor:
        # edge_index is undirected; we use both directions.
        src, dst = edge_index
        E = edge_index.size(1)
        assert self.T.size(0) == 2 * E

        T_fwd = self.T[:E]
        T_rev = self.T[E:]
        w_fwd = F.softplus(self.raw_w[:E])
        w_rev = F.softplus(self.raw_w[E:])

        x_src = x[src]
        x_dst = x[dst]

        # Optional dropout on features (message dropout)
        x_src = self.drop(x_src)
        x_dst = self.drop(x_dst)

        m_fwd = torch.einsum('edk,ek->ed', T_fwd, x_src) - x_dst
        m_rev = torch.einsum('edk,ek->ed', T_rev, x_dst) - x_src

        out = x.clone()
        # Degree-normalized aggregation
        out.index_add_(0, dst, (self.alpha * w_fwd /
                       (deg[dst].clamp_min(1))).unsqueeze(-1) * m_fwd)
        out.index_add_(0, src, (self.alpha * w_rev /
                       (deg[src].clamp_min(1))).unsqueeze(-1) * m_rev)
        return out


class SheafDiffusionEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, edge_index: Tensor, dropout: float = 0.5):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        E = edge_index.size(1)
        for _ in range(layers):
            self.layers.append(SheafConnectionLayer(
                hidden_dim, num_directed_edges=2*E, dropout=dropout))
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(layers)])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Precompute (undirected) degrees
        N = x.size(0)
        deg = torch.zeros(N, device=x.device, dtype=x.dtype)
        ones = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        deg.index_add_(0, edge_index[0], ones)
        deg.index_add_(0, edge_index[1], ones)

        h = self.act(self.in_proj(x))
        h = self.drop(h)
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edge_index, deg)
            h = norm(h)
            h = self.act(h)
            h = self.drop(h)
        return h


# -----------------------------------------------------------------------------
# Decoders / Heads
# -----------------------------------------------------------------------------

class DotDecoder(nn.Module):
    def forward(self, z: Tensor, edge_index: Tensor) -> Tensor:
        u, v = edge_index
        return (z[u] * z[v]).sum(dim=-1)  # logits


class NodeMLPHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(in_dim, num_classes)
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.mlp(h)


class SheafNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, edge_index: Tensor, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = SheafDiffusionEncoder(
            in_dim, hidden_dim, layers, edge_index, dropout=dropout)
        self.head = NodeMLPHead(hidden_dim, num_classes, dropout=dropout)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        h = self.encoder(x, edge_index)
        return self.head(h)


# -----------------------------------------------------------------------------
# Training / Evaluation utilities
# -----------------------------------------------------------------------------

def negative_edges(num_nodes: int, num_neg: int, pos_edge_index: Tensor, device: torch.device) -> Tensor:
    neg = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg,
        method='sparse'
    )
    return neg.to(device)


def evaluate_link(encoder, decoder, data: Data, edge_index_ref: Tensor, device: torch.device, return_raw: bool = False):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        z = encoder(data.x, edge_index_ref)
        logits = decoder(z, data.edge_label_index)
        y = data.edge_label.float().cpu().numpy()
        s = torch.sigmoid(logits).cpu().numpy()
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
    raw = (y, s) if return_raw else None
    return float(auc), float(ap), raw


def choose_threshold_f1(y: np.ndarray, s: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, s)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[int(np.nanargmax(f1s))])


def report_confusion(raw, split="Val", thr: float | None = None):
    y, s = raw
    if thr is None:
        thr = choose_threshold_f1(y, s)
    y_pred = (s >= thr).astype(int)
    cm = confusion_matrix(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"Confusion matrix ({split}, thr={thr:.3f}):\n{cm}\nF1={f1:.4f}")


# -----------------------------------------------------------------------------
# Config / Runner
# -----------------------------------------------------------------------------

@dataclass
class TrainConfig:
    dataset: str = "Cora"
    task: str = "linkpred"                 # or "nodeclf"
    hidden_dim: int = 64
    layers: int = 3
    epochs: int = 200                       # for nodeclf per paper-style training
    lr: float = 1e-2
    weight_decay: float = 5e-4
    batch_neg_ratio: float = 1.0            # for linkpred
    dropout: float = 0.5
    seed: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def run_on_dataset(cfg: TrainConfig):
    torch.manual_seed(cfg.seed)
    torch.cuda.empty_cache()
    random.seed(cfg.seed)
    device = torch.device(cfg.device)

    dataset = Planetoid(root=os.path.join(
        'data', 'Planetoid'), name=cfg.dataset)
    base = dataset[0]

    if cfg.task == 'linkpred':
        data = ToUndirected()(base)
        splitter = RandomLinkSplit(
            is_undirected=True, num_val=0.05, num_test=0.1, add_negative_train_samples=False)
        train_data, val_data, test_data = splitter(data)
        # to device
        for d in (train_data, val_data, test_data):
            d.x = d.x.to(device)
            d.edge_index = d.edge_index.to(device)
            d.edge_label_index = d.edge_label_index.to(device)
            d.edge_label = d.edge_label.to(device)
            d.num_nodes = base.num_nodes

        encoder = SheafDiffusionEncoder(in_dim=train_data.x.size(-1), hidden_dim=cg.hidden_dim if (cg := cfg) else cfg.hidden_dim,
                                        layers=cfg.layers, edge_index=train_data.edge_index, dropout=cfg.dropout).to(device)
        decoder = DotDecoder().to(device)
        opt = torch.optim.Adam(list(encoder.parameters(
        )) + list(decoder.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)

        best = -1.0
        best_state = None
        for epoch in range(1, cfg.epochs+1):
            encoder.train()
            decoder.train()
            opt.zero_grad()
            z = encoder(train_data.x, train_data.edge_index)
            pos_idx = train_data.edge_label_index[:,
                                                  train_data.edge_label == 1]
            num_pos = pos_idx.size(1)
            num_neg = max(1, int(cfg.batch_neg_ratio * num_pos))
            neg_idx = negative_edges(
                train_data.num_nodes, num_neg, pos_idx, device)
            pos_logits = decoder(z, pos_idx)
            neg_logits = decoder(z, neg_idx)
            y = torch.cat([torch.ones_like(pos_logits),
                          torch.zeros_like(neg_logits)])
            logits = torch.cat([pos_logits, neg_logits])
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()

            if epoch % 10 == 0 or epoch == 1:
                val_auc, val_ap, _ = evaluate_link(
                    encoder, decoder, val_data, train_data.edge_index, device)
                print(
                    f"[Epoch {epoch:03d}] loss={loss.item():.4f}  val_auc={val_auc:.4f}  val_ap={val_ap:.4f}")
                if val_auc > best:
                    best = val_auc
                    best_state = {
                        'encoder': {k: v.detach().cpu() for k, v in encoder.state_dict().items()},
                        'decoder': {k: v.detach().cpu() for k, v in decoder.state_dict().items()},
                    }
        if best_state is not None:
            encoder.load_state_dict(best_state['encoder'])
            decoder.load_state_dict(best_state['decoder'])

        val_auc, val_ap, raw_val = evaluate_link(
            encoder, decoder, val_data, train_data.edge_index, device, return_raw=True)
        test_auc, test_ap, raw_test = evaluate_link(
            encoder, decoder, test_data, train_data.edge_index, device, return_raw=True)
        thr = choose_threshold_f1(*raw_val)
        report_confusion(raw_val, split='Val', thr=thr)
        report_confusion(raw_test, split='Test', thr=thr)
        print(f"\nFinal Val:  AUC={val_auc:.4f}  AP={val_ap:.4f}")
        print(f"Final Test: AUC={test_auc:.4f}  AP={test_ap:.4f}")

    elif cfg.task == 'nodeclf':
        data = base.to(device)
        model = SheafNodeClassifier(in_dim=data.x.size(-1), hidden_dim=cfg.hidden_dim, layers=cfg.layers,
                                    edge_index=data.edge_index, num_classes=dataset.num_classes, dropout=cfg.dropout).to(device)
        opt = torch.optim.Adam(model.parameters(
        ), lr=0.01 if cfg.lr is None else cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=50, gamma=0.5)

        best_val = -1.0
        best_state = None
        for epoch in range(1, cfg.epochs+1):
            model.train()
            opt.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(
                out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            opt.step()
            sched.step()

            if epoch % 10 == 0 or epoch == 1:
                model.eval()
                with torch.no_grad():
                    logits = model(data.x, data.edge_index)
                    pred = logits.argmax(dim=-1)
                    acc_val = (pred[data.val_mask] ==
                               data.y[data.val_mask]).float().mean().item()
                print(
                    f"[Epoch {epoch:03d}] loss={loss.item():.4f}  val_acc={acc_val:.4f}")
                if acc_val > best_val:
                    best_val = acc_val
                    best_state = {k: v.detach().cpu()
                                  for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            pred = logits.argmax(dim=-1)
            acc_train = (pred[data.train_mask] ==
                         data.y[data.train_mask]).float().mean().item()
            acc_val = (pred[data.val_mask] ==
                       data.y[data.val_mask]).float().mean().item()
            acc_test = (pred[data.test_mask] ==
                        data.y[data.test_mask]).float().mean().item()
        print(f"Final Train Acc: {acc_train:.4f}")
        print(f"Final Val Acc:   {acc_val:.4f}")
        print(f"Final Test Acc:  {acc_test:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=str, default='Cora')
    p.add_argument('--task', type=str, default='nodeclf',
                   choices=['linkpred', 'nodeclf'])
    p.add_argument('--hidden-dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--weight-decay', type=float, default=5e-4)
    p.add_argument('--batch-neg-ratio', type=float, default=1.0)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str,
                   default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    cfg = TrainConfig(dataset=args.dataset, task=args.task, hidden_dim=args.hidden_dim, layers=args.layers,
                      epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, batch_neg_ratio=args.batch_neg_ratio,
                      dropout=args.dropout, seed=args.seed, device=args.device)
    run_on_dataset(cfg)

import torch
import wandb
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, Metric, AveragePrecision
from sklearn.metrics import confusion_matrix
import numpy as np


class MetricLogger:
    def __init__(self, device, task="binary"):
        self.metrics = {
            "train": MultiMetric(task=task, prefix="train/").to(device),
            "test": MultiMetric(task=task, prefix="test/").to(device)
        }

    def update(self, split: str, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        self.metrics[split].update(logits, targets, loss)

    def compute(self, split: str):
        return self.metrics[split].compute()

    def reset(self, split: str):
        self.metrics[split].reset()

    def compute_all(self):
        return {split: m.compute() for split, m in self.metrics.items()}

    def reset_all(self):
        for m in self.metrics.values():
            m.reset()

    def log_to_wandb(self, epoch: int):
        metrics_dict = self.compute_all()
        log_dict = {"epoch": epoch}
        for split, metrics in metrics_dict.items():
            for k, v in metrics.items():
                log_dict[f"{split}/{k}"] = v if isinstance(
                    v, float) else v.item()
        wandb.log(log_dict, step=epoch)


class MultiMetric(Metric):
    def __init__(self, task, prefix):
        super().__init__()
        self.prefix = prefix
        self.accuracy = Accuracy(task=task)
        self.precision = Precision(task=task)
        self.recall = Recall(task=task)
        self.auroc = AUROC(task=task)
        self.ap = AveragePrecision(task=task)
        self.loss = 0.0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: float):
        pred_classes = (logits > 0).int()
        self.accuracy.update(pred_classes, targets)
        self.precision.update(pred_classes, targets)
        self.recall.update(pred_classes, targets)

        probs = torch.sigmoid(logits)
        self.ap.update(probs, targets)
        self.auroc.update(probs, targets)
        self.loss += loss

    def compute(self):
        return {
            f"accuracy": self.accuracy.compute().item(),
            f"precision": self.precision.compute().item(),
            f"recall": self.recall.compute().item(),
            f"loss": self.loss,
            f"aucroc": self.auroc.compute().item(),
            f"ap": self.ap.compute().item()
        }

    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.auroc.reset()
        self.ap.reset()
        self.loss = 0.0


class LayerwiseGradNormMetric(Metric):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.layer_names = [name for name, _ in model.named_parameters()]

        self.grad_norms = {name: [
        ] for name in self.layer_names if 'bias' not in name and 'running' not in name}

    def update(self, epoch):
        for name, param in self.model.named_parameters():
            if "bias" in name or "running" in name:
                continue
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.grad_norms[name].append(grad_norm)
            wandb.log({f"grad_norm/{name}": grad_norm},
                      step=epoch)

    def compute(self):
        return self.grad_norms

    def reset(self):
        self.grad_norms = {name: [] for name in self.layer_names}


def compute_log_confusion_matrix(labels, preds, train_mask, test_mask):
    pred_classes = (preds > 0.5).int()
    neg_preds = 1 - preds
    roc_preds_train = torch.vstack(
        [neg_preds[train_mask].flatten(), preds[train_mask].flatten()]).T
    roc_preds_test = torch.vstack(
        [neg_preds[test_mask].flatten(), preds[test_mask].flatten()]).T
    wandb.log(
        {
            "train_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.array(labels[train_mask].cpu().flatten()).tolist(),
                preds=np.array(
                    pred_classes[train_mask].cpu().flatten()).tolist(),
                class_names=["Class 0", "Class 1"],
                title="Confusion Matrix (Train)"
            ),
            "test_confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.array(
                    labels[test_mask].cpu().flatten()),
                preds=np.array(
                    pred_classes[test_mask].cpu().flatten()),
                class_names=["Class 0", "Class 1"],
                title="Confusion Matrix (Test)"
            ),
            "train_roc_auc": wandb.plot.roc_curve(
                y_true=np.array(labels[train_mask].cpu().flatten()),
                y_probas=np.array(
                    roc_preds_train.cpu()),
                labels=["Class 0", "Class 1"],
                title="ROC Curve (Train)"
            ),
            "test_roc_auc": wandb.plot.roc_curve(
                y_true=np.array(labels[test_mask].cpu().flatten()),
                y_probas=np.array(roc_preds_test.cpu()),
                labels=["Class 0", "Class 1"],
                title="ROC Curve (Test)"
            )
        }
    )

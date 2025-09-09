import torch
import wandb
from metrics.comet_logger import get_experiment
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, Metric, AveragePrecision
from sklearn.metrics import confusion_matrix
import numpy as np


experiment = get_experiment()
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
                value = v if isinstance(v, float) else v.item()
                experiment.log_metric(f"{split}/{k}", value, step=epoch)


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
                experiment.log_metric(f"grad_norm/{name}", grad_norm, step=epoch)

    def compute(self):
        return self.grad_norms

    def reset(self):
        self.grad_norms = {name: [] for name in self.layer_names}


def compute_log_confusion_matrix(labels, preds, train_mask, test_mask):
    pred_classes = (preds > 0.5).int()
    neg_preds = 1 - preds
    
    experiment.log_confusion_matrix(
        y_true=labels[train_mask].cpu().numpy(),
        y_predicted=pred_classes[train_mask].cpu().numpy(),
        title="Confusion Matrix (Train)",
        file_name="train_confusion_matrix.json"
    )

    experiment.log_confusion_matrix(
        y_true=labels[test_mask].cpu().numpy(),
        y_predicted=pred_classes[test_mask].cpu().numpy(),
        title="Confusion Matrix (Test)",
        file_name="test_confusion_matrix.json"
    )

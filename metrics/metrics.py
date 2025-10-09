import os
import tempfile
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, Metric, AveragePrecision
from torchmetrics.classification import BinaryROC, BinaryPrecisionRecallCurve
from torchmetrics.functional.classification import binary_auroc, binary_average_precision


class MetricLogger:
    def __init__(self, device, comet_logger, task="binary", ):
        self.experiment = comet_logger.get_experiment()
        self.metrics = {
            "train": MultiMetric(task=task, prefix="train/", loss_only=True).to(device),
            "val": MultiMetric(task=task, prefix="val/").to(device)

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

    def log_other(self, name, val):
        self.experiment.log_other(name, val)

    def log_to_wandb(self, epoch: int):
        metrics_dict = self.compute_all()
        for split, metrics in metrics_dict.items():
            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.detach().to("cpu", non_blocking=True)
                    try:
                        v = float(v)
                    except Exception:
                        v = v.mean().item()
                self.experiment.log_metric(f"{split}/{k}", v, step=epoch)

    def log_model_params(self, model: torch.nn.Module, maps_norms, step: int | None = None):
        """
        Логгирует в Comet:
            - нормы параметров (L2-норма весов по каждому слою)
            - нормы градиентов (L2-норма по каждому слою)
        Вызывается после backward() и перед/после optimizer.step().
        """
        experiment = self.experiment
        total_param_norm = 0.0
        total_grad_norm = 0.0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Норма весов
                param_norm = param.data.norm(2).item()
                total_param_norm += param_norm ** 2
                experiment.log_metric(
                    f"param_norm/{name}", param_norm, step=step)

                # Норма градиентов (если они посчитаны)
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
                    experiment.log_metric(
                        f"grad_norm/{name}", grad_norm, step=step)

        total_param_norm = total_param_norm ** 0.5
        total_grad_norm = total_grad_norm ** 0.5

        experiment.log_metric("param_norm/total", total_param_norm, step=step)
        experiment.log_metric("grad_norm/total", total_grad_norm, step=step)

        for i, norm_val in enumerate(maps_norms):
            experiment.log_metric(
                f"maps/layer_{i}_mean_norm", norm_val, step=step)


class MultiMetric(Metric):
    def __init__(self, task, prefix, loss_only=False):
        super().__init__()
        self.prefix = prefix
        # self.accuracy = Accuracy(task=task)
        # self.precision = Precision(task=task)
        # self.recall = Recall(task=task)
        self.loss_only = loss_only
        self.loss = torch.tensor(0.0)
        self.val_logits = []
        self.val_targets = []
        # if not loss_only:
        #     self.auroc = AUROC(task=task)
        #     self.ap = AveragePrecision(task=task)

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor):
        # self.accuracy.update(pred_classes, targets)
        # self.precision.update(pred_classes, targets)
        # self.recall.update(pred_classes, targets)
        self.loss = self.loss + loss.detach()
        if not self.loss_only:
            self.val_logits.append(logits.detach())
            self.val_targets.append(targets.detach())
            # probs = torch.sigmoid(logits)
            # self.ap.update(probs, targets)
            # self.auroc.update(probs, targets)
            # self.loss += loss

    def compute(self):
        result = {"loss": self.loss}
        if not self.loss_only:
            # result["aucroc"] = self.auroc.compute()
            # result["ap"] = self.ap.compute()
            logits = torch.cat(self.val_logits, dim=0)
            targets = torch.cat(self.val_targets, dim=0)
            probs = torch.sigmoid(logits)
            # считаем метрики напрямую — без sync
            result["aucroc"] = binary_auroc(probs, targets)
            result["ap"] = binary_average_precision(probs, targets)
        return result

    def reset(self):
        self.loss = torch.tensor(0.0)
        self.val_logits.clear()
        self.val_targets.clear()
        # if not self.loss_only:
        #     self.auroc.reset()
        #     self.ap.reset()


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
                self.experiment.log_metric(
                    f"grad_norm/{name}", grad_norm, step=epoch)

    def compute(self):
        return self.grad_norms

    def reset(self):
        self.grad_norms = {name: [] for name in self.layer_names}

    def compute_confusion_matrix_torch(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        # предполагаем бинарную классификацию (0/1)
        tp = ((y_true == 1) & (y_pred == 1)).sum().item()
        tn = ((y_true == 0) & (y_pred == 0)).sum().item()
        fp = ((y_true == 0) & (y_pred == 1)).sum().item()
        fn = ((y_true == 1) & (y_pred == 0)).sum().item()
        return [[tn, fp], [fn, tp]]

    def compute_log_confusion_matrix(self, labels, preds, train_mask, test_mask):
        pred_classes = (preds > 0.5).int()

        # train
        cm_train = self.compute_confusion_matrix_torch(
            labels[train_mask], pred_classes[train_mask])
        self.experiment.log_confusion_matrix(
            matrix=cm_train,
            labels=["0", "1"],
            title="Confusion Matrix (Train)",
            file_name="train_confusion_matrix.json"
        )

        # test
        cm_test = self.compute_confusion_matrix_torch(
            labels[test_mask], pred_classes[test_mask])
        self.experiment.log_confusion_matrix(
            matrix=cm_test,
            labels=["0", "1"],
            title="Confusion Matrix (Test)",
            file_name="test_confusion_matrix.json"
        )

    def log_curves_to_comet(self, probs, targets, split: str):
        # считаем метрики на GPU
        roc_metric = BinaryROC()
        pr_metric = BinaryPrecisionRecallCurve()
        roc_metric.update(probs, targets)
        pr_metric.update(probs, targets)

        fpr, tpr, _ = roc_metric.compute()
        precision_torch, recall_torch, _ = pr_metric.compute()

        # считаем PR ещё раз через sklearn для совместимости (если нужно)
        precision, recall, _ = precision_recall_curve(
            targets.cpu().numpy(), probs.cpu().numpy()
        )

        # === ROC plot ===
        plt.figure()
        plt.plot(fpr.cpu().numpy(), tpr.cpu().numpy(), label="ROC curve")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{split} ROC curve")
        plt.legend(loc="lower right")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            self.experiment.log_image(
                tmpfile.name, name=f"{split}_roc_curve.png")
        plt.close()

        # === Precision-Recall plot ===
        plt.figure()
        plt.plot(recall, precision, label="PR curve (sklearn)")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{split} Precision-Recall curve")
        plt.legend(loc="lower left")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            plt.savefig(tmpfile.name)
            self.experiment.log_image(
                tmpfile.name, name=f"{split}_pr_curve.png")
        plt.close()

        # подчистим временные файлы
        try:
            os.remove(tmpfile.name)
        except:
            pass

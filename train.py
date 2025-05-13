from tqdm import tqdm
import torch
from metrics.metrics import LayerwiseGradNormMetric, MetricLogger, compute_log_confusion_matrix
import networkx as nx
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def train(epochs, model, criterion, data, labels, graph: nx.Graph, optimizer, mask, metric_logger: MetricLogger, scheduler=None, early_stop_iters=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    metric = LayerwiseGradNormMetric(model)
    val_aucs = []
    val_aps = []
    val_accs = []
    train_mask, test_mask = mask
    train_mask, test_mask = train_mask.to(
        device), test_mask.to(device)  # ПОФИКСИТЬ
    print("TRAIN: train mask edges", train_mask.int().sum().item())
    print("TRAIN: mask shape, train = ",
          train_mask.shape, " test = ", test_mask.shape)
    print("TRAIN: edges in train mask = ",
          labels[train_mask].int().sum().item())
    print("TRAIN: edges in test mask = ",
          labels[test_mask].int().sum().item())
    best_test_loss, stop_counter = 1e10, 0
    best_test_accuracy = 0

    with tqdm(range(1, epochs + 1)) as pbar:
        for epoch in pbar:
            metric_logger.reset_all()

            model.train()
            optimizer.zero_grad()

            logits = model(data, graph)
            probs = torch.sigmoid(logits).detach()
            preds = (probs > 0.5).int()

            loss = criterion(logits[train_mask], labels[train_mask])
            print("loss", loss.item())
            acc = (preds[train_mask] == labels[train_mask]
                   ).sum() / preds[train_mask].numel()
            print("acc", acc.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # test_accuracy = ((preds > 0.5).flatten().int() ==
            #                  labels.flatten().int()).sum() / preds.numel()
            # print("test_accuracy", test_accuracy)

            metric_logger.update(
                # "train", preds.flatten(), labels.flatten(), loss.item())

                "train", logits[train_mask], labels[train_mask], loss.item())
            # metric.update(epoch)

            model.eval()
            with torch.no_grad():

                # logits, _, _ = model(data, graph)
                # preds = (logits > 0).float()

                # test_loss = criterion(logits[test_mask], labels[test_mask])

                metric_logger.update(
                    "test", logits[test_mask], labels[test_mask], 0)

                labels_1 = labels.detach().cpu().numpy()
                probs = probs.detach().cpu().numpy()
                preds = preds.detach().cpu().numpy()
                test_mask_1 = test_mask.cpu().numpy()
                print("labels", labels_1[test_mask_1].sum(),
                      " / ", labels_1[test_mask_1].shape)
                auc = roc_auc_score(labels_1[test_mask_1], probs[test_mask_1])
                ap = average_precision_score(
                    labels_1[test_mask_1], probs[test_mask_1])
                acc = accuracy_score(labels_1[test_mask_1], preds[test_mask_1])

                print(
                    f"TRAIN: epoch {epoch} loss {loss.item()} test_acc {acc} test_auc {auc} test_ap {ap}")

            # Early stopping
            if loss.item() <= best_test_loss:
                stop_counter = 0
                best_test_loss = loss.item()
            else:
                stop_counter += 1

            metric_logger.log_to_wandb(epoch)
            best_test_accuracy = max(best_test_accuracy,
                                     metric_logger.metrics["train"].accuracy.compute().item())
            pbar.set_postfix(
                train_loss=loss.item(),
                train_accuracy=metric_logger.metrics["train"].accuracy.compute(
                ).item(),
                # test_loss=test_loss.item(),
                test_accuracy=metric_logger.metrics["test"].accuracy.compute(
                ).item()
            )

            if (early_stop_iters and stop_counter == early_stop_iters):
                break

    compute_log_confusion_matrix(labels, preds, train_mask, test_mask)

    return best_test_accuracy

from tqdm import tqdm
import torch
from metrics.metrics import LayerwiseGradNormMetric, MetricLogger, compute_log_confusion_matrix
import networkx as nx


def train(epochs, model, criterion, data, labels: torch.Tensor, graph: nx.Graph, optimizer, mask, metric_logger: MetricLogger, scheduler=None, early_stop_iters=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    metric = LayerwiseGradNormMetric(model)

    train_mask, test_mask = mask
    train_mask, test_mask = train_mask.to(
        device), test_mask.to(device)  # ПОФИКСИТЬ
    labels = labels.int()

    print("TRAIN: train mask edges", train_mask.int().sum().item())
    print("TRAIN: mask shape, train = ",
          train_mask.shape, " test = ", test_mask.shape)
    print("TRAIN: edges in train mask = ",
          labels[train_mask].int().sum().item())
    print("TRAIN: edges in test mask = ",
          labels[test_mask].int().sum().item())
    print("TRAIN: labels_type: ", labels.dtype)
    best_test_auc, stop_counter = 0, 0

    with tqdm(range(1, epochs + 1)) as pbar:
        for epoch in pbar:
            metric_logger.reset_all()

            model.train()
            optimizer.zero_grad()

            logits = model(data, graph)
            probs = torch.sigmoid(logits).detach()

            loss = criterion(logits[train_mask], labels[train_mask])
            # print("loss", loss.item())
            # acc = (preds[train_mask] == labels[train_mask]
            #        ).sum() / preds[train_mask].numel()
            # print("acc", acc.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            metric_logger.update(
                "train", logits[train_mask], labels[train_mask], loss.item())
            metric.update(epoch)

            model.eval()
            with torch.no_grad():

                # logits, _, _ = model(data, graph)
                # preds = (logits > 0).float()

                # test_loss = criterion(logits[test_mask], labels[test_mask])

                metric_logger.update(
                    "test", logits[test_mask], labels[test_mask], 0)

                # labels_1 = labels.detach().cpu().numpy()
                # probs = probs.detach().cpu().numpy()
                # preds = preds.detach().cpu().numpy()
                # test_mask_1 = test_mask.cpu().numpy()
                # print("labels", labels_1[test_mask_1].sum(),
                #       " / ", labels_1[test_mask_1].shape)
                # auc = roc_auc_score(labels_1[test_mask_1], probs[test_mask_1])
                # ap = average_precision_score(
                #     labels_1[test_mask_1], probs[test_mask_1])
                # acc = accuracy_score(labels_1[test_mask_1], preds[test_mask_1])

                # print(
                #     f"TRAIN: epoch {epoch} loss {loss.item()} test_acc {acc} test_auc {auc} test_ap {ap}")

            # Early stopping
            test_auc = metric_logger.metrics["test"].auroc.compute().item()
            if test_auc > best_test_auc:
                stop_counter = 0
                best_test_auc = test_auc
            else:
                stop_counter += 1

            metric_logger.log_to_wandb(epoch)

            pbar.set_postfix(
                train_loss=loss.item(),
                train_ap=metric_logger.metrics["train"].ap.compute().item(),
                train_auroc=metric_logger.metrics["train"].auroc.compute().item(),
                # train_accuracy=metric_logger.metrics["train"].accuracy.compute(
                # ).item(),
                # test_accuracy=metric_logger.metrics["test"].accuracy.compute(
                # ).item(),
                test_ap=metric_logger.metrics["test"].ap.compute().item(),
                test_auc=metric_logger.metrics["test"].auroc.compute().item()
            )

            if (early_stop_iters and stop_counter == early_stop_iters):
                break

    compute_log_confusion_matrix(labels, probs, train_mask, test_mask)

    return best_test_auc

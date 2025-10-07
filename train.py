from tqdm import tqdm
import torch
import logging
from torch_geometric.utils import negative_sampling


def train(epochs, model, data, optimizer,
          metric_logger, scheduler=None, early_stop_iters=0, min_iters=0, **cfg):
    """
    Тренировка модели с логгированием train/val/test метрик в Comet.
    Лучшие метрики определяются по val_auc.
    """

    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    best_val_auc, best_val_ap = 0, 0
    best_test_auc, best_test_ap = 0, 0
    best_epoch = 0
    best_state_dict = None
    stop_counter = 0

    with tqdm(range(1, epochs + 1)) as pbar:
        for epoch in pbar:
            metric_logger.reset_all()
            model.train()
            optimizer.zero_grad()

            # --- Encode ---
            z = model.encode(data)

            # --- Train loss ---
            loss = model.recon_loss(z, data.train_pos_edge_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # --- Train metrics ---
            pos_all_edge_index = torch.cat([
                data.train_pos_edge_index,
                data.val_pos_edge_index,
                data.test_pos_edge_index
            ], dim=1)

            train_neg_edge_index = negative_sampling(
                edge_index=pos_all_edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.train_pos_edge_index.size(1),
                force_undirected=True
            )

            pos_train_logits = model.decode(z, data.train_pos_edge_index)
            neg_train_logits = model.decode(z, train_neg_edge_index)
            train_logits = torch.cat(
                [pos_train_logits, neg_train_logits], dim=0)
            train_targets = torch.cat([
                torch.ones(pos_train_logits.size(0), device=device),
                torch.zeros(neg_train_logits.size(0), device=device)
            ]).long()
            metric_logger.update("train", train_logits,
                                 train_targets, loss.item())

            # --- Validation metrics ---
            pos_val_logits = model.decode(z, data.val_pos_edge_index)
            neg_val_logits = model.decode(z, data.val_neg_edge_index)
            val_logits = torch.cat([pos_val_logits, neg_val_logits], dim=0)
            val_targets = torch.cat([
                torch.ones(pos_val_logits.size(0), device=device),
                torch.zeros(neg_val_logits.size(0), device=device)
            ]).long()
            metric_logger.update("val", val_logits, val_targets, 0.0)

            # --- Compute val metrics ---
            val_auc = metric_logger.metrics["val"].auroc.compute().item()
            val_ap = metric_logger.metrics["val"].ap.compute().item()

            # --- Save best model by val AUC ---
            if val_auc > best_val_auc and epoch > min_iters:
                best_val_auc = val_auc
                best_val_ap = val_ap
                best_epoch = epoch
                best_state_dict = model.state_dict()
                stop_counter = 0
            else:
                stop_counter += 1

            metric_logger.log_to_wandb(epoch)

            pbar.set_postfix(
                train_loss=loss.item(),
                val_ap=val_ap,
                val_auc=val_auc
            )

            # --- Early stopping ---
            if early_stop_iters and epoch >= min_iters and stop_counter == early_stop_iters:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # --- Evaluate test metrics on best model ---
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.eval()
        with torch.no_grad():
            z = model.encode(data)

            pos_test_logits = model.decode(z, data.test_pos_edge_index)
            neg_test_logits = model.decode(z, data.test_neg_edge_index)
            test_logits = torch.cat([pos_test_logits, neg_test_logits], dim=0)
            test_targets = torch.cat([
                torch.ones(pos_test_logits.size(0), device=device),
                torch.zeros(neg_test_logits.size(0), device=device)
            ]).long()

            # вычисляем метрики напрямую, чтобы не перезаписывать logger
            probs = torch.sigmoid(test_logits)
            from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
            test_auc = BinaryAUROC().to(device)(probs, test_targets).item()
            test_ap = BinaryAveragePrecision().to(device)(probs, test_targets).item()

            best_test_auc = test_auc
            best_test_ap = test_ap

    # --- Log final results to Comet ---
    metric_logger.log_other("best_epoch", best_epoch)
    metric_logger.log_other("best_val_auc", best_val_auc)
    metric_logger.log_other("best_val_ap", best_val_ap)
    metric_logger.log_other("test_auc_at_best_val", best_test_auc)
    metric_logger.log_other("test_ap_at_best_val", best_test_ap)

    logger.info(f"Best epoch: {best_epoch} | val_auc={best_val_auc:.4f} | "
                f"val_ap={best_val_ap:.4f} | test_auc={best_test_auc:.4f} | test_ap={best_test_ap:.4f}")

    return best_val_auc

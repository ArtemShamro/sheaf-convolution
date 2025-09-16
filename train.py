from tqdm import tqdm
import torch
import logging
from torch_geometric.utils import negative_sampling


def train(epochs, model, criterion, data, optimizer,
          metric_logger, scheduler=None, early_stop_iters=0):
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: {}".format(device))

    best_val_auc, best_test_auc = 0, 0
    stop_counter = 0

    with tqdm(range(1, epochs + 1)) as pbar:
        for epoch in pbar:
            metric_logger.reset_all()
            model.train()
            optimizer.zero_grad()

            # --- Encode ---
            z = model.encode(data)

            # --- Loss ---
            loss = model.recon_loss(z, data.train_pos_edge_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            # --- Train metrics (сэмплируем негативные рёбра) ---

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
            metric_logger.update("test", val_logits, val_targets, 0.0)

            # --- Early stopping ---
            test_auc = metric_logger.metrics["test"].auroc.compute().item()
            if test_auc > best_val_auc:
                best_val_auc = test_auc
                best_test_auc = test_auc
                stop_counter = 0
            elif epoch > 20:
                stop_counter += 1

            metric_logger.log_to_wandb(epoch)

            pbar.set_postfix(
                train_loss=loss.item(),
                train_ap=metric_logger.metrics["train"].ap.compute().item(),
                train_auc=metric_logger.metrics["train"].auroc.compute(
                ).item(),
                val_ap=metric_logger.metrics["test"].ap.compute().item(),
                val_auc=metric_logger.metrics["test"].auroc.compute().item(),
            )

            if (early_stop_iters and stop_counter == early_stop_iters):
                logger.info("Early stopping triggered.")
                break

    return best_test_auc

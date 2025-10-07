# main.py
import comet_ml
from metrics.metrics import MetricLogger
from model.baseline_model import GAE
from utils import set_seed
from train import train
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate
import torch
from metrics.comet_logger import CometLogger
from comet_ml.integration.pytorch import log_model, watch
import logging


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.info(f'Hydra Config:\n {OmegaConf.to_yaml(cfg)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg.seed is not None:
        set_seed(cfg.seed_data)

    # --- Новый способ загрузки датасета через Hydra ---
    dataset = instantiate(cfg.dataset, device=device, seed=cfg.seed_data)
    data, in_channels = dataset.get_data()

    if cfg.seed is not None:
        set_seed(cfg.seed)
    # --- Инициализация модели ---
    model = instantiate(cfg.model, input_dim=in_channels,
                        device=device).to(device)
    model_name = model.__class__.__name__

    comet_logger = CometLogger()

    experiment = comet_logger.get_experiment()
    experiment.set_name(f"{model_name}_{dataset.__class__.__name__}")
    experiment.log_parameters(OmegaConf.to_container(
        cfg, resolve=True))  # type: ignore
    experiment.add_tags([model_name, dataset.__class__.__name__])
    if cfg.exp_group is not None:
        experiment.log_other("experiment_group", cfg.exp_group)

    comet_logger.log_model_info(model)
    comet_logger.log_dataset_info(dataset)

    log_model(experiment, model=model, model_name="TheModel")
    watch(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    metric_logger = MetricLogger(device, comet_logger=comet_logger)

    best_val_auc = train(
        cfg.epochs,
        model,
        data.to(device),
        optimizer,
        metric_logger=metric_logger,
        early_stop_iters=cfg.optimizer.early_stop_iters,
        scheduler=scheduler,
        min_iters=cfg.optimizer.min_iters,
        log_epoch=cfg.log_epoch
    )
    comet_logger.end()
    return best_val_auc


if __name__ == "__main__":
    main()

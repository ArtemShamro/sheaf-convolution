import comet_ml
from metrics.metrics import MetricLogger
from model.baseline_model import GAE
from dataloader.dataloader import generate_dataset
from utils import set_seed, CustomBCELoss
from train import train
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.utils import instantiate
import torch
from metrics.comet_logger import get_experiment
from comet_ml.integration.pytorch import log_model, watch
import logging


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    logger.info(f'Hydra Config:\n {OmegaConf.to_yaml(cfg)}')

    # Конфиг модели
    # model_config = ModelDiffusionConfig(
    #     **OmegaConf.to_container(cfg.model, resolve=True))  # type:ignore

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_config.device = device

    if cfg.seed != 0:
        set_seed(cfg.seed)

    # Загружаем датасет (только data и число признаков)
    data, in_channels = generate_dataset(
        name=cfg.dataset.name,
        device=device
    )

    model = instantiate(cfg.model, input_dim=in_channels,
                        device=device).to(device)
    model_name = model.__class__.__name__

    experiment = get_experiment()
    experiment.set_name(f"{model_name}_{cfg.dataset.name}")
    experiment.log_parameters(OmegaConf.to_container(
        cfg, resolve=True))  # type:ignore
    experiment.add_tags([model_name, cfg.dataset.name])

    log_model(experiment, model=model, model_name="TheModel")
    watch(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs)

    metric_logger = MetricLogger(device)

    # Обучение
    best_test_accuracy = train(
        cfg.epochs,
        model,
        CustomBCELoss(print_loss=False),
        data.to(device),              # <--- теперь только data
        optimizer,
        metric_logger=metric_logger,
        early_stop_iters=cfg.optimizer.early_stop_iters,
        scheduler=scheduler
    )

    return best_test_accuracy


if __name__ == '__main__':
    main()

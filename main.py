import comet_ml
from model.model import Diffusion, ModelDiffusionConfig
from metrics.metrics import MetricLogger
from model.baseline_model import GAE
from dataloader.dataloader import generate_dataset
from config.model_config import ModelDiffusionConfig
from utils import set_seed, get_adj_mat, CustomBCELoss
from train import train
from omegaconf import OmegaConf, DictConfig
import hydra
from dataclasses import replace
import torch.nn as nn
import torch
from metrics.comet_logger import get_experiment
from comet_ml.integration.pytorch import log_model
from comet_ml.integration.pytorch import watch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('Hydra Config:\n')
    print(OmegaConf.to_yaml(cfg))
    experiment = get_experiment()
    experiment.set_name(f"{cfg.model.type}_{cfg.dataset.name}")

    experiment.log_parameters(OmegaConf.to_container(cfg, resolve=True))

    experiment.add_tags([cfg.model.type, cfg.dataset.name])

    model_config = ModelDiffusionConfig(
        **OmegaConf.to_container(cfg.model, resolve=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config.device = device

    set_seed(cfg.seed)

    G, data, labels, train_mask, test_mask = generate_dataset(name=cfg.dataset.name,
                                                              test_size=cfg.dataset.test_size,
                                                              ndata=cfg.dataset.ndata,
                                                              dimx=cfg.dataset.dimx)

    criterion = CustomBCELoss(print_loss=False)
    model_config = replace(
        model_config,
        output_dim=1,
        input_dim=data.shape[1]
    )

    model = Diffusion(model_config).to(
        device) if cfg.model.type == "diffusion" else GAE(config=model_config).to(device)
    print("MAIN : model = ", type(model))
    log_model(experiment, model=model, model_name="TheModel")
    watch(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=300)

    metric_logger = MetricLogger(device)

    best_test_accuracy = train(
        cfg.epochs, model, criterion,
        data.to(device), labels.to(device), G, optimizer,
        mask=(train_mask, test_mask),
        metric_logger=metric_logger,
        early_stop_iters=cfg.optimizer.early_stop_iters,
        scheduler=scheduler
    )  # scheduler=scheduler

    for name, param in model.named_parameters():
        if "alpha" in name:
            print(f"{name}: {param.data:.2f}")

    return best_test_accuracy


if __name__ == '__main__':
    main()

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
import wandb
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print('Hydra Config:\n')
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project="diffusion-gnn",
        name=f"{cfg.dataset.name}_{cfg.model.task}",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[cfg.model.task, cfg.dataset.name]
    )

    model_config = ModelDiffusionConfig(
        **OmegaConf.to_container(cfg.model, resolve=True))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config.device = device

    set_seed(cfg.seed)

    # заменить labels на true_adj_mat, возвращаемую из generate_dataset
    G, data, labels, train_mask, test_mask = generate_dataset(name=cfg.dataset.name,
                                                              task=cfg.model.task,
                                                              test_size=cfg.dataset.test_size,
                                                              ndata=cfg.dataset.ndata,
                                                              dimx=cfg.dataset.dimx)

    match cfg.model.task:
        case "edges_prediction":
            criterion = CustomBCELoss(print_loss=False)
            # criterion = nn.BCEWithLogitsLoss(pos_weight=650)
            model_config = replace(
                model_config,
                output_dim=1,
                input_dim=data.shape[1]
            )

        case "node_classification":
            criterion = nn.CrossEntropyLoss()
            model_config = replace(
                model_config,
                output_dim=labels.unique().shape[0],
                input_dim=data.shape[1]
            )

    model = Diffusion(model_config).to(
        device) if cfg.model.type == "diffusion" else GAE(config=model_config).to(device)

    wandb.watch(model, log="all", log_freq=50)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        amsgrad=True
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    metric_logger = MetricLogger(device)

    best_test_accuracy = train(
        cfg.epochs, model, criterion,
        data.to(device), labels.to(device), G, optimizer,
        mask=(train_mask, test_mask),
        metric_logger=metric_logger,
        early_stop_iters=cfg.optimizer.early_stop_iters
    )  # scheduler=scheduler

    for name, param in model.named_parameters():
        if "alpha" in name:
            print(f"{name}: {param.data:.2f}")

    wandb.finish()

    return best_test_accuracy


if __name__ == '__main__':
    main()

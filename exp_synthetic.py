
from hydra import initialize, compose
from main import main as train_main

MODELS = ["GAE", "Diffusion"]
SEEDS = [1, 2, 3]
SEED_DATA = 11

BASE_SYNTHETIC = {
    "ndata": 2000,
    "dimx": 16,
    "nproj": 4,
}

BASE_RUNTIME = {
    "GAE": {
        "epochs": 1000,
        "optimizer.lr": 8e-3,
        "optimizer.weight_decay": 1e-5,
        "model.dropout": 0.1, },
    "Diffusion": {
        "epochs": 1000,
        "optimizer.lr": 8e-3,
        "optimizer.weight_decay": 3e-5,
        "model.dropout": 0.1, }
}


def _compose_overrides(model_name: str, dataset_kwargs: dict, seed_train: int, exp_group: str):
    overrides = [
        f"model={model_name}",
        f"seed_data={SEED_DATA}",
        f"seed={seed_train}",
        f"exp_group={exp_group}",
    ]

    for k, v in dataset_kwargs.items():
        overrides.append(f"dataset.{k}={v}")

    for k, v in BASE_RUNTIME[model_name].items():
        if v is not None:
            overrides.append(f"{k}={v}")

    overrides.append(
        "dataset._target_=dataloader.dataloader_synthetic.SyntheticDataset")

    return overrides


def _run_single(cfg_overrides):
    with initialize(config_path="config", version_base=None):
        cfg = compose(config_name="config", overrides=cfg_overrides)

    best_val_auc = train_main(cfg)
    return float(best_val_auc)


def experiment_size_effect():
    """
    1) Влияние размерности графа (n_data)
       n_data = 200, 1000, 2000, 5000, 10000
       n_proj = 4, d_x = 16
    """
    exp_name = "size_effect"
    levels = [200, 1000, 2000, 5000, 10000]

    for model in MODELS:
        for n_data in levels:
            dataset_kwargs = dict(BASE_SYNTHETIC)
            dataset_kwargs.update({"ndata": n_data, "dimx": 16, "nproj": 4})

            exp_group = exp_name
            for s in SEEDS:
                overrides = _compose_overrides(
                    model, dataset_kwargs, s, exp_group)
                _run_single(overrides)


def experiment_featdim_effect():
    """
    2) Влияние размерности признаков (d_x)
       n_data = 2000, n_proj = 4, d_x = 4, 8, 16, 32, 64
    """
    exp_name = "featdim_effect"
    levels = [4, 8, 16, 32, 64]

    for model in MODELS:
        for d_x in levels:
            dataset_kwargs = dict(BASE_SYNTHETIC)
            dataset_kwargs.update({"ndata": 2000, "dimx": d_x, "nproj": 4})

            exp_group = exp_name
            for s in SEEDS:
                overrides = _compose_overrides(
                    model, dataset_kwargs, s, exp_group)
                _run_single(overrides)


def experiment_nproj_effect():
    """
    3) Влияние n_proj
       n_data = 2000, n_proj = 2, 4, 6, 8, 10, d_x = 16
    """
    exp_name = "nproj_effect"
    levels = [2, 4, 6, 8, 10]

    for model in MODELS:
        for n_proj in levels:
            dataset_kwargs = dict(BASE_SYNTHETIC)
            dataset_kwargs.update({"ndata": 2000, "dimx": 16, "nproj": n_proj})

            exp_group = exp_name
            for s in SEEDS:
                overrides = _compose_overrides(
                    model, dataset_kwargs, s, exp_group)
                _run_single(overrides)


def main():
    print("=== Synthetic experiments launcher ===")
    print("Models:", MODELS)
    print("Seeds per setting:", SEEDS)
    print("Fixed dataset seed (seed_data):", SEED_DATA)
    print("Base dataset params:", BASE_SYNTHETIC)
    print()

    experiment_size_effect()
    experiment_featdim_effect()
    experiment_nproj_effect()


if __name__ == "__main__":
    main()

# exp_real.py
import torch
from hydra import initialize, compose
from main import main as train_main

MODELS = ["GAE", "Diffusion"]
SEEDS = [1, 2, 3]
SEED_DATA = 11

REAL_GRAPHS = [
    "roman-empire",
    "amazon-ratings",
    "minesweeper",
    "tolokers",
    "questions",
]

# --- Базовые гиперпараметры (подобраны аналогично синтетическим) ---
BASE_RUNTIME = {
    "GAE": {
        "epochs": 2000,
        "optimizer.lr": 8e-3,
        "optimizer.weight_decay": 1e-5,
        "model.dropout": 0.1,
    },
    "Diffusion": {
        "epochs": 2000,
        "optimizer.lr": 8e-3,
        "optimizer.weight_decay": 3e-5,
        "model.dropout": 0.1,
    },
}


def _compose_overrides(model_name: str, dataset_name: str, seed_train: int, exp_group: str):
    overrides = [
        f"model={model_name}",
        f"seed_data={SEED_DATA}",
        f"seed={seed_train}",
        f"exp_group={exp_group}",
        f"dataset.name={dataset_name}",
        "dataset._target_=dataloader.dataloader_hetero.HeteroGraphDataset",
        "dataset.val_ratio=0.1",
        "dataset.test_ratio=0.1",
    ]

    for k, v in BASE_RUNTIME[model_name].items():
        overrides.append(f"{k}={v}")

    return overrides


def _run_single(cfg_overrides):
    """Запускает один эксперимент с защитой от OOM."""
    try:
        with initialize(config_path="config", version_base=None):
            cfg = compose(config_name="config", overrides=cfg_overrides)

        print(
            f"\n[RUN] {cfg.model._target_} on {cfg.dataset.name}, seed={cfg.seed}")
        best_val_auc = train_main(cfg)
        torch.cuda.empty_cache()
        return float(best_val_auc)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(
                f"[SKIP] OOM on {cfg.dataset.name} with {cfg.model._target_} (seed={cfg.seed})")
            torch.cuda.empty_cache()
            return None
        else:
            raise e


def run_real_experiments():
    print("=== Real Graph Experiments Launcher ===")
    print("Models:", MODELS)
    print("Graphs:", REAL_GRAPHS)
    print("Seeds:", SEEDS)
    print("Fixed seed_data:", SEED_DATA)
    print()

    for dataset_name in REAL_GRAPHS:
        exp_group = f"real_{dataset_name}"

        for model in MODELS:
            for s in SEEDS:
                overrides = _compose_overrides(
                    model, dataset_name, s, exp_group)
                _run_single(overrides)


def main():
    run_real_experiments()


if __name__ == "__main__":
    main()

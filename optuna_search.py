import os
import optuna
import torch
from hydra import initialize, compose
from main import main as train_main
from optuna.samplers import TPESampler

# === Настройки ===
MODELS = ["GAE", "Diffusion"]
SEED_DATA = 11
INIT_SEEDS = [1, 2]
N_TRIALS = 60


def objective(trial, model_name):
    # Оптимизируемые параметры
    lr = trial.suggest_loguniform("optimizer.lr", 1e-4, 1e-2)
    dropout = trial.suggest_uniform("model.dropout", 0.0, 0.5)
    weight_decay = trial.suggest_loguniform(
        "optimizer.weight_decay", 1e-6, 1e-2)

    aucs = []

    for seed in INIT_SEEDS:
        # === Используем Hydra Compose API ===
        # Это то же самое, что "python main.py model=GAE optimizer.lr=... model.dropout=..."
        with initialize(config_path="config", version_base=None):
            cfg = compose(
                config_name="config",  # главный YAML-файл (config.yaml)
                overrides=[
                    f"model={model_name}",
                    f"optimizer.lr={lr}",
                    f"optimizer.weight_decay={weight_decay}",
                    f"model.dropout={dropout}",
                    f"seed_data={SEED_DATA}",
                    f"seed={seed}",
                ],
            )

        # Запуск train_main с готовым Hydra-конфигом
        best_val_auc = train_main(cfg)
        aucs.append(best_val_auc)

    mean_auc = sum(aucs) / len(aucs)
    return mean_auc


if __name__ == "__main__":
    os.makedirs("optuna_logs", exist_ok=True)

    for model_name in MODELS:
        db_path = f"optuna_logs/{model_name.lower()}_tuning.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        storage = f"sqlite:///./{db_path}"

        sampler = TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=SEED_DATA
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_tuning_seed{SEED_DATA}",
            storage=storage,
            load_if_exists=True,
            sampler=sampler
        )

        print(f"🚀 Оптимизация для {model_name}")
        study.optimize(lambda trial: objective(trial, model_name),
                       n_trials=N_TRIALS,
                       gc_after_trial=True)

        print(f"\n🏁 Завершено: {model_name}")
        print(f"Лучшие параметры: {study.best_trial.params}")
        print(f"Лучший val_auc: {study.best_value:.4f}")

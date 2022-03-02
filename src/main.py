from __future__ import annotations

import functools
import os

import numpy as np
import optuna
import torch
from torch.utils import data

import config
import dataset
import training


def objective(trial: optuna.trial.Trial, dataset: data.Dataset) -> np.float32:
    params = {
        "conv1_out": trial.suggest_int("conv1_out", 16, 64, step=8),
        "conv2_out": trial.suggest_int("conv2_out", 16, 64, step=8),
        "conv3_out": trial.suggest_int("conv3_out", 16, 64, step=8),
        "conv4_out": trial.suggest_int("conv4_out", 16, 64, step=8),
        "linear1_out": trial.suggest_int("linear1_out", 16, 64, step=8),
        "linear2_out": trial.suggest_int("linear2_out", 16, 64, step=8),
        "linear3_out": trial.suggest_int("linear3_out", 16, 64, step=8),
        "kernel_size1": trial.suggest_categorical("kernel_size1", [3, 5, 7]),
        "kernel_size2": trial.suggest_categorical("kernel_size2", [3, 5, 7]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.7),
        "early_stopping_rounds": trial.suggest_categorical(
            "early_stopping_rounds", [5, 10, 15, 20]
        ),
    }

    return training.train(dataset, params)


def hyperparameter_search(train_data: data.Dataset) -> optuna.Study:
    objective_ = functools.partial(objective, dataset=train_data)

    pruner = optuna.pruners.SuccessiveHalvingPruner()
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)

    study = optuna.create_study(direction="minimize", pruner=pruner, sampler=sampler)

    study.optimize(objective_, n_trials=20, gc_after_trial=True)

    return study


def main() -> None:
    dataset.dir_to_csv("train", "train_data.csv")
    dataset.dir_to_csv("test1", "test_data.csv", has_labels=False)
    dataset.order_test_data("test_data.csv")

    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)

    if not os.path.exists(config.INFERENCE_DIR):
        os.mkdir(config.INFERENCE_DIR)

    torch.manual_seed(42)

    dataset_path = os.path.join(config.DATA_DIR, "train_data.csv")
    transform = dataset.get_transforms()
    train_data = dataset.CatsDogsDataset(csv=dataset_path, transform=transform)

    study = hyperparameter_search(train_data=train_data)

    best_trial = study.best_trial

    print(f"Best parameters:{best_trial.params}")

    val_loss = training.train(
        dataset=train_data, params=best_trial.params, save_model=True
    )

    print(f"Avg. validation loss:{val_loss:.4f}")

    dataset_path = os.path.join(config.DATA_DIR, "test_data.csv")
    transform = dataset.get_transforms()
    test_data = dataset.CatsDogsDataset(csv=dataset_path, transform=transform)

    training.make_inference(data=test_data, params=best_trial.params)


if __name__ == "__main__":
    main()

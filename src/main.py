from __future__ import annotations

import functools
import glob
import os
from collections import defaultdict
from typing import Any

import albumentations as A
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data

import config
import dataset as ds
import model as mod
import training as tr
from console import console


def make_metric_string(metrics: dict[str, float]) -> str:
    fmt = "[red]{metric}[/red]: [green bold]{value: .4f}[/green bold]"
    gen = (fmt.format(metric=metric, value=value) for metric, value in metrics.items())
    return "; ".join(gen)


def get_model_params(params: dict[str, Any]) -> dict[str, Any]:
    dict_params = {}

    dict_params["conv_out_channels"] = [
        params["conv1_out"],
        params["conv2_out"],
        params["conv3_out"],
        params["conv4_out"],
    ]

    k1, k2 = params["kernel_size1"], params["kernel_size2"]

    dict_params["kernel_sizes"] = [k1, k1, k2, k2]

    dict_params["linear_out_features"] = [
        params["linear1_out"],
        params["linear2_out"],
        params["linear3_out"],
    ]

    dict_params["dropout"] = params["dropout"]

    return dict_params


def train_one_fold(
    dataset: data.Dataset,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    fold: int,
    params: dict[str, Any],
    *,
    save_model: bool = False,
) -> float:
    key = fold + 1

    console.print(f"[green]Fold {key}[/green]", justify="center")

    train_sampler = data.SubsetRandomSampler(train_ids)
    val_sampler = data.SubsetRandomSampler(val_ids)

    batch_size = params["batch_size"]

    train_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = mod.CatsDogsModel(n_targets=1, **get_model_params(params))
    model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config.COSINE_ANNEALING_T0
    )

    engine = tr.Engine(model=model, optimizer=optimizer, scheduler=scheduler)
    engine.reset_model_weights()

    best_loss = np.inf
    counter = 0
    history = defaultdict(list)

    for epoch in range(config.EPOCHS):
        console.print(
            f"[green]Epoch {epoch + 1}/{config.EPOCHS}[/green]", justify="center"
        )

        t_loss, t_acc = engine.train(data_loader=train_loader, epoch_num=epoch)
        val_loss, val_acc = engine.evaluate(data_loader=val_loader)

        console.print(
            make_metric_string(
                {
                    "Training loss": t_loss,
                    "Training accuracy": t_acc,
                    "Validation loss": val_loss,
                    "Validation accuracy": val_acc,
                }
            )
        )

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_loss:
            best_loss = val_loss
            if save_model is True:
                engine.save_model(f"model-{key}.pth")
        else:
            counter += 1

        if counter > config.EARLY_STOPPING_ROUNDS:
            break

    console.print(history)

    return best_loss


def train(
    params: dict[str, Any], csv_file: str, *, save_model: bool = False
) -> np.float32:
    k_fold = KFold(n_splits=config.FOLDS, shuffle=True, random_state=42)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(always_apply=True),
        ]
    )

    dataset_path = os.path.join(config.DATA_DIR, csv_file)

    train_data = ds.CatsDogsDataset(
        csv=dataset_path,
        transform=transform,
        resize=(config.IMG_HEIGHT, config.IMG_WIDTH),
    )

    losses = []

    for fold, (train_ids, val_ids) in enumerate(k_fold.split(train_data)):
        fold_loss = train_one_fold(
            dataset=train_data,
            train_ids=train_ids,
            val_ids=val_ids,
            fold=fold,
            params=params,
            save_model=save_model,
        )
        losses.append(fold_loss)

    return np.mean(losses, dtype=np.float32)


def objective(trial: optuna.trial.Trial, csv_file: str) -> np.float32:
    params = {
        "conv1_out": trial.suggest_int("conv1_out", 16, 64),
        "conv2_out": trial.suggest_int("conv2_out", 16, 64),
        "conv3_out": trial.suggest_int("conv3_out", 16, 64),
        "conv4_out": trial.suggest_int("conv4_out", 16, 64),
        "linear1_out": trial.suggest_int("linear1_out", 16, 64),
        "linear2_out": trial.suggest_int("linear2_out", 16, 64),
        "linear3_out": trial.suggest_int("linear3_out", 16, 64),
        "kernel_size1": trial.suggest_categorical("kernel_size1", [3, 5]),
        "kernel_size2": trial.suggest_categorical("kernel_size2", [3, 5]),
        "lr": trial.suggest_loguniform("lr", 1e-6, 1e-3),
        "batch_size": trial.suggest_int("batch_size", 64, 256),
        "dropout": trial.suggest_uniform("dropout", 0.1, 0.7),
    }

    return train(params, csv_file)


def make_inference(csv_file: str) -> None:
    transform = A.Compose([A.Normalize(always_apply=True)])

    test_data_path = os.path.join(config.DATA_DIR, csv_file)

    test_data = ds.CatsDogsDataset(
        csv=test_data_path,
        transform=transform,
        resize=(config.IMG_HEIGHT, config.IMG_WIDTH),
        labels=False,
    )

    model_loader = (
        functools.partial(torch.load, map_location="cpu")
        if torch.cuda.is_available() is False
        else torch.load
    )

    models_path = os.path.join(config.MODEL_DIR, "*.pth")

    for fold, model_path in enumerate(glob.glob(models_path)):
        model = mod.CatsDogsModel()

        state_dict = model_loader(model_path)
        model.load_state_dict(state_dict)
        model.to(config.DEVICE)
        model.eval()

        data_loader = data.DataLoader(
            test_data, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
        )

        df_dict = defaultdict(list)

        for idx, batch in enumerate(data_loader, 1):
            for k, v in batch.items():
                batch[k] = v.to(config.DEVICE)

            preds = model(image=batch["image"])

            df_dict["id"].append(idx)
            df_dict["label"].append((preds > 0.5).item())

        df = pd.DataFrame.from_dict(df_dict)
        path = os.path.join(config.INFERENCE_DIR, f"model-fold{fold + 1}.csv")
        df.to_csv(path, index=False)


def main() -> None:
    ds.dir_to_csv("train", "train_data.csv")
    ds.dir_to_csv("test1", "test_data.csv", has_labels=False)
    ds.order_test_data("test_data.csv")

    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)

    if not os.path.exists(config.INFERENCE_DIR):
        os.mkdir(config.INFERENCE_DIR)

    torch.manual_seed(42)

    objective_ = functools.partial(objective, csv_file="train_data.csv")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_, n_trials=20)

    console.print("Best Trial:")
    best_trial = study.best_trial

    print(best_trial.values)
    print(best_trial.params)

    train(best_trial.params, save_model=True)


if __name__ == "__main__":
    main()

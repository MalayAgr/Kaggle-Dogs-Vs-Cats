from __future__ import annotations

import csv
import functools
import glob
import os
from collections import defaultdict

import albumentations as A
import pandas as pd
import torch
from rich import print
from rich.rule import Rule
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

import config
import dataset as ds
import model as mod
import training as tr


def train_one_fold(dataset, loss_fn, train_ids, val_ids, fold):
    key = fold + 1

    print(Rule(f"[green bold]Fold {key}[/green bold]"))

    train_sampler = data.SubsetRandomSampler(train_ids)
    val_sampler = data.SubsetRandomSampler(val_ids)

    train_loader = data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = data.DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        sampler=val_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = mod.CatsDogsModel()
    model.to(config.DEVICE)
    model.apply(mod.reset_model_weights)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config.COSINE_ANNEALING_T0
    )

    print(Rule("[green bold]Training[/green bold]"))

    train_loss, train_history = tr.train_loop(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
    )

    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"model-fold{key}.pth"))

    print(Rule("[green bold]Validating[/green bold]"))

    val_loss, val_history = tr.validation_loop(
        model=model, data_loader=val_loader, loss_fn=loss_fn
    )

    print(
        f"[green]Train loss: {train_loss: .3f}, Validation loss: {val_loss: .3f}[green]"
    )

    return {"train": train_history, "val": val_history}


def train():
    k_fold = KFold(n_splits=config.FOLDS, shuffle=True)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(always_apply=True),
        ]
    )

    dataset = ds.CatsDogsDataset(
        csv="data/train_data.csv",
        transform=transform,
        resize=(config.IMG_HEIGHT, config.IMG_WIDTH),
    )

    loss_fn = nn.BCELoss()

    fold_history = {}

    for fold, (train_ids, val_ids) in enumerate(k_fold.split(dataset)):
        fold_history[f"fold{fold + 1}"] = train_one_fold(
            dataset=dataset,
            loss_fn=loss_fn,
            train_ids=train_ids,
            val_ids=val_ids,
            fold=fold,
        )

    return fold_history


def make_inference():
    transform = A.Compose([A.Normalize(always_apply=True)])

    test_data = ds.CatsDogsDataset(
        "data/test_data.csv",
        transform=transform,
        resize=(config.IMG_HEIGHT, config.IMG_WIDTH),
        labels=False,
    )

    model_loader = (
        functools.partial(torch.load, map_location="cpu")
        if torch.cuda.is_available() is False
        else torch.load
    )

    for fold, model_path in enumerate(glob.glob("models/*.pth")):
        model = mod.CatsDogsModel()

        state_dict = model_loader(model_path)
        model.load_state_dict(state_dict)
        model.to(config.DEVICE)
        model.eval()

        data_loader = data.DataLoader(test_data)

        df_dict = defaultdict(list)

        for idx, batch in enumerate(data_loader, 1):
            for k, v in batch.items():
                batch[k] = v.to(config.DEVICE)

            preds = model(image=batch["image"])

            df_dict["id"].append(idx)
            df_dict["label"].append((preds > 0.5).item())

        df = pd.DataFrame.from_dict(df_dict)
        df.to_csv(f"inferences/model-fold{fold + 1}.csv", index=False)


def main():
    ds.dir_to_csv("train", "train_data.csv")
    ds.dir_to_csv("test1", "test_data.csv", has_labels=False)
    ds.order_test_data("test_data.csv")

    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)

    if not os.path.exists(config.INFERENCE_DIR):
        os.mkdir(config.INFERENCE_DIR)

    torch.manual_seed(42)

    history = train()
    make_inference()


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import glob
import os

import albumentations as A
import torch
from rich import print
from rich.rule import Rule
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data

import config
from dataset import CatsDogsDataset
from model import CatsDogsModel
from training import train_one_epoch, validate_one_epoch


def reset_model_weights(model: nn.Module):
    for layer in model.children():
        try:
            layer.reset_parameters()
        except AttributeError:
            pass


def dir_to_csv(dir_name, dest):
    path = os.path.join(config.DATA_DIR, dir_name, "*.jpg")
    target = os.path.join(config.DATA_DIR, dest)

    with open(target, mode="w+") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label"])
        writer.writeheader()

        label_map = config.LABEL_MAP

        writer.writerows(
            [
                {
                    "filename": filename,
                    "label": label_map["cat" if "cat" in filename else "dog"],
                }
                for filename in glob.glob(path)
            ]
        )


def train(model: nn.Module, data_loader, optimizer, loss_fn, scheduler=None):
    model.train()

    history = {}

    for epoch in range(config.EPOCHS):
        loss, epoch_history = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch_num=epoch,
            scheduler=scheduler,
        )
        history[f"epoch{epoch + 1}"] = epoch_history

    return loss, history


def validate(model: nn.Module, data_loader, loss_fn):
    model.eval()
    history = {}
    for epoch in range(config.EPOCHS):
        loss, epoch_history = validate_one_epoch(
            model=model, data_loader=data_loader, loss_fn=loss_fn, epoch_num=epoch
        )
        history[f"epoch{epoch + 1}"] = epoch_history
    return loss, history


def train_one_fold(dataset, loss_fn, train_ids, val_ids, fold):
    key = fold + 1

    print(Rule(f"[green bold]Fold {key}[/green bold]"))

    train_sampler = data.SubsetRandomSampler(train_ids)
    val_sampler = data.SubsetRandomSampler(val_ids)

    train_loader = data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, sampler=train_sampler
    )
    val_loader = data.DataLoader(
        dataset, batch_size=config.BATCH_SIZE, sampler=val_sampler
    )

    model = CatsDogsModel()
    model.apply(reset_model_weights)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config.COSINE_ANNEALING_T0
    )

    print(Rule("[green bold]Training[/green bold]"))

    train_loss, train_history = train(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
    )

    torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, f"model-fold{key}"))

    print(Rule("[green bold]Validating[/green bold]"))

    val_loss, val_history = validate(
        model=model, data_loader=val_loader, loss_fn=loss_fn
    )

    print(
        f"[green]Train loss: {train_loss: .3f}, Validation loss: {val_loss: .3f}[green]"
    )

    return {"train": train_history, "val": val_history}


def main():
    dir_to_csv("train", "train_data.csv")

    torch.manual_seed(42)

    k_fold = KFold(n_splits=config.FOLDS, shuffle=True)

    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(always_apply=True),
        ]
    )

    dataset = CatsDogsDataset(
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


if __name__ == "__main__":
    history = main()
    print(history)

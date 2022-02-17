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
from dataset import CatsDogsDataset
from model import CatsDogsModel
from training import train_one_epoch, validate_one_epoch



def dir_to_csv(dir_name, dest, has_labels=True):
    def with_labels(path):
        label_map = config.LABEL_MAP
        yield from (
            {
                "filename": filename,
                "label": label_map["cat" if "cat" in filename else "dog"],
            }
            for filename in glob.glob(path)
        )

    path = os.path.join(config.DATA_DIR, dir_name, "*.jpg")
    target = os.path.join(config.DATA_DIR, dest)

    with open(target, mode="w+") as f:

        fieldnames = ["filename"]

        if has_labels is True:
            fieldnames.append("label")
            rows = with_labels(path)
        else:
            rows = ({"filename": filename} for filename in glob.glob(path))

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def reset_model_weights(model: nn.Module):
    for layer in model.children():
        try:
            layer.reset_parameters()
        except AttributeError:
            pass


def train_loop(model: nn.Module, data_loader, optimizer, loss_fn, scheduler=None):
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


def validation_loop(model: nn.Module, data_loader, loss_fn):
    model.eval()
    history = {}
    loss, epoch_history = validate_one_epoch(
        model=model, data_loader=data_loader, loss_fn=loss_fn, epoch_num=0
    )
    history[f"epoch1"] = epoch_history
    return loss, history


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

    model = CatsDogsModel()
    model.to(config.DEVICE)
    model.apply(reset_model_weights)

    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config.COSINE_ANNEALING_T0
    )

    print(Rule("[green bold]Training[/green bold]"))

    train_loss, train_history = train_loop(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
    )

    torch.save(
        model.state_dict(), os.path.join(config.MODEL_DIR, f"model-fold{key}.pth")
    )

    print(Rule("[green bold]Validating[/green bold]"))

    val_loss, val_history = validation_loop(
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


def make_inference():
    if not os.path.exists("inferences"):
        os.mkdir("inferences")

    transform = A.Compose([A.Normalize(always_apply=True)])

    test_data = CatsDogsDataset(
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
        model = CatsDogsModel()

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
    dir_to_csv("train", "train_data.csv")

    if not os.path.exists(config.MODEL_DIR):
        os.mkdir(config.MODEL_DIR)

    torch.manual_seed(42)

    history = train()

    dir_to_csv("test1", "test_data.csv", has_labels=False)

    df: pd.DataFrame = pd.read_csv("data/test_data.csv")

    target_names = []
    for filename in df["filename"]:
        basename = os.path.basename(filename)
        name, _ = os.path.splitext(basename)
        target_names.append(int(name))

    df["target_name"] = target_names
    df = df.sort_values(by=["target_name"])
    df = df.drop("target_name", axis=1)
    df.to_csv("data/test_data.csv", index=False)

    make_inference()


if __name__ == "__main__":
    main()

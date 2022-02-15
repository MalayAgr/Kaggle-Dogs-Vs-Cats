import csv
import glob
import os

import albumentations as A
import torch
from sklearn.model_selection import KFold
from torch import nn, optim
from torch.utils import data

import config
from dataset import CatsDogsDataset
from model import CatsDogsModel
from training import train_one_epoch


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


def train(model, data_loader, optimizer, scheduler=None):
    for epoch in range(config.EPOCHS):
        loss = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        print(f"Loss at the end of {epoch + 1}: {loss: .3f}")


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

    for fold, (train_ids, val_ids) in enumerate(k_fold.split(dataset)):
        print(f"Fold {fold + 1}")
        print("-" * 10)

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

        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        loss = train(model=model, data_loader=train_loader, optimizer=optimizer)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import glob
import os

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


class CatsDogsDataset(Dataset):
    def __init__(
        self,
        csv: str,
        transform: A.Compose = None,
        labels: bool = True,
    ):
        self.df: pd.DataFrame = pd.read_csv(csv)
        self.transform = transform
        self.labels = labels

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Load image path and label
        if isinstance(idx, slice):
            raise TypeError(f"{self.__class__.__name__} doesn't support slicing.")

        img = self.df.iloc[idx, 0]

        # Read image and convert to NumPy array
        img = Image.open(img)
        img = np.array(img)

        # Apply transformations
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        # Move channels in front
        img = np.swapaxes(img, -1, 0)
        img = torch.tensor(img, dtype=torch.float32)

        sample = {"image": img}

        # Add label to sample if required
        if self.labels is True:
            label = self.df.iloc[idx, 1]
            sample["label"] = torch.tensor(label, dtype=torch.float32)

        return sample


def dir_to_csv(dir_name: str, dest: str, has_labels: bool = True) -> None:
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


def order_test_data(csv_path: str) -> None:
    path = os.path.join(config.DATA_DIR, csv_path)

    df: pd.DataFrame = pd.read_csv(path)

    target_names = []

    for filename in df["filename"]:
        # Take out filename from full path
        basename = os.path.basename(filename)

        # Remove extension
        name, _ = os.path.splitext(basename)

        target_names.append(int(name))

    df["target_name"] = target_names

    # Sort by IDs
    df = df.sort_values(by=["target_name"])
    df = df.drop("target_name", axis=1)

    # Overwrite existing file
    df.to_csv(path, index=False)


def get_transforms(only_normalize: bool = False) -> A.Compose:
    if only_normalize is True:
        return A.Compose([A.Normalize(always_apply=True)])

    return A.Compose(
        [
            A.HorizontalFlip(p=0.4),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(always_apply=True),
        ]
    )

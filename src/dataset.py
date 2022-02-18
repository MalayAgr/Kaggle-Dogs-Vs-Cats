from __future__ import annotations

import csv
import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import config


class CatsDogsDataset(Dataset):
    def __init__(
        self, csv, transform=None, resize: tuple[int, int] = None, labels=True
    ):
        self.df: pd.DataFrame = pd.read_csv(csv)
        self.transform = transform
        self.resize = resize
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image path and label
        img = self.df.iloc[idx, 0]

        # Read image
        img = Image.open(img)

        # Resize if required
        if self.resize is not None:
            width, height = reversed(self.resize)
            img = img.resize((width, height), resample=Image.BILINEAR)

        # Convert to NumPy array
        img = np.array(img)

        # Apply transformations
        if self.transform is not None:
            img = self.transform(image=img)["image"]

        # Move the channels axis to the front
        img = np.transpose(img, axes=(-1, 0, 1))
        img = torch.tensor(img)

        sample = {"image": img}

        # Add label to sample if required
        if self.labels is True:
            label = self.df.iloc[idx, 1]
            sample["label"] = torch.tensor(label, dtype=torch.float32)

        return sample


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


def order_test_data(csv_path):
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

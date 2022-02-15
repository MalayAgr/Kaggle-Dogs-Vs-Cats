import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CatsDogsDataset(Dataset):
    def __init__(self, csv, transform=None, resize: tuple[int, int] = None):
        self.df: pd.DataFrame = pd.read_csv(csv)
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image path and label
        img = self.df.iloc[idx, 0]
        label = self.df.iloc[idx, 1]

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

        return {"image": img, "label": label}

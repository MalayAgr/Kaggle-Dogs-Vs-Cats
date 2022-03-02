from __future__ import annotations

import glob
import os
from collections import defaultdict
from typing import Any

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils import data
from tqdm import tqdm

import config
from model import CatsDogsModel, get_model_params


class Engine:
    def __init__(
        self, model: nn.Module, optimizer: Optimizer, scheduler: _LRScheduler = None
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.history = defaultdict(list)

    @staticmethod
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        fn = nn.BCELoss()
        fn.to(config.DEVICE)
        return fn(y_pred, y_true)

    @staticmethod
    def correct_and_accuracy(
        y_pred: torch.Tensor, y_true: torch.Tensor
    ) -> tuple[int, float]:
        correct = (y_true == (y_pred > 0.5).float()).float().sum()
        return correct, correct / len(y_pred)

    def save_loss_curve(self, filename: str, best_iter: int = None) -> None:
        train_loss = self.history["train_loss"]
        val_loss = self.history["val_loss"]

        print(train_loss, val_loss)

        plt.figure()
        plt.plot(train_loss, label="Training")
        plt.plot(val_loss, label="Validation")

        if best_iter is not None:
            plt.axvline(best_iter, color="black", linestyle="--")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        path = os.path.join(config.MODEL_DIR, filename)
        plt.savefig(path)
        plt.clf()

    def save_model(self, filename: str) -> None:
        path = os.path.join(config.MODEL_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def _train_one_step(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int, float]:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        preds = self.model(image=data["image"])

        true = data["label"]
        loss = self.loss_fn(preds, true)

        loss.backward()
        self.optimizer.step()

        correct, accuracy = self.correct_and_accuracy(preds, true)

        return loss, correct, accuracy

    def train(
        self, data_loader: data.DataLoader, epoch_num: int = 0
    ) -> tuple[float, float]:
        self.model.train()
        total_correct = 0

        optimizer = self.optimizer
        scheduler = self.scheduler

        total_loss = accuracy = 0.0
        iters = len(data_loader)

        with tqdm(data_loader, unit="batch", desc="Training") as p_loader:
            for batch_index, data in enumerate(p_loader):
                optimizer.zero_grad(set_to_none=True)

                loss, correct, accuracy = self._train_one_step(data=data)

                total_correct += correct
                loss = loss.item()
                accuracy_per = accuracy * 100

                postfix = {"loss": loss, "batch acc": f"{accuracy_per: .2f}%"}

                postfix["lr"] = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                )

                p_loader.set_postfix(postfix)

                total_loss += loss

                if scheduler is not None:
                    scheduler.step(epoch_num + batch_index / iters)

            avg_loss = total_loss / (batch_index + 1)
            self.history["train_loss"].append(avg_loss)
            return avg_loss, total_correct

    def _evaluate_one_step(
        self, data: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, int, float]:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        preds = self.model(image=data["image"])

        true = data["label"]
        loss = self.loss_fn(preds, true)
        correct, accuracy = self.correct_and_accuracy(preds, true)

        return loss, correct, accuracy

    def evaluate(self, data_loader: data.DataLoader) -> tuple[float, float]:
        self.model.eval()

        total_correct = 0
        total_loss = accuracy = 0.0

        with tqdm(data_loader, unit="batch", desc="Validation") as p_loader:
            for batch_index, data in enumerate(p_loader):
                with torch.no_grad():
                    loss, correct, accuracy = self._evaluate_one_step(data=data)

                total_correct += correct
                loss = loss.item()
                accuracy_per = accuracy * 100

                postfix = {"loss": loss, "batch acc": f"{accuracy_per: .2f}%"}

                p_loader.set_postfix(postfix)

                total_loss += loss

            avg_loss = total_loss / (batch_index + 1)
            self.history["val_loss"].append(avg_loss)
            return avg_loss, total_correct

def train(
    dataset: data.Dataset,
    params: dict[str, Any],
    *,
    val_split: float = 0.2,
    save_model: bool = False,
) -> np.float32:
    length = len(dataset)
    val_len = int(val_split * length)
    t_len = length - val_len

    generator = torch.Generator().manual_seed(42)
    split = data.random_split(dataset, [t_len, val_len], generator=generator)

    batch_size = params["batch_size"]

    train_loader = data.DataLoader(
        split[0],
        shuffle=True,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = data.DataLoader(
        split[1],
        shuffle=True,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    model = CatsDogsModel(n_targets=1, **get_model_params(params))
    model.init_weights()
    model.to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer=optimizer, T_0=config.COSINE_ANNEALING_T0
    )

    engine = Engine(model=model, optimizer=optimizer, scheduler=scheduler)

    esr = params.get("early_stopping_rounds", 5)
    best_loss = np.inf
    best_iter = counter = 0
    losses = []

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")

        t_loss, t_correct = engine.train(data_loader=train_loader, epoch_num=epoch)
        val_loss, val_correct = engine.evaluate(data_loader=val_loader)

        losses.append(val_loss)

        t_acc = t_correct / t_len
        val_acc = val_correct / val_len

        print(f"Avg. train loss={t_loss:.4f}, Train acc={t_acc:.4f}")
        print(f"Avg. val loss={val_loss:.4f}, Val acc={val_acc:.4f}")

        # Simple early stopping
        if val_loss < best_loss:
            best_loss, best_iter = val_loss, epoch
            if save_model is True:
                engine.save_model(f"model-{epoch + 1}.pth")
        else:
            counter += 1

        if counter > esr:
            break

    if save_model is True:
        engine.save_loss_curve(f"loss_curve.png", best_iter)

    return np.mean(losses, dtype=np.float32)


def make_inference(data: data.Dataset, params: dict[str, Any]) -> None:
    models_path = os.path.join(config.MODEL_DIR, "*.pth")

    for model_path in glob.glob(models_path):
        model = CatsDogsModel(**get_model_params(params), n_targets=1)

        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        model.to(config.DEVICE)
        model.eval()

        data_loader = data.DataLoader(
            data,
            batch_size=params["batch_size"],
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
        )

        df_dict = defaultdict(list)
        df_dict["id"] = range(1, len(data) + 1)

        for batch in data_loader:
            for k, v in batch.items():
                batch[k] = v.to(config.DEVICE)

            with torch.no_grad():
                preds = model(image=batch["image"])

            preds = (preds > 0.5).int()
            df_dict["label"].extend(preds.tolist())

        basename = os.path.basename(model_path)
        filename, _ = os.path.splitext(basename)

        df = pd.DataFrame.from_dict(df_dict)
        path = os.path.join(config.INFERENCE_DIR, f"{filename}_submission.csv")
        df.to_csv(path, index=False)

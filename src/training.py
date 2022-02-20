from __future__ import annotations

import os

import torch
from sklearn import metrics
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

import config


class Engine:
    def __init__(self, model: nn.Module, optimizer: Optimizer, scheduler=None) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    @staticmethod
    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        fn = nn.BCELoss()
        return fn(y_pred, y_true)

    @staticmethod
    def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        if y_true.is_cuda is True and y_pred.is_cuda is True:
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()

        y_true = y_true.detach().numpy()
        y_pred = y_pred.detach().numpy() > 0.5

        return metrics.accuracy_score(y_true, y_pred)

    def reset_model_weights(self):
        for layer in self.model.children():
            try:
                layer.reset_parameters()
            except AttributeError:
                pass

    def save_model(self, filename: str) -> None:
        path = os.path.join(config.MODEL_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def _train_one_step(self, data) -> tuple[torch.Tensor, float]:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        preds = self.model(image=data["image"])

        true = data["label"]
        loss = self.loss_fn(preds, true)

        loss.backward()
        self.optimizer.step()

        accuracy = self.accuracy(preds, true)

        return loss, accuracy

    def train(self, data_loader, epoch_num=0) -> tuple[float, float]:
        self.model.train()

        optimizer = self.optimizer
        scheduler = self.scheduler

        total_loss = accuracy = 0.0
        iters = len(data_loader)

        with tqdm(data_loader, unit="batch", desc="Training") as p_loader:
            for batch_index, data in enumerate(p_loader):
                optimizer.zero_grad()

                loss, accuracy = self._train_one_step(data=data)

                loss = loss.item()
                accuracy_per = accuracy * 100

                postfix = {"loss": loss, "accuracy": f"{accuracy_per: .2f}%"}

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

            return avg_loss, accuracy

    def _evaluate_one_step(self, data) -> tuple[torch.Tensor, float]:
        for key, value in data.items():
            data[key] = value.to(config.DEVICE)

        preds = self.model(image=data["image"])

        true = data["label"]
        loss = self.loss_fn(preds, true)
        accuracy = self.accuracy(preds, true)

        return loss, accuracy

    def evaluate(self, data_loader) -> tuple[float, float]:
        self.model.eval()

        total_loss = accuracy = 0.0

        with tqdm(data_loader, unit="batch", desc="Validation") as p_loader:
            for batch_index, data in enumerate(p_loader):
                with torch.no_grad():
                    loss, accuracy = self._evaluate_one_step(data=data)

                loss = loss.item()
                accuracy_per = accuracy * 100

                postfix = {"loss": loss, "accuracy": f"{accuracy_per: .2f}%"}

                p_loader.set_postfix(postfix)

                total_loss += loss

            avg_loss = total_loss / (batch_index + 1)

            return avg_loss, accuracy

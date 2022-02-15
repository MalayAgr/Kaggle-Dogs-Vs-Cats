from __future__ import annotations

import torch
from sklearn import metrics
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

import config


def train_one_step(
    model: nn.Module,
    data,
    optimizer: Optimizer,
    loss_fn,
) -> torch.Tensor:
    optimizer.zero_grad()

    for key, value in data.items():
        data[key] = value.to(config.DEVICE)

    true = data["label"]

    preds = model(image=data["image"])
    loss = loss_fn(preds, true)

    loss.backward()
    optimizer.step()

    accuracy = metrics.accuracy_score(
        true.detach().numpy(), preds.detach().numpy() > 0.5
    )

    return loss, accuracy


def train_one_epoch(
    model: nn.Module,
    data_loader,
    optimizer: Optimizer,
    loss_fn,
    epoch_num: int,
    scheduler=None,
):
    total_loss = 0.0
    history = []

    iters = len(data_loader)

    with tqdm(data_loader, unit="batch", desc=f"Epoch {epoch_num + 1}") as p_loader:
        for batch_index, data in enumerate(p_loader):
            loss, accuracy = train_one_step(
                model=model, data=data, optimizer=optimizer, loss_fn=loss_fn
            )

            postfix = {"loss": loss.item(), "accuracy": f"{accuracy * 100: .2f}%"}

            postfix["lr"] = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )

            p_loader.set_postfix(postfix)

            history.append(loss.item())

            total_loss += loss.item()

            if scheduler is not None:
                scheduler.step(epoch_num + batch_index / iters)

        avg_loss = total_loss / (batch_index + 1)

        p_loader.write(
            f"Avg. training loss={avg_loss: .3f}; Training accuracy={accuracy * 100: .2f}%"
        )

        return avg_loss, history


def validate_one_step(model: nn.Module, data, loss_fn) -> torch.Tensor:
    for key, value in data.items():
        data[key] = value.to(config.DEVICE)

    true = data["label"]

    preds = model(image=data["image"])
    loss = loss_fn(preds, true)

    accuracy = metrics.accuracy_score(
        true.detach().numpy(), preds.detach().numpy() > 0.5
    )

    return loss, accuracy


def validate_one_epoch(model: nn.Module, data_loader, loss_fn, epoch_num: int):
    total_loss = 0.0
    history = []

    with tqdm(data_loader, unit="batch", desc=f"Epoch {epoch_num + 1}") as p_loader:
        for batch_index, data in enumerate(p_loader):

            with torch.no_grad():
                loss, accuracy = validate_one_step(model, data, loss_fn=loss_fn)

            p_loader.set_postfix(loss=loss.item(), accuracy=f"{accuracy * 100: .2f}%")

            history.append(loss.item())

            total_loss += loss.item()

        avg_loss = total_loss / (batch_index + 1)

        p_loader.write(
            f"Avg. validation loss={avg_loss: .3f}; Accuracy={accuracy * 100: .2f}%"
        )

        return total_loss, history

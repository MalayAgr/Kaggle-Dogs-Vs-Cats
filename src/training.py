from __future__ import annotations

import torch
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

    preds = model(image=data["image"])
    loss = loss_fn(preds, data["label"])

    loss.backward()
    optimizer.step()

    return loss


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
            loss = train_one_step(
                model=model, data=data, optimizer=optimizer, loss_fn=loss_fn
            )

            postfix = {"loss": loss.item()}

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

        p_loader.write(f"Avg. training loss at the end of epoch {avg_loss: .3f}")

        return avg_loss, history


def validate_one_step(model: nn.Module, data, loss_fn) -> torch.Tensor:
    for key, value in data.items():
        data[key] = value.to(config.DEVICE)

    preds = model(image=data["image"])
    loss = loss_fn(preds, data["label"])
    return loss


def validate_one_epoch(model: nn.Module, data_loader, loss_fn, epoch_num: int):
    total_loss = 0.0
    history = []

    with tqdm(data_loader, unit="batch", desc=f"Epoch {epoch_num + 1}") as p_loader:
        for batch_index, data in enumerate(p_loader):

            with torch.no_grad():
                loss = validate_one_step(model, data, loss_fn=loss_fn)

            p_loader.set_postfix({"loss": loss.item()})

            history.append(loss.item())

            total_loss += loss.item()

        avg_loss = total_loss / (batch_index + 1)

        p_loader.write(f"Avg. validation loss at the end of epoch {avg_loss: .3f}")

        return total_loss, history

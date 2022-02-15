import torch
from torch import nn
from torch.optim import Optimizer

import config


def train_one_step(model: nn.Module, data, optimizer: Optimizer, loss_fn):
    optimizer.zero_grad()
    preds = model(image=data["image"])
    loss = loss_fn(preds, data["label"])
    loss.backward()
    optimizer.step()
    return loss


def train_one_epoch(
    model: nn.Module, data_loader, optimizer: Optimizer, loss_fn, scheduler
):
    model.train()

    total_loss = 0
    for batch_index, data in enumerate(data_loader):
        loss = train_one_step(model, data, optimizer, loss_fn=loss_fn)
        total_loss += loss

        if batch_index % 100 == 99:
            print(f"Loss after mini-batch {batch_index + 1}: {loss: .3f}")

    return total_loss / (batch_index + 1)


def validate_one_step(model: nn.Module, data, loss_fn):
    preds = model(image=data["image"])
    loss = loss_fn(preds, data["label"])
    return loss


def validate_one_epoch(model: nn.Module, data_loader, loss_fn):
    model.eval()

    total_loss = 0
    for batch_index, data in enumerate(data_loader):

        with torch.no_grad():
            loss = validate_one_step(model, data, loss_fn=loss_fn)

        total_loss += loss

        if batch_index % 100 == 99:
            print(f"Loss after mini-batch {batch_index + 1}: {loss: .3f}")

    return total_loss / (batch_index + 1)

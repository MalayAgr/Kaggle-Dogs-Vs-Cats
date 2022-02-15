from torch import nn
from torch.optim import Optimizer

import config


def train_one_step(model: nn.Module, data, optimizer: Optimizer):
    optimizer.zero_grad()
    loss = model(**data)
    loss.backward()
    optimizer.step()
    return loss


def train_one_epoch(model: nn.Module, data_loader, optimizer: Optimizer, scheduler):
    model.train()

    total_loss = 0
    for batch_index, data in enumerate(data_loader):
        loss = train_one_step(model, data, optimizer)
        total_loss += loss

        if batch_index % 10 == 0:
            print()

    return total_loss / (batch_index + 1)

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm


def train_one_step(
    model: nn.Module,
    data,
    optimizer: Optimizer,
    loss_fn,
) -> torch.Tensor:
    optimizer.zero_grad()

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
    total_loss = torch.tensor(0, dtype=torch.float32)
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

            total_loss += loss

            if scheduler is not None:
                scheduler.step(epoch_num + batch_index / iters)

        total_loss = (total_loss / (batch_index + 1)).item()

        p_loader.write(f"Avg. training loss at the end of epoch {total_loss: .3f}")

        return total_loss, history


def validate_one_step(model: nn.Module, data, loss_fn) -> torch.Tensor:
    preds = model(image=data["image"])
    loss = loss_fn(preds, data["label"])
    return loss


def validate_one_epoch(model: nn.Module, data_loader, loss_fn, epoch_num: int):
    model.eval()

    total_loss = torch.tensor(0, dtype=torch.float32)
    history = []

    with tqdm(data_loader, unit="batch", desc=f"Epoch {epoch_num + 1}") as p_loader:
        for batch_index, data in enumerate(p_loader):

            with torch.no_grad():
                loss = validate_one_step(model, data, loss_fn=loss_fn)

            p_loader.set_postfix({"loss": loss.item()})

            history.append(loss.item())

            total_loss += loss

        total_loss = (total_loss / (batch_index + 1)).item()

        p_loader.write(f"Avg. validation loss at the end of epoch {total_loss: .3f}")

        return total_loss, history

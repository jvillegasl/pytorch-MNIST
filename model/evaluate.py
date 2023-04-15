import numpy as np
import torch
from torch import nn, Tensor
from torch.types import Number
from torch.utils.data.dataloader import DataLoader
from typing import overload, Callable, Optional, Literal, Tuple


from model.loss import loss_batch

__all__ = ['evaluate']


@overload
def evaluate(
        model: nn.Module,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        valid_dl: DataLoader[Tuple[Tensor, Tensor]],
        metric: Literal[None]
) -> Tuple[float, float, None]:
    ...


@overload
def evaluate(
        model: nn.Module,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        valid_dl: DataLoader[Tuple[Tensor, Tensor]],
        metric: Callable[[Tensor, Tensor], float]
) -> Tuple[float, float, float]:
    ...


def evaluate(
        model: nn.Module,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        valid_dl: DataLoader[Tuple[Tensor, Tensor]],
        metric: Optional[Callable[[Tensor, Tensor], float]] = None
):
    avg_loss: float
    total: float
    avg_metric: Optional[float] = None

    with torch.no_grad():
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb, yb in valid_dl]

        losses = [result[0] for result in results]
        nums = [result[1] for result in results]
        metrics = [result[2] for result in results if result[2] is not None]

        total = np.sum(nums)

        avg_loss = np.sum(np.multiply(losses, nums)) / total

        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total

    return avg_loss, total, avg_metric

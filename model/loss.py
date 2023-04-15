from typing import overload, Callable, Optional, Literal, Tuple
from torch import nn, Tensor
from torch.types import Number
from torch.optim import Optimizer

__all__ = ['loss_batch']


@overload
def loss_batch(
        model: nn.Module,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        xb: Tensor,
        yb: Tensor,
        opt: Optional[Optimizer] = None,
        metric: Literal[None] = ...
) -> Tuple[Number, int, None]:
    ...


@overload
def loss_batch(
        model: nn.Module,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        xb: Tensor,
        yb: Tensor,
        opt: Optional[Optimizer] = None,
        metric: Callable[[Tensor, Tensor], float] = ...
) -> Tuple[Number, int, float]:
    ...


def loss_batch(
        model: nn.Module,
        loss_func: Callable[[Tensor, Tensor], Tensor],
        xb: Tensor,
        yb: Tensor,
        opt: Optional[Optimizer] = None,
        metric: Optional[Callable[[Tensor, Tensor], float]] = None
):
    preds: Tensor
    preds = model(xb)

    loss = loss_func(preds, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metric_result = None

    if metric is not None:
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result

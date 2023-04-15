import torch
from torch import Tensor

__all__ = ['accuracy']

def accuracy(outputs: Tensor, labels: Tensor) -> float:
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)
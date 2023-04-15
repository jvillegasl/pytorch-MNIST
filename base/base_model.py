import torch.nn as nn
from abc import abstractmethod

__all__ = ['BaseModel']

class BaseModel(nn.Module):

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_latest(self):
        raise NotImplementedError
        
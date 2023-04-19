import torch.nn.functional as F
from torch.optim import Adam

from model import accuracy

HYPER = {
    'batch_size': 100,
    'num_epochs': 16,
    'loss_fn': F.cross_entropy,
    'lr': 0.001,
    'opt_fn': Adam,
    'metric': accuracy
}

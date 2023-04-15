import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from base import BaseModel

from data_loader import MNISTDataLoader
from model import accuracy
from trainer import Trainer

def train(model: BaseModel):
    batch_size = 100
    num_epochs = 10
    loss_fn = F.cross_entropy
    lr = 0.005
    opt_fn = Adam
    metric = accuracy

    mnist = MNISTDataLoader(batch_size=100)
    train_dl = mnist.train_dl
    val_dl = mnist.val_dl

    trainer = Trainer(model, num_epochs, loss_fn, train_dl, val_dl, lr, opt_fn, metric)
    history = trainer.fit()
    model.save()
    
    pass
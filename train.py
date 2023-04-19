from base import BaseModel
from config import HYPER
from data_loader import MNISTDataLoader
from trainer import Trainer


def train(model: BaseModel, save_model: bool = True):
    batch_size = HYPER['batch_size']
    num_epochs = HYPER['num_epochs']
    loss_fn = HYPER['loss_fn']
    lr = HYPER['lr']
    opt_fn = HYPER['opt_fn']
    metric = HYPER['metric']

    mnist = MNISTDataLoader(batch_size)
    train_dl = mnist.train_dl
    val_dl = mnist.val_dl

    trainer = Trainer(model, num_epochs, loss_fn,
                      train_dl, val_dl, lr, opt_fn, metric)
    history = trainer.fit()

    if save_model:
        model.save()
    
    return history

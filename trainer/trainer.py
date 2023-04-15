import time
from torch import nn, Tensor
import torch
from torch.types import Number
from torch.optim import Adam, SGD
from torch.utils.data.dataloader import DataLoader
from typing import Callable, Optional, List, Tuple, Type
from model import evaluate

from model.loss import loss_batch
from utils import progress_bar

__all__ = ['Trainer']


class Trainer():

    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        train_dl: DataLoader[Tuple[Tensor, Tensor]],
        valid_dl: DataLoader[Tuple[Tensor, Tensor]],
        lr: float,
        opt_fn: Optional[Type[Adam | SGD]] = None,
        metric: Optional[Callable[[Tensor, Tensor], float]] = None
    ):
        self.model = model
        self.epoch = 0
        self.epochs = epochs
        self.loss_fn = loss_fn
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.metric = metric

        if opt_fn is None:
            opt_fn = SGD
        opt = opt_fn(model.parameters(), lr=lr)

        self.opt = opt

    def fit(self):
        print("==============================Starting training==============================", end='\n\n')

        train_losses, val_losses, val_metrics = self.__train_epochs()

        print("\n==============================Training finished==============================")
        return train_losses, val_losses, val_metrics

    def __logged_train(self, training_starting_time: float):
        epoch_starting_time = time.time()
        print(
            'Epoch [ {} / {} ] [{}] - {:.2f}%'
            .format(self.epoch+1,
                    self.epochs,
                    progress_bar(self.epoch, self.epochs),
                    100 * self.epoch/self.epochs),
            end='\r'
        )

        results = self.__train()

        train_loss, val_loss, val_metric = results

        epoch_finishing_time = time.time()
        epoch_duration = epoch_finishing_time - epoch_starting_time
        finished_at = epoch_finishing_time - training_starting_time

        print('', end='\x1b[1K\r')

        if self.metric is None:
            print(
                'Epoch[ {} / {} ], train_loss: {:.4f}, val_loss: {:.4f}, duration: {:.4f}, finished_at: {:.4f}'
                .format(self.epoch+1,
                        self.epochs,
                        train_loss,
                        val_loss,
                        epoch_duration,
                        finished_at)
            )
        else:
            print(
                'Epoch [ {} / {} ], train_loss: {:.4f}, val_loss: {:.4f}, val_{}: {:.4f}, duration: {:.4f}, finished_at: {:.4f}'
                .format(self.epoch+1,
                        self.epochs,
                        train_loss,
                        val_loss,
                        self.metric.__name__,
                        val_metric,
                        epoch_duration,
                        finished_at)
            )

        return results

    def __logged_batch(self, index: int, total: int):
        print('', end='\x1b[1K\r')

        print(
            'Epoch [ {} / {} ] [{}] - {:.2f}%'
            .format(self.epoch+1,
                    self.epochs,
                    progress_bar((self.epoch + index/total), self.epochs),
                    100 * (self.epoch + index/total)/self.epochs, 1),
            end='\r'
        )

    def __train(self):
        self.model.train()

        train_loss = -1

        for i, data in enumerate(self.train_dl):
            xb: Tensor
            yb: Tensor

            self.__logged_batch(i, len(self.train_dl))
            
            xb, yb = data
            train_loss, _, _ = loss_batch(
                self.model, self.loss_fn, xb, yb, self.opt)

        self.model.eval()

        results = evaluate(self.model, self.loss_fn,
                           self.valid_dl, self.metric)

        val_loss, _, val_metric = results

        return train_loss, val_loss, val_metric

    def __train_epochs(self):
        train_losses: List[Number] = []
        val_losses: List[float] = []
        val_metrics: List[float] = []

        training_starting_time = time.time()
        for epoch in range(self.epochs):
            self.epoch = epoch
            results = self.__logged_train(training_starting_time)

            train_loss, val_loss, val_metric = results

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_metric is not None:
                val_metrics.append(val_metric)

        return train_losses, val_losses, val_metrics

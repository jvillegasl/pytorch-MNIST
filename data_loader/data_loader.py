from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from definitions import DATA_PATH
from utils import get_default_device, to_device, split_indices
from base import DeviceDataLoader

from torch import device as Device
from torch.utils.data.dataset import Dataset
from typing import List, Tuple
from torch import Tensor


__all__ = ['MNISTDataLoader']

class MNISTDataLoader():
    train_dl: DeviceDataLoader
    val_dl: DeviceDataLoader
    classes: List[str]
    device: Device
    batch_size: int
    validation_split: float

    def __init__(self, batch_size: int, validation_split: float = 0.2,):
        self.dataset = MNIST(
            root=DATA_PATH, download=True, transform=ToTensor(), train=True)
        
        self.test_dataset = MNIST(
            root=DATA_PATH, download=True, transform=ToTensor(), train=False)

        self.batch_size = batch_size
        self.device = get_default_device()
        self.validation_split = validation_split

        self.train_dl, self.val_dl, self.test_dl = self.__split_sampler(self.dataset)

    def __split_sampler(self, dataset: MNIST):
        train_indices, val_indices = split_indices(
            len(dataset),
            self.validation_split
        )

        train_sampler = SubsetRandomSampler(train_indices)
        train_dl = DataLoader(dataset, self.batch_size, sampler=train_sampler)

        val_sampler = SubsetRandomSampler(val_indices)
        val_dl = DataLoader(dataset, self.batch_size, sampler=val_sampler)

        test_dl = DataLoader(dataset, self.batch_size, shuffle=True)

        train_dl = DeviceDataLoader(train_dl, self.device)
        val_dl = DeviceDataLoader(val_dl, self.device)
        test_dl = DeviceDataLoader(test_dl, self.device)

        return train_dl, val_dl, test_dl

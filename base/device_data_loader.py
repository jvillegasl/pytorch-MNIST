from torch.utils.data.dataloader import DataLoader
from torch import device as Device

from utils import to_device

__all__ = ['DeviceDataLoader']

class DeviceDataLoader(DataLoader):

    dl: DataLoader
    device: Device

    def __init__(self, dl: DataLoader, device: Device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    
    def __len__(self):
        return len(self.dl)
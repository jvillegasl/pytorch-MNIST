import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

from torch import Tensor
from torch.utils.data.dataloader import DataLoader

__all__ = ['show_batch']

def show_batch(dl: DataLoader):
    
    for images, labels in dl:
        images: Tensor
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(make_grid(images, 10).permute(1, 2, 0))
        plt.show()
        break

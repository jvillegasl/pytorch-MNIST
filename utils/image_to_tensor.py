import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
import numpy as np
import torch
from torchvision.transforms.functional import pil_to_tensor, rgb_to_grayscale, invert

__all__ = ['image_to_tensor']


def image_to_tensor(image_path: str, invert_colors: bool = False):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((28, 28))

    tensor = pil_to_tensor(image)
    tensor = rgb_to_grayscale(tensor)

    if invert_colors:
        tensor = invert(tensor)

    tensor = tensor.to(dtype=torch.float32)
    tensor = tensor / 255.0

    return tensor

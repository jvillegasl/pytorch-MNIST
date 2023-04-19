import numpy as np
import torch
from onnx.utils import extract_model
import onnx
from torch.jit._serialization import load
from data_loader.data_loader import MNISTDataLoader
from model import MNISTModel
from test import test, test_onnx
import onnxruntime
from PIL import Image

from utils import image_to_tensor


def main():
    # model = MNISTModel()
    # model.load_latest()
    # model.to('cpu')
    # model.eval()
    # model.export('mnist02')
    # mnist = MNISTDataLoader(batch_size=100)
    # dataset = mnist.dataset
    # print(dataset[0])
    preds = test_onnx('./exports/mnist02.onnx', './assets/images/four.jpg')
    print(preds)
    print(np.argmax(preds))





if __name__ == '__main__':
    main()
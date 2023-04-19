import torch
import torch.nn as nn
from torch import Tensor
import onnxruntime
import matplotlib.pyplot as plt
from typing import List

from config import HYPER
from data_loader import MNISTDataLoader
from model import evaluate
from utils import image_to_tensor


def test(model: nn.Module):
    batch_size = HYPER['batch_size']
    loss_fn = HYPER['loss_fn']
    metric = HYPER['metric']

    mnist = MNISTDataLoader(batch_size)
    test_dl = mnist.test_dl

    test_loss, _, test_acc = evaluate(model, loss_fn, test_dl, metric)

    print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))


def predict_image(model: nn.Module, image: Tensor, label_index: int, classes: List[str]):
    xb = image.unsqueeze(0)

    yb: Tensor = model(xb)
    print(yb)

    preds: Tensor
    _, preds = torch.max(yb, dim=1)

    pred_index = int(preds[0].item())

    print('Label: {}, Predicted: {}'.format(
        classes[label_index], classes[pred_index]))
    plt.imshow(image.permute(1, 2, 0).cpu())
    plt.title('Label: {}, Predicted: {}'.format(
        classes[label_index], classes[pred_index]))
    plt.show()

    return classes[pred_index]


def predict_random_image(model: nn.Module):
    mnist = MNISTDataLoader(100)
    classes = mnist.dataset.classes

    test_dl = mnist.test_dl

    for xb, yb in test_dl:
        xb: Tensor
        yb: Tensor

        image = xb[0]
        label_index = int(yb[0].item())
        print(image)
        predict_image(model, image, label_index, classes)
        break


def test_onnx(model_path: str, image_path: str):
    session = onnxruntime.InferenceSession(model_path)

    def to_numpy(tensor: Tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    img = image_to_tensor(image_path, invert_colors=True)
    inputs = {session.get_inputs()[0].name: to_numpy(img.unsqueeze(0))}
    outputs = session.run(None, inputs)
    y = list(outputs[0])

    return y
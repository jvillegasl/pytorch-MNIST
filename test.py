import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from typing import List, Type
import matplotlib.pyplot as plt

from base import BaseModel
from data_loader import MNISTDataLoader
from model import accuracy, evaluate,MNISTModel


def test(model: BaseModel):
    batch_size = 100
    num_epochs = 10
    loss_fn = F.cross_entropy
    lr = 0.005
    opt_fn = Adam
    metric = accuracy

    mnist = MNISTDataLoader(batch_size)
    test_dl = mnist.test_dl

    test_loss, _, test_acc = evaluate(model, loss_fn, test_dl, metric)

    print('Loss: {:.4f}, Accuracy: {:.4f}'.format(test_loss, test_acc))

def predict_image(model: BaseModel, image: Tensor, label_index: int, classes: List[str]):
    xb = image.unsqueeze(0)

    yb: Tensor = model(xb)

    preds: Tensor
    _, preds = torch.max(yb, dim=1)

    pred_index = int(preds[0].item())

    print('Label: {}, Predicted: {}'.format(classes[label_index], classes[pred_index]))
    plt.imshow(image.permute(1,2,0).cpu())
    plt.title('Label: {}, Predicted: {}'.format(classes[label_index], classes[pred_index]))
    plt.show()

    return classes[pred_index]

def predict_random_image(model: BaseModel):
    mnist = MNISTDataLoader(100)
    classes = mnist.dataset.classes

    test_dl = mnist.test_dl

    for xb, yb in test_dl:
        xb: Tensor
        yb: Tensor

        image = xb[0]
        label_index = int(yb[0].item())

        predict_image(model, image, label_index, classes)
        break

if __name__ == '__main__':
    model = MNISTModel()
    
    predict_random_image(model)
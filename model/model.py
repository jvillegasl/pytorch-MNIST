import os
import time
from torch import nn, Tensor
from torch.jit._script import script, ScriptModule
import torch
import torch.nn.functional as F
import torchsummary
from base import BaseModel

from definitions import DATA_PATH, EXPORTS_PATH
from utils import get_default_device


class MNISTModel(BaseModel):
    def __init__(self):
        super().__init__()

        # input: bs x 1 x 28 x 28

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        # output: bs x 16 x 14 x 14

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(2, 2)
        # output: bs x 16 x 7 x 7

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool2d(2, 2)
        # output: bs x 16 x 4 x 4

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.max_pool4 = nn.MaxPool2d(2, 2)
        # output: bs x 16 x 2 x 2

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.max_pool5 = nn.MaxPool2d(2, 2)
        # output: bs x 16 x 1 x 1

        self.dropout1 = nn.Dropout2d(0.25)
        # output: bs x 16 x 1 x 1

        self.flatten1 = nn.Flatten()  # output: bs x 16
        self.linear1 = nn.Linear(16, 10)  # output: bs x 64

        self.to(get_default_device())

    def forward(self, x: Tensor):
        out: Tensor = self.conv1(x)
        out = self.relu1(out)
        out = self.max_pool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.max_pool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.max_pool3(out)

        out = self.conv4(out)
        out = self.relu4(out)
        out = self.max_pool4(out)

        out = self.conv5(out)
        out = self.relu5(out)
        out = self.max_pool5(out)

        out = self.dropout1(out)

        out = self.flatten1(out)
        out = self.linear1(out)

        out = F.softmax(out, dim=1)

        return out

    def summary(self):
        torchsummary.summary(self, (1, 28, 28), device='cuda')

    def save(self):
        file_name = time.strftime("%Y_%m_%dT%H_%M_%S") + ".pth"
        folder_path = os.path.join(DATA_PATH, 'models', 'MNIST')

        torch.save(self.state_dict(), os.path.join(folder_path, file_name))

    def load_latest(self):
        folder_path = os.path.join(DATA_PATH, 'models', 'MNIST')

        files_list = os.listdir(folder_path)

        weights_files = [os.path.join(folder_path, file)
                         for file in files_list if file.endswith('.pth')]
        file_path = max(weights_files, key=os.path.getctime)

        self.load_state_dict(torch.load(file_path))
        print('Model loaded from: {}'.format(file_path))

    def export(self, file_name: str, as_onnx: bool = False):

        if as_onnx:
            file_path = os.path.join(EXPORTS_PATH, file_name + '.onnx')

            self.to('cpu')
            self.eval()
            x = torch.randn(1, 1, 28, 28, requires_grad=True)
            y = self(x)

            torch.onnx.export(
                self,
                x,
                file_path,
                export_params=True,
                opset_version=9,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                              'output': {0: 'batch_size'}}
            )

            self.to(get_default_device())
            return

        file_path = os.path.join(EXPORTS_PATH, file_name + '.pt')
        model_scripted = script(self)
        model_scripted.to('cpu')
        model_scripted.save(file_path)

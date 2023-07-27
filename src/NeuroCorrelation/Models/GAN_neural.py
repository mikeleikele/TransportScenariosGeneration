from torch import Tensor, zeros
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot
from pathlib import Path
import os
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch
from torch.functional import F
from torchmetrics.functional.regression import kendall_rank_corrcoef
from torchmetrics import SpearmanCorrCoef


class GAN_Linear_neural_16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_LinearDiscriminator_16
        self.G = GAN_LinearGenerator_16

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

class GAN_LinearDiscriminator_16(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=16, out_features=12)
        self.hidden_layer_2 = nn.Linear(in_features=12, out_features=8)
        self.hidden_layer_3 = nn.Linear(in_features=8, out_features=4)
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        
        #self.activation_function_1 = nn.LeakyReLU(0.2, inplace=True)
        #self.activation_function_2 = nn.LeakyReLU(0.2, inplace=True)
        #self.activation_function_3 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_16(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=12, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=16)
        self.hidden_layer_3 = nn.Linear(in_features=16, out_features=32)
        self.hidden_layer_4 = nn.Linear(in_features=32, out_features=48)
        self.hidden_layer_5 = nn.Linear(in_features=48, out_features=64)
        self.hidden_layer_6 = nn.Linear(in_features=64, out_features=48)
        self.hidden_layer_7 = nn.Linear(in_features=48, out_features=16)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_7(layer_nn)
        return {"x_input":x, "x_output":x_out}


class GAN_LinearNeural_7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_LinearDiscriminator_7
        self.G = GAN_LinearGenerator_7

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D
    
class GAN_LinearDiscriminator_7(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=7, out_features=10)
        self.hidden_layer_2 = nn.Linear(in_features=10, out_features=6)
        self.hidden_layer_3 = nn.Linear(in_features=6, out_features=4)
        self.hidden_layer_4 = nn.Linear(in_features=4, out_features=1)
        

    def forward(self, x):
        x_flat = x.view(-1)
        layer_nn = self.hidden_layer_1(x_flat)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_LinearGenerator_7(nn.Module):
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=4, out_features=12)
        self.hidden_layer_2 = nn.Linear(in_features=12, out_features=18)
        self.hidden_layer_3 = nn.Linear(in_features=18, out_features=24)
        self.hidden_layer_4 = nn.Linear(in_features=24, out_features=7)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_4(layer_nn)
        
        return {"x_input":x, "x_output":x_out}


class GAN_neural_mixed_7(nn.Module):
    def __init__(self, generator=None, discriminator=None):
        super().__init__()
        if generator is None:
            self.G = GAN_LinearGenerator_7
        else:
            self.G = generator
        
        if discriminator is None:
            self.D = GAN_LinearDiscriminator_7
        else:
            self.D = discriminator

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D


class GAN_Conv_neural_16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_ConvDiscriminator_16
        self.G = GAN_ConvGenerator_16

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

class GAN_ConvDiscriminator_16(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_2 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_3 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_4 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=2, stride=4, padding=0)
        self.hidden_layer_5 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_ConvGenerator_16(nn.Module):
    def __init__(self):       
        # in  1, 1, 2, 8
        # out 1, 1, 2, 64
        super().__init__()
        self.hidden_layer_1 = nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_2 = nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_3 = nn.ConvTranspose2d(in_channels=5, out_channels=1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_3(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}


class GAN_Conv_neural_7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.D = GAN_ConvDiscriminator_7
        self.G = GAN_ConvGenerator_7

    def get_generator(self):
        return self.G

    def get_discriminator(self):
        return self.D

class GAN_ConvDiscriminator_7(nn.Module):
    # in  1, 1, 7, 32
    # out 1, 1, 1, 1
    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=2, stride=3, padding=2)
        self.hidden_layer_2 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1, padding=2)
        self.hidden_layer_3 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=3, padding=2)
        self.hidden_layer_4 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, stride=2, padding=1)
        self.hidden_layer_5 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        x_hat = F.sigmoid(layer_nn)
        return {"x_input":x, "x_output":x_hat}

class GAN_ConvGenerator_7(nn.Module):
    # in  1, 1, 2, 6
    # out 1, 1, 7, 32
    def __init__(self):       
        super().__init__()
        self.hidden_layer_1 = nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=2, stride=2, padding=0)
        self.hidden_layer_2 = nn.ConvTranspose2d(in_channels=5, out_channels=3, kernel_size=2, stride=2, padding=(0,3))
        self.hidden_layer_3 = nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=2, stride=(1,2), padding=(1,2))

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_3(layer_nn)
        return {"x_input":x, "x_output":x_out}
from torch import Tensor, zeros
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch.nn.functional as F

class NeuralModels():

    def __init__(self, model_case):
        self.model_case = model_case
    
    def get_model(self):
        if self.model_case=="fullyRectangle":
            return self.fullyRectangle()
        else:
            return None


    def fullyRectangle(self):
        model = GEN_fl()
        return model

class GEN_fl(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=78, out_features=78)
        self.encoder_output_layer = nn.Linear(in_features=78, out_features=78)
        self.decoder_hidden_layer = nn.Linear(in_features=78, out_features=78)
        self.decoder_output_layer = nn.Linear(in_features=78, out_features=78)

    def forward(self, x):
        activation = self.encoder_hidden_layer(x)
        activation = F.tanh(activation)
        activation = self.encoder_output_layer(activation)
        activation = F.tanh(activation)
        activation = self.decoder_hidden_layer(activation)
        activation = F.tanh(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = F.tanh(activation)
        return reconstructed
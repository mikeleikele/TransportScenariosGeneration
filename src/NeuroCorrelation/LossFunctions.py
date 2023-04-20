from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss


class LossFunction(nn.Module):
    def __init__(self, input, target):
        self.input = input
        self.target = target
    
    def loss_computate(self,verbose=False):      
        mse_loss = nn.MSELoss()
        loss_out = mse_loss(self.input, self.target)
        return loss_out
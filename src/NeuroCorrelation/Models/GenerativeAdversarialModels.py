import torchinfo
import torch
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
import torch_geometric.nn as gm
from pathlib import Path
import os
import json
from .BaseModel import BaseModel

class GenerativeAdversarialModels(nn.Module,BaseModel):

    def __init__(self, device, layers_list=None, load_from_file =False, json_filepath= None, edge_index=None, **kwargs):
        super().__init__()
        self.model_type="GAN"

        self.models = dict()
        self.device = device
        self.edge_index = edge_index
        self.permutation_forward = dict()
        if load_from_file:
            self.layers_list = self.load_fileJson(json_filepath, self.model_type)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        self.models_layers['generator'], self.models_size['generator'], self.permutation_forward['generator'] = self.list_to_model(self.layers_list['generator_layers'])
        self.models_layers['discriminator'], self.models_size['discriminator'], self.permutation_forward['discriminator'] = self.list_to_model(self.layers_list['discriminator_layers'])
        self.deploy_nnModel()
    
    def get_size(self, ):
        return self.models_size
    
    def deploy_nnModel(self):
        
        if self.edge_index is not None:
            self.models['discriminator'] = nn_Model(layers= self.models_layers['discriminator'], permutation_forward = self.permutation_forward['discriminator'], edge_index= self.edge_index)
            self.models['generator']     = nn_Model(layers= self.models_layers['generator'],     permutation_forward = self.permutation_forward['generator'], edge_index= self.edge_index)
        else:
            self.models['discriminator'] = nn_Model(layers= self.models_layers['discriminator'])
            self.models['generator']     = nn_Model(layers= self.models_layers['generator'])
        
        self.add_module('discriminator', self.models['discriminator'])
        self.add_module('generator', self.models['generator'])
    
    def set_partialModel(self, key, model_net, model_size, model_permutation_forward):
        self.models[key] = model_net
        self.models_size[key] = model_size
        self.permutation_forward[key] = model_permutation_forward
        self.add_module(key, self.models[key])
        
    def get_generator(self, size=False):
        if size:
            return self.models['generator'], self.models_size['generator']
        else:
            return self.models['generator']
        
    def get_discriminator(self, size=False):
        if size:
            return self.models['discriminator'], self.models_size['discriminator']
        else:
            return self.models['discriminator']

    def get_generator(self, size=False):
        if size:
            return self.models['generator'], self.models_size['generator']
        else:
            return self.models['generator']
        
    
    def summary(self):
        summary = dict()
        summary['discriminator'] = torchinfo.summary(self.models['discriminator'], input_size=(1, self.models_size['discriminator']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary['generator'] = torchinfo.summary(self.models['generator'], input_size=(1, self.models_size['generator']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
    
    def forward(self, x):
        raise Exception("forward not implemented.")

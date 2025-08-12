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


class AutoEncoderModels(nn.Module):

    def __init__(self, device, layers_list=None, load_from_file =False, json_filepath= None, edge_index=None,  **kwargs):
        super().__init__()
        self.device = device
        self.models = dict()
        self.edge_index = edge_index
        self.permutation_forward = dict()
        self.graph_forward = dict()
        if load_from_file:
            self.layers_list = self.load_fileJson(json_filepath)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        self.models_layers['encoder'], self.models_size['encoder'], self.permutation_forward['encoder'], self.graph_forward['encoder'] = self.list_to_model(self.layers_list['encoder_layers'])
        self.models_layers['decoder'], self.models_size['decoder'], self.permutation_forward['decoder'], self.graph_forward['decoder'] = self.list_to_model(self.layers_list['decoder_layers'])
        self.deploy_nnModel()
        
    def get_size(self, ):
        return self.models_size
        
    def deploy_nnModel(self):
        
        if self.edge_index is not None:
            self.models['encoder'] = nn_Model(layers= self.models_layers['encoder'], permutation_forward = self.permutation_forward['encoder'], edge_index= self.edge_index ,graph_forward=self.graph_forward['encoder'])
            self.models['decoder'] = nn_Model(layers= self.models_layers['decoder'], permutation_forward = self.permutation_forward['decoder'], edge_index= self.edge_index ,graph_forward=self.graph_forward['decoder'])
        else:
            self.models['encoder'] = nn_Model(layers= self.models_layers['encoder'])
            self.models['decoder'] = nn_Model(layers= self.models_layers['decoder'])
        
        self.add_module('encoder', self.models['encoder'])
        self.add_module('decoder', self.models['decoder'])
        
        
    def get_decoder(self, extra_info=False):
        if extra_info:
            return self.models['decoder'], self.models_size['decoder'], self.permutation_forward['decoder']
        else:
            return self.models['decoder']
        

    def get_encoder(self, extra_info=False):
        if extra_info:
            return self.models['encoder'], self.models_size['encoder'], self.permutation_forward['encoder']
        else:
            return self.models['encoder']       
    
    def summary(self):
        summary = dict()
        print("models_size[encoder][input_size]\t\t\t",self.models_size['encoder']["input_size"])
        print("models_size[decoder][input_size]\t\t\t",self.models_size['decoder']["input_size"])
        summary['encoder'] = torchinfo.summary(self.models['encoder'], input_size=(1, self.models_size['encoder']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary['decoder'] = torchinfo.summary(self.models['decoder'], input_size=(1, self.models_size['decoder']["input_size"]), device=self.device, batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
    
    def forward(self, x):
        x_latent = self.models['encoder'](x)
        x_hat = self.models['decoder'](x_latent["x_output"]['data'])
        return {"x_input":{"data":x}, "x_latent":{"latent":x_latent["x_output"]['data']}, "x_output":{"data":x_hat["x_output"]['data']}}

   
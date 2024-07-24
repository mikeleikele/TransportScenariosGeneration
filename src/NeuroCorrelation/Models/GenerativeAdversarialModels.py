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


class GenerativeAdversarialModels(nn.Module):

    def __init__(self, layers_list=None, load_from_file =False, json_filepath= None, edge_index=None, **kwargs):
        super().__init__()
        self.models = dict()
        self.edge_index = edge_index
        if load_from_file:
            self.layers_list = self.load_fileJson(json_filepath)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        self.models_layers['discriminator'], self.models_size['discriminator'] = self.list_to_model(self.layers_list['discriminator_layers'])
        self.models_layers['generator'], self.models_size['generator'] = self.list_to_model(self.layers_list['generator_layers'])
        self.deploy_nnModel()
        
    def deploy_nnModel(self):
        
        if self.edge_index is not None:
            self.models['discriminator'] = nn_Model(layers= self.models_layers['discriminator'], edge_index= self.edge_index)
            self.models['generator'] = nn_Model(layers= self.models_layers['generator'], edge_index= self.edge_index)
        else:
            self.models['discriminator'] = nn_Model(layers= self.models_layers['discriminator'])
            self.models['generator'] = nn_Model(layers= self.models_layers['generator'])
        
        self.add_module('discriminator', self.models['discriminator'])
        self.add_module('generator', self.models['generator'])
        
        
    def get_discriminator(self):
        return self.models['discriminator']

    def get_generator(self):
        return self.models['generator']       
    
    def summary(self):
        summary = dict()
        summary['encoder'] = torchinfo.summary(self.models['encoder'], input_size=(1, self.models_size['encoder']["input_size"]), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary['decoder'] = torchinfo.summary(self.models['decoder'], input_size=(1, self.models_size['decoder']["input_size"]), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
    
    def forward(self, x):
        x_latent = self.models['encoder'](x)
        x_hat = self.models['decoder'](x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def list_to_model(self, layers_list):
        layers = list()
        size = {"input_size":None, "output_size":None}
        for layer_item in layers_list:
            
            #layer
            if layer_item['layer'] == "Linear":
                layers.append(nn.Linear(in_features=layer_item['in_features'], out_features=layer_item['out_features']))
                if size["input_size"] == None:
                    size["input_size"] = layer_item['in_features']
                size["output_size"] = layer_item['out_features']
            elif layer_item['layer'] == "GCNConv":
                layers.append(gm.GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
            
            #activation function
            elif layer_item['layer'] == "Tanh":
                layers.append(nn.Tanh())
            elif layer_item['layer'] == "LeakyReLU":
                layers.append(nn.LeakyReLU(layer_item['negative_slope']))
            elif layer_item['layer'] == "sigmoid":
                layers.append(nn.Sigmoid())
            #batch norm
            elif layer_item['layer'] == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(num_features=layer_item['num_features'], affine=layer_item['affine']))
            
            #dropout
            elif layer_item['layer'] == "Dropout":
                layers.append(nn.Dropout(p=layer_item['p']))
        return layers, size

    def load_fileJson(self, filepath):
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        layers_list = dict()
        layers_list["discriminator_layers"] = config["GAN"]["discriminator_layers"]
        layers_list["generator_layers"] = config["GAN"]["generator_layers"]
        return layers_list

class nn_Model(nn.Module):
    def __init__(self, layers, edge_index=None):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.edge_index = edge_index
        print("Layers initialized:", self.layers)
        
    def forward(self, x):
        for layer in self.layers:
            if isinstance(layer, gm.GCNConv):
                x = layer(x, self.edge_index)
            else:
                x = layer(x)
        return {"x_input": x, "x_output": x}
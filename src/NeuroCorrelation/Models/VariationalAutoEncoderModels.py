
import numpy as np
import torch

import torchinfo
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
from torch_geometric.nn import GCNConv, ChebConv, GATConv
from pathlib import Path
import os
import json
from .BaseModel import BaseModel

class VariationalAutoEncoderModels(nn.Module, BaseModel):

    def __init__(self, device, layers_list=None, load_from_file=False, json_filepath=None, edge_index=None, **kwargs):
        super().__init__()
        self.model_type="VAE"
        
        self.device = device
        self.models = dict()
        self.edge_index = edge_index
        self.permutation_forward = dict()
        self.models_layers_parallel = dict()
        self.layers_name = dict()
        if load_from_file:
            print("json_filepathjson_filepath",json_filepath)
            self.layers_list = self.load_fileJson(json_filepath, self.model_type)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        # Encoder
        self.models_layers['encoder'], self.models_size['encoder'], self.permutation_forward['encoder'], self.models_layers_parallel['encoder'], self.layers_name['encoder'] = self.list_to_model(self.layers_list['encoder_layers'])
        
        

        # Decoder
        self.models_layers['decoder'], self.models_size['decoder'], self.permutation_forward['decoder'], self.models_layers_parallel['decoder'], self.layers_name['decoder'], = self.list_to_model(self.layers_list['decoder_layers'])

        # Deploy the VAE model
        self.deploy_vae_model()
        
    def get_size(self):
        return self.models_size
    
    def deploy_vae_model(self):
        # Latent space layers for mean and logvar (specific to VAE)
        self.fc_mu = nn.Linear(self.models_size['encoder']["output_size"], self.models_size['encoder']["output_size"])
        self.fc_logvar = nn.Linear(self.models_size['encoder']["output_size"], self.models_size['encoder']["output_size"])
        
        if self.edge_index is not None:
            self.models['encoder'] = nn_Model(layers=self.models_layers['encoder'], permutation_forward=self.permutation_forward['encoder'], edge_index=self.edge_index, parallel_layers=self.models_layers_parallel['encoder'], layers_name =self.layers_name['encoder'])
            self.models['decoder'] = nn_Model(layers=self.models_layers['decoder'], permutation_forward=self.permutation_forward['decoder'], edge_index=self.edge_index, parallel_layers=self.models_layers_parallel['decoder'], layers_name =self.layers_name['decoder'])
        else:
            self.models['encoder'] = nn_Model(layers=self.models_layers['encoder'], parallel_layers=self.models_layers_parallel['encoder'])
            self.models['decoder'] = nn_Model(layers=self.models_layers['decoder'], parallel_layers=self.models_layers_parallel['decoder'])

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
        print("models_size[encoder][input_size]\t\t\t", self.models_size['encoder']["input_size"])
        print("models_size[decoder][input_size]\t\t\t", self.models_size['decoder']["input_size"])
        summary['encoder'] = torchinfo.summary(self.models['encoder'], input_size=(1, self.models_size['encoder']["input_size"]), device=self.device, batch_dim=0, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose=0)
        summary['decoder'] = torchinfo.summary(self.models['decoder'], input_size=(1, self.models_size['decoder']["input_size"]), device=self.device, batch_dim=0, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose=0)

    def forward(self, x):
        x_latent = self.models['encoder'](x)        
        mu = x_latent["mu"]
        logvar = x_latent["logvar"]                
        z = self.reparameterize(mu, logvar)
        
        x_hat = self.models['decoder'](z)        
        return {"x_input": {"data":x}, "x_latent":{"mu": mu, "logvar": logvar, "z":z}, "x_output": {"data": x_hat["x_output"]['data']}}

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        reparameterized = mu + eps * std        
        return reparameterized


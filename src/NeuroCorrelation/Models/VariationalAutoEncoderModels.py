
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
from .BaseModel import BaseModel, nn_Model
from termcolor import cprint
from colorama import init, Style

class VariationalAutoEncoderModels(nn.Module, BaseModel):

    def __init__(self, device, channels_dim, layers_list=None, load_from_file=False, json_filepath=None, edge_index=None, **kwargs):
        super().__init__()
        self.model_type="VAE"
        
        self.device = device
        self.models = dict()
        self.edge_index = edge_index
        self.permutation_forward = dict()
        self.models_layers_parallel = dict()
        self.layers_name = dict()
        if load_from_file:
            cprint(Style.BRIGHT + f"| Load model from: {json_filepath}" + Style.RESET_ALL, 'black', attrs=["bold"])
            self.layers_list, self.channel_modes = self.load_fileJson(json_filepath, self.model_type)
        else:
            self.layers_list = layers_list
        self.models_layers = dict()
        self.models_size = dict()
        # Encoder
        self.models_layers['encoder'], self.models_size['encoder'], self.permutation_forward['encoder'], self.models_layers_parallel['encoder'], self.layers_name['encoder'] = self.list_to_model(self.layers_list['encoder_layers'])
        self.channels_dim = channels_dim
        

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
            self.models['encoder'] = nn_Model(layers=self.models_layers['encoder'], permutation_forward=self.permutation_forward['encoder'], edge_index=self.edge_index, parallel_layers=self.models_layers_parallel['encoder'], channels_dim=self.channels_dim, channels=self.channel_modes, layers_name =self.layers_name['encoder'])
            self.models['decoder'] = nn_Model(layers=self.models_layers['decoder'], permutation_forward=self.permutation_forward['decoder'], edge_index=self.edge_index, parallel_layers=self.models_layers_parallel['decoder'], channels_dim=self.channels_dim, channels=self.channel_modes, layers_name =self.layers_name['decoder'])
        else:
            self.models['encoder'] = nn_Model(layers=self.models_layers['encoder'], channels_dim=self.channels_dim['encoder'], channel_modes=self.channel_modes['encoder'], parallel_layers=self.models_layers_parallel['encoder'])
            self.models['decoder'] = nn_Model(layers=self.models_layers['decoder'], channels_dim=self.channels_dim['decoder'], channel_modes=self.channel_modes['decoder'], parallel_layers=self.models_layers_parallel['decoder'])

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

    def set_model_mode(self, parts=['encoder','decoder'], eval_mode=True):
        BLUE = '\033[38;5;27m'
        print(f"{Style.BRIGHT}{BLUE}| Model set mode{Style.RESET_ALL}")
        for part in parts:
            if eval_mode:
                self.models[part].eval()
                print(f"{Style.BRIGHT}{BLUE}| \t{part} SET to EVAL{Style.RESET_ALL}")
            else:
                self.models[part].train()
                print(f"{Style.BRIGHT}{BLUE}| \t{part} SET to TRAIN{Style.RESET_ALL}")
        
    def get_model_mode(self, parts=['encoder', 'decoder']):
        BLUE = '\033[38;5;27m'
        print(f"{Style.BRIGHT}{BLUE}| Model mode:{Style.RESET_ALL}")
        for part in parts:
            mode = "EVAL" if not self.models[part].training else "TRAIN"
            print(f"{Style.BRIGHT}{BLUE}| \t{part} -> {mode}{Style.RESET_ALL}")
    
            

    def forward(self, x, info_eval=False):
        x_latent = self.models['encoder'](x)
        mu = x_latent["mu"]["data"]
        logvar = x_latent["logvar"]["data"]             
        z = self.reparameterize(mu, logvar)
        x_hat = self.models['decoder'](z)  
        return {"x_input": {"data":x}, "x_latent":{"mu": mu, "logvar": logvar, "z":z}, "x_output": {"data": x_hat["x_output"]['data']}}
        

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        reparameterized = mu + eps * std        
        return reparameterized


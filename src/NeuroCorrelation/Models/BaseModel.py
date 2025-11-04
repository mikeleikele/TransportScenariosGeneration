
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
from termcolor import cprint
from colorama import init, Style

class BaseModel:

    def load_fileJson(self, filepath, model_type):
        with open(filepath, 'r') as f:
            config = json.load(f)
        layers_list = dict()
        channel_modes = dict()

        if model_type in ["AE","VAE","CVAE"]:
            layers_list["encoder_layers"] = config[model_type]["encoder"]["layers"]
            layers_list["decoder_layers"] = config[model_type]["decoder"]["layers"]
        
        elif model_type == "GAN":
            layers_list["discriminator_layers"] = config[model_type]["discriminator_layers"]
            layers_list["generator_layers"] = config[model_type]["generator_layers"]

        channel_modes["encoder"] = {
            "in": config[model_type]["encoder"]["input_mode"],
            "out": config[model_type]["encoder"]["output_mode"]
        }
        channel_modes["decoder"] = {
            "in": config[model_type]["decoder"]["input_mode"],
            "out": config[model_type]["decoder"]["output_mode"]
        }
        return layers_list, channel_modes
    
    def list_to_model(self, layers_list):
        layers = list()
        parallel_layers_flag = False
        parallel_layers = []
        permutation_forward = dict()
        layers_name = dict()
        size = {"input_size": None, "output_size": None}
        
        for index, layer_item in enumerate(layers_list):
            parallel_layers_name = None
            # Layers
            
            if layer_item['layer'] == "Linear":
                layers.append(nn.Linear(in_features=layer_item['in_features'], out_features=layer_item['out_features']))
                if size["input_size"] is None:
                    size["input_size"] = layer_item['in_features']
                size["output_size"] = layer_item['out_features']


            elif layer_item['layer'] == "GCNConv" or layer_item['layer'] == "GCNConv_Permute":
                layers.append(GCNConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels']))
                if layer_item['layer'] == "GCNConv_Permute":
                    permutation_forward[index] = {"in_permute": layer_item['in_permute'], "out_permute": layer_item['out_permute']}


            elif layer_item['layer'] == "ChebConv" or layer_item['layer'] == "ChebConv_Permute":
                layers.append(ChebConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels'], K=layer_item['K']))
                if layer_item['layer'] == "ChebConv_Permute":
                    permutation_forward[index] = {"in_permute": layer_item['in_permute'], "out_permute": layer_item['out_permute']}

            elif layer_item['layer'] == 'GATConv' or layer_item['layer'] == "GATConv_Permute":
                layers.append(GATConv(in_channels=layer_item['in_channels'], out_channels=layer_item['out_channels'], heads=layer_item['heads'], concat=layer_item['concat'],dropout=layer_item['dropout']))
                if layer_item['layer'] == "GATConv_Permute":
                    permutation_forward[index] = {"in_permute": layer_item['in_permute'], "out_permute": layer_item['out_permute']}





            # Activation functions
            elif layer_item['layer'] == "Tanh":
                layers.append(nn.Tanh())
            elif layer_item['layer'] == "LeakyReLU":
                layers.append(nn.LeakyReLU(layer_item['negative_slope']))
            elif layer_item['layer'] == "Sigmoid":
                layers.append(nn.Sigmoid())
            
            # Batch normalization
            elif layer_item['layer'] == "BatchNorm1d":
                layers.append(nn.BatchNorm1d(num_features=layer_item['num_features'], affine=layer_item['affine']))

            # Dropout
            elif layer_item['layer'] == "Dropout":
                layers.append(nn.Dropout(p=layer_item['p']))
            
            
                
            elif layer_item['layer'] == "Parallel":
                parallel_layers_flag = True       
                
                for sub_layer in layer_item['layers']:
                    sub_layers, sub_size, _, _, sub_name = self.list_to_model(sub_layer['layers'])
                    parallel_layers.append((sub_layer['name'], nn.Sequential(*sub_layers)))
            
                layers_name[index] = {"parallel": [name for name, _ in parallel_layers]}
                layers.append(parallel_layers)
                
            if 'name' in layer_item:
                layers_name[index] = layer_item['name']
            elif parallel_layers_flag:
                layers_name[index] = {"parallel": [sub_layer['name'] for sub_layer in layer_item['layers']]}
            else:
                layers_name[index] = f"{layer_item['layer']}_{index}"
        
        return layers, size, permutation_forward, parallel_layers_flag, layers_name


class nn_Model(nn.Module):
    def __init__(self, layers, channel_modes, channels_dim, permutation_forward=None, edge_index=None, parallel_layers=False, layers_name=None):
       
        super().__init__()

        self.edge_index = edge_index
        self.permutation_forward = permutation_forward or {}
        self.layers_name = layers_name or {}
        self.parallel_layers_flag = parallel_layers
        self.channel_modes = channel_modes
        self.sequential_layers = nn.Sequential()
        self.parallel_blocks = nn.ModuleDict()
        self.channels_dim = channels_dim
        self._initialize_layers(layers)
        self.apply(self.weights_init_normal)
        
        # per-road transform as submodule
        self.channel_transform_in  = ChannelTransform(mode="in", type=self.channel_modes, channels_dim=self.channels_dim)
        self.channel_transform_out = ChannelTransform(mode="out", type=self.channel_modes, channels_dim=self.channels_dim)
        
        
        BLUE = '\033[38;5;27m'

        print(f"{Style.BRIGHT}{BLUE}| Model inizialization:{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{BLUE}|  Sequential layers:\t {self.sequential_layers}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{BLUE}|  Parallel blocks:\t {self.parallel_blocks}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{BLUE}| Channels:{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{BLUE}|  Input  {self.channels_dim['in']} \tmode: {self.channel_modes['in']}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}{BLUE}|  Output {self.channels_dim['out']}\tmode: {self.channel_modes['out']}{Style.RESET_ALL}")
        

    def _initialize_layers(self, layers):
        sequential_layers = []
        for index, layer in enumerate(layers):
            if isinstance(layer, list):
                for name, sub_block in layer:
                    self.parallel_blocks[name] = nn.Sequential(*sub_block)
            else:
                sequential_layers.append(layer)
        self.sequential_layers = nn.Sequential(*sequential_layers)

    def weights_init_normal(self, m):        
        if isinstance(m, nn.Linear):
            init_mode = "xavier_uniform"
            if init_mode == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=0.01)
            elif init_mode == "normal_":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_in = x
        forward_dict = {"x_input": {'data': x}}
        x = self.channel_transform_in(x)
        # Process through sequential layers
        for index, layer in enumerate(self.sequential_layers):            
            if isinstance(layer, (GCNConv, ChebConv, GATConv)):
                if index in self.permutation_forward:
                    in_permute = self.permutation_forward[index]["in_permute"]
                    x = x.permute(*in_permute)
                x = layer(x, self.edge_index)
                if index in self.permutation_forward:
                    out_permute = self.permutation_forward[index]["out_permute"]
                    x = x.permute(*out_permute)
            else:
                x = layer(x)
        
        x = self.channel_transform_out(x)
        forward_dict["x_output"] = {'data': x}
        
        # Process parallel blocks if present
        if self.parallel_layers_flag:
            for block_name, block in self.parallel_blocks.items():
                forward_dict[block_name] = {'data': block(x)}
        return forward_dict

class ChannelTransform(nn.Module):
    def __init__(self, mode: str, type: str, channels_dim: dict):
        """
        Trasformazione per gestire input/output con struttura per-road.
        
        Args:
            mode: 'in' o 'out' - direzione della trasformazione
            type: 'per_road', 'flat', o 'none'
            channels_dim: dict con 'in' e 'out' che indicano il numero di canali
        
        Comportamento:
        - type='per_road' + mode='in':  (B, R, C_in) → (B, R) via Linear(C_in, 1) per ogni road
        - type='per_road' + mode='out': (B, R) → (B, R, C_out) via Linear(1, C_out) per ogni road
        - type='flat' + mode='in':      (B, R, C) → (B, R*C) via view
        - type='flat' + mode='out':     (B, R*C) → (B, R, C) via view
        - type='none':                  passthrough (nessuna trasformazione)
        """
        super().__init__()
        self.mode = mode
        self.type = type
        self.channels_dim = channels_dim
        
        if self.type[self.mode] == "per_road":
            if self.mode == "in":
                # Ogni road: (C_in) → (1) tramite Linear
                C_in = channels_dim["in"]
                self.linear = nn.Linear(C_in, 1)
                #print(f"PerRoadTransform 'in' per_road: Linear({C_in}, 1) per ogni road")
            elif self.mode == "out":
                # Ogni road: (1) → (C_out) tramite Linear
                C_out = channels_dim["out"]
                self.linear = nn.Linear(1, C_out)
                #print(f"PerRoadTransform 'out' per_road: Linear(1, {C_out}) per ogni road")
            else:
                raise ValueError(f"mode deve essere 'in' o 'out', ricevuto: {mode}")
                
        elif self.type[self.mode]  == "flat":
            a = 0
            #print(f"PerRoadTransform '{self.mode}' flat: usa view per appiattimento/ricostruzione")
            
        elif self.type[self.mode]  == "none":
            a = 0
            #print(f"PerRoadTransform '{self.mode}' none: nessuna trasformazione (passthrough)")
            
        else:
            a = 0
            #raise ValueError(f"type deve essere 'per_road', 'flat' o 'none', ricevuto: {type}")

    def forward(self, x):
        if self.type[self.mode]  == "none":
            # Nessuna trasformazione
            return x
        
        if self.type[self.mode]  == "per_road":
            if self.mode == "in":
                # Input:  (B, R, C_in)
                # Linear: (B, R, C_in) → (B, R, 1)
                # Output: (B, R)
                if x.dim() != 3:
                    raise ValueError(f"Per mode='in' e type='per_road' aspetto (B,R,C), ricevuto shape {x.shape}")
                
                x = self.linear(x)  # (B, R, C_in) → (B, R, 1)
                x = x.squeeze(-1)   # (B, R, 1) → (B, R)
                return x
                
            elif self.mode == "out":
                # Input:  (B, R)
                # Expand: (B, R, 1)
                # Linear: (B, R, 1) → (B, R, C_out)
                # Output: (B, R, C_out)
                if x.dim() != 2:
                    raise ValueError(f"Per mode='out' e type='per_road' aspetto (B,R), ricevuto shape {x.shape}")
                
                x = x.unsqueeze(-1)  # (B, R) → (B, R, 1)
                x = self.linear(x)   # (B, R, 1) → (B, R, C_out)
                
                return x
        
        elif self.type[self.mode]  == "flat":
            if self.mode == "in":
                # Input:  (B, R, C)
                # Output: (B, R*C)
                if x.dim() != 3:
                    raise ValueError(f"Per mode='in' e type='flat' aspetto (B,R,C), ricevuto shape {x.shape}")
                
                B, R, C = x.shape
                x = x.view(B, R * C)
                return x
                
            elif self.mode == "out":
                # Input:  (B, R*C)
                # Output: (B, R, C)
                if x.dim() != 2:
                    raise ValueError(f"Per mode='out' e type='flat' aspetto (B,R*C), ricevuto shape {x.shape}")
                
                B = x.shape[0]
                C_out = self.channels_dim.get("out")
                if C_out is None:
                    raise ValueError("Per mode='out' e type='flat' serve channels_dim['out']")
                
                total = x.shape[1]
                if total % C_out != 0:
                    raise ValueError(f"Dimensione {total} non divisibile per C_out={C_out}")
                
                R = total // C_out
                x = x.view(B, R, C_out)
                return x
        
        return x
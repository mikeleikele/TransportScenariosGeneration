import torch.nn as nn
import json

class BaseModel:

    def load_fileJson(self, filepath, model_type):
        with open(filepath, 'r') as f:
            config = json.load(f)
        layers_list = dict()
        if model_type in ["AE","VAE","CVAE"]:
            layers_list["encoder_layers"] = config[model_type]["encoder_layers"]
            layers_list["decoder_layers"] = config[model_type]["decoder_layers"]
        
        elif model_type == "GAN":
            layers_list["discriminator_layers"] = config[model_type]["discriminator_layers"]
            layers_list["generator_layers"] = config[model_type]["generator_layers"]
    
        return layers_list
    
    def list_to_model(self, layers):
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
    def __init__(self, layers, permutation_forward=None, edge_index=None, parallel_layers=False, layers_name=None):
       
        super().__init__()

        self.edge_index = edge_index
        self.permutation_forward = permutation_forward or {}
        self.layers_name = layers_name or {}
        self.parallel_layers_flag = parallel_layers

        # Inizializza i layer sequenziali e paralleli
        self.sequential_layers = nn.Sequential()
        self.parallel_blocks = nn.ModuleDict()
        
        self._initialize_layers(layers)
        self.apply(self.weights_init_normal)
        
        print("Model inizialization:")
        print(f" - Sequential layers:\t {self.sequential_layers}")
        print(f" - Parallel blocks:\t {self.parallel_blocks}")
        
        

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
        forward_dict = {"x_input": {'data':x}}
        for index, layer in enumerate(self.sequential_layers):            
            if isinstance(layer, GCNConv, ChebConv, GATConv):
                if index in self.permutation_forward:
                    in_permute = self.permutation_forward[index]["in_permute"]
                    x = x.permute(*in_permute)
                x = layer(x, self.edge_index)
                if index in self.permutation_forward:
                    out_permute = self.permutation_forward[index]["out_permute"]
                    x = x.permute(*out_permute)
            else:
                x = layer(x)
        forward_dict["x_output"] = {'data':x}

        if self.parallel_layers_flag:
            for block_name, block in self.parallel_blocks.items():
                forward_dict[block_name] = block(x)
        return forward_dict
    

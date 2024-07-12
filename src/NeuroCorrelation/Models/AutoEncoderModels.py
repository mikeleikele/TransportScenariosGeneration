<<<<<<< HEAD
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

class AutoEncoderModels():

    def __init__(self, model, named_parameters_dict, univar_counts, folder):
        self.univar_counts = univar_counts
        self.folder = folder
        if not os.path.exists(folder):
                os.makedirs(folder)
        self.model = model
        self.named_parameters_dict = named_parameters_dict
        
    
    def get_model(self):
        if self.model_case=="fullyRectangle":
            return self.fullyRectangle()
        else:
            return None

    def drawModel(self):
        dot = make_dot(self.model,  params=self.named_parameters_dict, show_attrs=True, show_saved=True)
        dot.format = 'png'
        path_file = Path(self.folder,"model_plot.png")
        dot.render(filename=path_file)



    def fullyRectangle(self):
        model = GEN_fl()
        return model

class GEN_fl(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_2 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_3 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_4 = nn.Linear(in_features=78, out_features=78)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        x_hat = F.tanh(layer_nn)
        return {"x_input":x, "x_latent":None, "x_output":x_hat}

class GEN_autoEncoder_78(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_78()
        self.decoder = GEN_autoEncoder_Decoder_78()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent)
        return {"x_input":x, "x_latent":x_latent, "x_output":x_hat}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_78(nn.Module):
    def __init__(self):
       
        super().__init__()
        #self.bn_1 = nn.BatchNorm1d(40)
        #20 30 38 46 62 78
        self.hidden_layer_1 = nn.Linear(in_features=78, out_features=62)
        self.hidden_layer_2 = nn.Linear(in_features=62, out_features=46)
        self.hidden_layer_3 = nn.Linear(in_features=46, out_features=38)
        self.hidden_layer_4 = nn.Linear(in_features=38, out_features=30)
        self.hidden_layer_5 = nn.Linear(in_features=30, out_features=20)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        #layer_nn = self.bn_1(layer_nn)
        return layer_nn

class GEN_autoEncoder_Decoder_78(nn.Module):

    def __init__(self):
       
        super().__init__()
        #20 30 38 46 62 78
        #self.bn_1 = nn.BatchNorm1d(40)
        self.hidden_layer_1 = nn.Linear(in_features=20, out_features=30)
        self.hidden_layer_2 = nn.Linear(in_features=30, out_features=38)
        self.hidden_layer_3 = nn.Linear(in_features=38, out_features=46)
        self.hidden_layer_4 = nn.Linear(in_features=46, out_features=62)
        self.hidden_layer_5 = nn.Linear(in_features=62, out_features=78)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        return layer_nn


class GEN_autoEncoder_3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_3()
        self.decoder = GEN_autoEncoder_Decoder_3()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):       
        enc_summary = []#torchinfo.summary(self.encoder, (1, 1, 1, 7), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = []#torchinfo.summary(self.decoder, (1, 1, 1, 4), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_3(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=7, out_features=24)
        self.hidden_layer_2 = nn.Linear(in_features=24, out_features=18)
        self.hidden_layer_3 = nn.Linear(in_features=18, out_features=12)
        self.hidden_layer_4 = nn.Linear(in_features=12, out_features=4)
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(4, affine=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        #print(layer_nn.shape)
        x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_3(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=4, out_features=12)
        self.hidden_layer_2 = nn.Linear(in_features=12, out_features=18)
        self.hidden_layer_3 = nn.Linear(in_features=18, out_features=24)
        self.hidden_layer_4 = nn.Linear(in_features=24, out_features=7)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_4(layer_nn)
        return {"x_input":x, "x_output":x_out}


class GEN_autoEncoder_16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_16()
        self.decoder = GEN_autoEncoder_Decoder_16()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 16), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 12), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_16(nn.Module):
    def __init__(self):

        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=16, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=12)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        
        self.dp_1 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
                
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_16(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=12, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=16)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        self.dp_1 = nn.Dropout(p=0.2)        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        #layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
        

class GEN_autoEncoder_325(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_325()
        self.decoder = GEN_autoEncoder_Decoder_325()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_325(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=325, out_features=896)
        self.hidden_layer_2 = nn.Linear(in_features=896, out_features=642)
        self.hidden_layer_3 = nn.Linear(in_features=642, out_features=524)
        self.hidden_layer_4 = nn.Linear(in_features=524, out_features=448)
        self.hidden_layer_5 = nn.Linear(in_features=448, out_features=342)
        self.hidden_layer_6 = nn.Linear(in_features=342, out_features=280)
        self.hidden_layer_7 = nn.Linear(in_features=280, out_features=224)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

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
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_325(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=224, out_features=280)
        self.hidden_layer_2 = nn.Linear(in_features=280, out_features=342)
        self.hidden_layer_3 = nn.Linear(in_features=342, out_features=448)
        self.hidden_layer_4 = nn.Linear(in_features=448, out_features=524)
        self.hidden_layer_5 = nn.Linear(in_features=524, out_features=642)
        self.hidden_layer_6 = nn.Linear(in_features=642, out_features=896)
        self.hidden_layer_7 = nn.Linear(in_features=896, out_features=325)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
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
    
class GEN_autoEncoder_207(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_207()
        self.decoder = GEN_autoEncoder_Decoder_207()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_207(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=207, out_features=800)
        self.hidden_layer_2 = nn.Linear(in_features=800, out_features=642)
        self.hidden_layer_3 = nn.Linear(in_features=642, out_features=448)
        self.hidden_layer_4 = nn.Linear(in_features=448, out_features=224)
        self.hidden_layer_5 = nn.Linear(in_features=224, out_features=112)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_5(layer_nn)
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_207(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_3 = nn.Linear(in_features=112, out_features=224)
        self.hidden_layer_4 = nn.Linear(in_features=224, out_features=448)
        self.hidden_layer_5 = nn.Linear(in_features=448, out_features=642)
        self.hidden_layer_6 = nn.Linear(in_features=642, out_features=800)
        self.hidden_layer_7 = nn.Linear(in_features=800, out_features=207)

    def forward(self, x):
        layer_nn = self.hidden_layer_3(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_7(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}


class GEN_ConvAutoEncoder_7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_ConvAutoEncoder_Encoder_7()
        self.decoder = GEN_ConvAutoEncoder_Decoder_7()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_ConvAutoEncoder_Encoder_7(nn.Module):
    # in  1, 1, 7, 32
    # out 1, 1, 1, 1
    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Conv2d(in_channels= 1 , out_channels= 3 , kernel_size= 3 , stride= 2 , padding= 1 )
        self.hidden_layer_2 = nn.Conv2d(in_channels= 3 , out_channels= 4 , kernel_size= 2 , stride= 2 , padding= 2 )
        self.hidden_layer_3 = nn.Conv2d(in_channels= 4 , out_channels= 3 , kernel_size= 3 , stride= 1 , padding= (1,2) )
        self.hidden_layer_4 = nn.Conv2d(in_channels= 3 , out_channels= 2 , kernel_size= (2,3) , stride= 1 , padding= (0,1) )
        self.hidden_layer_5 = nn.Conv2d(in_channels= 2 , out_channels= 1 , kernel_size= 2 , stride= (1,2) , padding= 0 )

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

class GEN_ConvAutoEncoder_Decoder_7(nn.Module):
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


####################################


class GEN_autoEncoder_6k(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_6k()
        self.decoder = GEN_autoEncoder_Decoder_6k()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_6k(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=5943, out_features=5000)
        self.hidden_layer_2 = nn.Linear(in_features=5000, out_features=4500)
        self.hidden_layer_3 = nn.Linear(in_features=4500, out_features=4000)
        self.hidden_layer_4 = nn.Linear(in_features=4000, out_features=3000)
        self.hidden_layer_5 = nn.Linear(in_features=3000, out_features=2000)
        self.hidden_layer_6 = nn.Linear(in_features=2000, out_features=1200)
        self.hidden_layer_7 = nn.Linear(in_features=1200, out_features=750)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

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
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_6k(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=750, out_features=1200)
        self.hidden_layer_2 = nn.Linear(in_features=1200, out_features=2000)
        self.hidden_layer_3 = nn.Linear(in_features=2000, out_features=3000)
        self.hidden_layer_4 = nn.Linear(in_features=3000, out_features=4000)
        self.hidden_layer_5 = nn.Linear(in_features=4000, out_features=4500)
        self.hidden_layer_6 = nn.Linear(in_features=4500, out_features=5000)
        self.hidden_layer_7 = nn.Linear(in_features=5000, out_features=5943)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
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
    
    
##########

class GEN_autoEncoder_05k(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_05k()
        self.decoder = GEN_autoEncoder_Decoder_05k()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_05k(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=200, out_features=400)
        self.hidden_layer_2 = n# The code appears to be a Python script that may be related to a
        # Generative Adversarial Network (GAN) using Graph Convolutional
        # Networks (GCN) with linear layers, possibly utilizing a pretrained
        # ResNet-0128 model. The script may be designed for a specific task or
        # project related to image generation or manipulation. The mention of
        # "CHENGDU" and "bt" is not clear from the code snippet provided.
        n.Linear(in_features=400, out_features=300)
        self.hidden_layer_3 = nn.Linear(in_features=300, out_features=250)
        self.hidden_layer_4 = nn.Linear(in_features=250, out_features=200)
        self.hidden_layer_5 = nn.Linear(in_features=200, out_features=150)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_5(layer_nn)
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_05k(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_3 = nn.Linear(in_features=150, out_features=200)
        self.hidden_layer_4 = nn.Linear(in_features=200, out_features=250)
        self.hidden_layer_5 = nn.Linear(in_features=250, out_features=300)
        self.hidden_layer_6 = nn.Linear(in_features=300, out_features=400)
        self.hidden_layer_7 = nn.Linear(in_features=400, out_features=200)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_3(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_7(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}
    
    


    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_64()
        self.decoder = GEN_autoEncoder_Decoder_64()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        enc_summary = torchsummary.summary(self.encoder, (1, 1, 64), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchsummary.summary(self.decoder, (1, 1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

#---------------------------

#---------------------------
# MODEL EXOGENOUS  
class GEN_autoEncoder_Encoder_exogenous_7(nn.Module):
    def __init__(self,):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=7, out_features=5)
        self.hidden_layer_2 = nn.Linear(in_features=5, out_features=3)
        
        self.act_1 = nn.LeakyReLU(0.2)        
        self.dp_1 = nn.Dropout(p=0.2)
       
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}
#---------------------------

#---------------------------
# MODEL 32  
class GEN_autoEncoder_32(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_32()
        self.decoder = GEN_autoEncoder_Decoder_32()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 32), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 30), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_32(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=32, out_features=28)
        self.hidden_layer_2 = nn.Linear(in_features=28, out_features=22)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_32(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=22, out_features=28)
        self.hidden_layer_2 = nn.Linear(in_features=28, out_features=32)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)    
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 48
class GEN_autoEncoder_48(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_48()
        self.decoder = GEN_autoEncoder_Decoder_48()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 44), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_48(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=48, out_features=96)
        #self.hidden_layer_2 = nn.Linear(in_features=64, out_features=96)        
        #self.hidden_layer_3 = nn.Linear(in_features=96, out_features=128)
        self.hidden_layer_4 = nn.Linear(in_features=96, out_features=64)        
        self.hidden_layer_5 = nn.Linear(in_features=64, out_features=58)
        self.hidden_layer_6 = nn.Linear(in_features=58, out_features=52)
        self.hidden_layer_7 = nn.Linear(in_features=52, out_features=48)
        self.hidden_layer_8 = nn.Linear(in_features=48, out_features=44)
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)

    def forward(self, x):
        
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.batch_norm_1(layer_nn)
        
        #layer_nn = self.hidden_layer_2(layer_nn)
        #layer_nn = F.tanh(layer_nn)
        #layer_nn = self.hidden_layer_3(layer_nn)
        #layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_8(layer_nn)        
        
        #layer_nn = self.batch_norm_1(layer_nn)
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_48(nn.Module):
           
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=44, out_features=48)
        self.hidden_layer_2 = nn.Linear(in_features=48, out_features=52)
        self.hidden_layer_3 = nn.Linear(in_features=52, out_features=58)
        self.hidden_layer_4 = nn.Linear(in_features=58, out_features=64)
        self.hidden_layer_5 = nn.Linear(in_features=64, out_features=96)
        #self.hidden_layer_6 = nn.Linear(in_features=128, out_features=96)
        #self.hidden_layer_7 = nn.Linear(in_features=96, out_features=64)
        self.hidden_layer_8 = nn.Linear(in_features=96, out_features=48)
                
    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_1(layer_nn)
        
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_2(layer_nn)
        
        
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_3(layer_nn)
        
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_4(layer_nn)
        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        ##layer_nn = self.dropout_5(layer_nn)
        
        ##layer_nn = self.hidden_layer_6(layer_nn)
        ##layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_6(layer_nn)
        
        ##layer_nn = self.hidden_layer_7(layer_nn)
        ##layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_6(layer_nn)
        
        x_out = self.hidden_layer_8(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}
   
#---------------------------

#---------------------------
# MODEL 64
class GEN_autoEncoder_64(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_64()
        self.decoder = GEN_autoEncoder_Decoder_64()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 64), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 50), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_64(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=64, out_features=60)
        self.hidden_layer_2 = nn.Linear(in_features=60, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=52)        
        self.hidden_layer_4 = nn.Linear(in_features=52, out_features=50)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_64(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=50, out_features=52)
        self.hidden_layer_2 = nn.Linear(in_features=52, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=60)
        self.hidden_layer_4 = nn.Linear(in_features=60, out_features=64)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)       
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 64GCN
class GEN_autoEncoderGCN_64(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_64(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_64(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 64), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 50), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_64(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=64, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=52)        
        self.hidden_layer_4 = nn.Linear(in_features=52, out_features=50)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_64(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=50, out_features=52)
        self.hidden_layer_2 = nn.Linear(in_features=52, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=64)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 128  
class GEN_autoEncoder_128(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_128()
        self.decoder = GEN_autoEncoder_Decoder_128()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 128), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 80), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_128(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=128, out_features=112)
        self.hidden_layer_2 = nn.Linear(in_features=112, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=84)        
        self.hidden_layer_4 = nn.Linear(in_features=84, out_features=80)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_128(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=80, out_features=84)
        self.hidden_layer_2 = nn.Linear(in_features=84, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=112)
        self.hidden_layer_4 = nn.Linear(in_features=112, out_features=128)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)       
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 128
class GEN_autoEncoderGCN_128(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_graph_128(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_128(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 128), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 80), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_graph_128(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=128, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=84)        
        self.hidden_layer_4 = nn.Linear(in_features=84, out_features=80)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_128(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=80, out_features=84)
        self.hidden_layer_2 = nn.Linear(in_features=84, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=128)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
                
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#+++++++++++++++++++++++++++
# MODEL GCN EXO 128
class GEN_autoEncoderGCN_exo_128(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder_exo = GEN_autoEncoder_Encoder_exogenous_7()
        self.encoder_graph = GEN_autoEncoderGCN_Encoder_graph_128(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_exo_128(edge_index)
        

    def forward(self, x_graph, x_exo):
        x_latent_graph = self.encoder_graph(x_graph)
        x_latent_exo = self.encoder_exo(x_exo)
        x_latent = torch.cat((x_latent_graph["x_output"], x_latent_exo["x_output"]), dim=2)
        x_hat = self.decoder(x_latent)
        return {"x_input":x_graph, "x_input_exo":x_exo, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_graph_summary = torchinfo.summary(self.encoder_graph, input_size=[(1, 128), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        enc_exo_summary = torchinfo.summary(self.encoder_exo, input_size=[(1, 7), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 83), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder_graph": enc_graph_summary, "encoder_exo": enc_exo_summary, "decoder": dec_summary}
        return summary_dict
    
class GEN_autoEncoderGCN_Encoder_graph_128(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=128, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=84)        
        self.hidden_layer_4 = nn.Linear(in_features=84, out_features=80)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_exo_128(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=83, out_features=84)
        self.hidden_layer_2 = nn.Linear(in_features=84, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=128)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------

#---------------------------
# MODEL 256
class GEN_autoEncoder_256(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_256()
        self.decoder = GEN_autoEncoder_Decoder_256()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 256), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 80), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_256(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=256, out_features=112)
        self.hidden_layer_2 = nn.Linear(in_features=112, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=80)        
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_256(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=80,out_features=96)
        self.hidden_layer_2 = nn.Linear(in_features=96, out_features=112)
        self.hidden_layer_3 = nn.Linear(in_features=112, out_features=256)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 742
class GEN_autoEncoderGCN_742(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_742(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_742(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 742), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 400), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_742(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=742, out_features=628)
        self.hidden_layer_3 = nn.Linear(in_features=628, out_features=514)        
        self.hidden_layer_4 = nn.Linear(in_features=514, out_features=400)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_742(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=400, out_features=514)
        self.hidden_layer_2 = nn.Linear(in_features=514, out_features=628)
        self.hidden_layer_3 = nn.Linear(in_features=628, out_features=742)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 5943  
class GEN_autoEncoder_5943(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_5943()
        self.decoder = GEN_autoEncoder_Decoder_5943()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 5943), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 2048), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_5943(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=5943, out_features=11886)
        self.hidden_layer_2 = nn.Linear(in_features=11886, out_features=23772)
        self.hidden_layer_3 = nn.Linear(in_features=23772, out_features=14334)
        self.hidden_layer_4 = nn.Linear(in_features=14334, out_features=9792)
        self.hidden_layer_5 = nn.Linear(in_features=9792, out_features=4896)
        #self.hidden_layer_6 = nn.Linear(in_features=3383, out_features=2871)
        #self.hidden_layer_7 = nn.Linear(in_features=2871, out_features=2359)
        #self.hidden_layer_8 = nn.Linear(in_features=2359, out_features=2048)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=1, affine=True)
        self.batch_norm_3 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_5 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_6 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_7 = nn.Tanh()
        
        
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        self.dp_5 = nn.Dropout(p=0.2)
        self.dp_6 = nn.Dropout(p=0.2)
        self.dp_7 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        #layer_nn = F.tanh(layer_nn)
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer BN2 ===================
        layer_nn = self.batch_norm_2(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = self.act_4(layer_nn)
        layer_nn = self.dp_4(layer_nn)
        
        #== layer 05  ===================
        layer_nn = self.hidden_layer_5(layer_nn)
        ##layer_nn = self.act_5(layer_nn)
        ##layer_nn = self.dp_5(layer_nn)
        
        #== layer 06  ===================
        ##layer_nn = self.hidden_layer_6(layer_nn)
        ##layer_nn = self.act_6(layer_nn)
        ##layer_nn = self.dp_6(layer_nn)
        
        #== layer 07  ===================
        ##layer_nn = self.hidden_layer_7(layer_nn)
        ##layer_nn = self.act_7(layer_nn)
        ##layer_nn = self.dp_7(layer_nn)
        
        #== layer BN3 ===================
        ##layer_nn = self.batch_norm_3(layer_nn)
        
        #== layer 08  ===================
        ##layer_nn = self.hidden_layer_8(layer_nn)
        
        
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_5943(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=4896, out_features=9792)
        self.hidden_layer_2 = nn.Linear(in_features=9792, out_features=14334)
        self.hidden_layer_3 = nn.Linear(in_features=14334, out_features=23772)
        self.hidden_layer_4 = nn.Linear(in_features=23772, out_features=11886)
        self.hidden_layer_5 = nn.Linear(in_features=11886, out_features=5943)
        ##self.hidden_layer_6 = nn.Linear(in_features=4407, out_features=4919)
        ##self.hidden_layer_7 = nn.Linear(in_features=4919, out_features=5431)
        ##self.hidden_layer_8 = nn.Linear(in_features=5431, out_features=5943)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_5 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_6 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_7 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        self.dp_5 = nn.Dropout(p=0.2)
        self.dp_6 = nn.Dropout(p=0.2)
        self.dp_7 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        #layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        #layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        #layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = self.act_4(layer_nn)
        layer_nn = self.dp_4(layer_nn)
        
        #== layer 05  ===================        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = self.act_5(layer_nn)
        layer_nn = self.dp_5(layer_nn)
        
        #== layer 06  ===================
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = self.act_6(layer_nn)
        layer_nn = self.dp_6(layer_nn)
        
         #== layer 07  ===================
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = self.act_7(layer_nn)
        layer_nn = self.dp_7(layer_nn)
        
        #== layer 08  ===================
        layer_nn = self.hidden_layer_8(layer_nn)
        
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
 
=======
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

class AutoEncoderModels():

    def __init__(self, model, named_parameters_dict, univar_counts, folder):
        self.univar_counts = univar_counts
        self.folder = folder
        if not os.path.exists(folder):
                os.makedirs(folder)
        self.model = model
        self.named_parameters_dict = named_parameters_dict
        
    
    def get_model(self):
        if self.model_case=="fullyRectangle":
            return self.fullyRectangle()
        else:
            return None

    def drawModel(self):
        dot = make_dot(self.model,  params=self.named_parameters_dict, show_attrs=True, show_saved=True)
        dot.format = 'png'
        path_file = Path(self.folder,"model_plot.png")
        dot.render(filename=path_file)



    def fullyRectangle(self):
        model = GEN_fl()
        return model

class GEN_fl(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_2 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_3 = nn.Linear(in_features=78, out_features=78)
        self.hidden_layer_4 = nn.Linear(in_features=78, out_features=78)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        x_hat = F.tanh(layer_nn)
        return {"x_input":x, "x_latent":None, "x_output":x_hat}

class GEN_autoEncoder_78(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_78()
        self.decoder = GEN_autoEncoder_Decoder_78()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent)
        return {"x_input":x, "x_latent":x_latent, "x_output":x_hat}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_78(nn.Module):
    def __init__(self):
       
        super().__init__()
        #self.bn_1 = nn.BatchNorm1d(40)
        #20 30 38 46 62 78
        self.hidden_layer_1 = nn.Linear(in_features=78, out_features=62)
        self.hidden_layer_2 = nn.Linear(in_features=62, out_features=46)
        self.hidden_layer_3 = nn.Linear(in_features=46, out_features=38)
        self.hidden_layer_4 = nn.Linear(in_features=38, out_features=30)
        self.hidden_layer_5 = nn.Linear(in_features=30, out_features=20)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        #layer_nn = self.bn_1(layer_nn)
        return layer_nn

class GEN_autoEncoder_Decoder_78(nn.Module):

    def __init__(self):
       
        super().__init__()
        #20 30 38 46 62 78
        #self.bn_1 = nn.BatchNorm1d(40)
        self.hidden_layer_1 = nn.Linear(in_features=20, out_features=30)
        self.hidden_layer_2 = nn.Linear(in_features=30, out_features=38)
        self.hidden_layer_3 = nn.Linear(in_features=38, out_features=46)
        self.hidden_layer_4 = nn.Linear(in_features=46, out_features=62)
        self.hidden_layer_5 = nn.Linear(in_features=62, out_features=78)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        return layer_nn


class GEN_autoEncoder_3(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_3()
        self.decoder = GEN_autoEncoder_Decoder_3()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):       
        enc_summary = []#torchinfo.summary(self.encoder, (1, 1, 1, 7), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = []#torchinfo.summary(self.decoder, (1, 1, 1, 4), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_3(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=7, out_features=24)
        self.hidden_layer_2 = nn.Linear(in_features=24, out_features=18)
        self.hidden_layer_3 = nn.Linear(in_features=18, out_features=12)
        self.hidden_layer_4 = nn.Linear(in_features=12, out_features=4)
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(4, affine=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        #print(layer_nn.shape)
        x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_3(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=4, out_features=12)
        self.hidden_layer_2 = nn.Linear(in_features=12, out_features=18)
        self.hidden_layer_3 = nn.Linear(in_features=18, out_features=24)
        self.hidden_layer_4 = nn.Linear(in_features=24, out_features=7)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_4(layer_nn)
        return {"x_input":x, "x_output":x_out}


class GEN_autoEncoder_16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_16()
        self.decoder = GEN_autoEncoder_Decoder_16()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 16), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 12), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_16(nn.Module):
    def __init__(self):

        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=16, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=12)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        
        self.dp_1 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
                
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_16(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=12, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=16)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        self.dp_1 = nn.Dropout(p=0.2)        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        #layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
        

class GEN_autoEncoder_325(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_325()
        self.decoder = GEN_autoEncoder_Decoder_325()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_325(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=325, out_features=896)
        self.hidden_layer_2 = nn.Linear(in_features=896, out_features=642)
        self.hidden_layer_3 = nn.Linear(in_features=642, out_features=524)
        self.hidden_layer_4 = nn.Linear(in_features=524, out_features=448)
        self.hidden_layer_5 = nn.Linear(in_features=448, out_features=342)
        self.hidden_layer_6 = nn.Linear(in_features=342, out_features=280)
        self.hidden_layer_7 = nn.Linear(in_features=280, out_features=224)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

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
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_325(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=224, out_features=280)
        self.hidden_layer_2 = nn.Linear(in_features=280, out_features=342)
        self.hidden_layer_3 = nn.Linear(in_features=342, out_features=448)
        self.hidden_layer_4 = nn.Linear(in_features=448, out_features=524)
        self.hidden_layer_5 = nn.Linear(in_features=524, out_features=642)
        self.hidden_layer_6 = nn.Linear(in_features=642, out_features=896)
        self.hidden_layer_7 = nn.Linear(in_features=896, out_features=325)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
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
    
class GEN_autoEncoder_207(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_207()
        self.decoder = GEN_autoEncoder_Decoder_207()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_207(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=207, out_features=800)
        self.hidden_layer_2 = nn.Linear(in_features=800, out_features=642)
        self.hidden_layer_3 = nn.Linear(in_features=642, out_features=448)
        self.hidden_layer_4 = nn.Linear(in_features=448, out_features=224)
        self.hidden_layer_5 = nn.Linear(in_features=224, out_features=112)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_5(layer_nn)
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_207(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_3 = nn.Linear(in_features=112, out_features=224)
        self.hidden_layer_4 = nn.Linear(in_features=224, out_features=448)
        self.hidden_layer_5 = nn.Linear(in_features=448, out_features=642)
        self.hidden_layer_6 = nn.Linear(in_features=642, out_features=800)
        self.hidden_layer_7 = nn.Linear(in_features=800, out_features=207)

    def forward(self, x):
        layer_nn = self.hidden_layer_3(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_7(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}


class GEN_ConvAutoEncoder_7(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_ConvAutoEncoder_Encoder_7()
        self.decoder = GEN_ConvAutoEncoder_Decoder_7()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_ConvAutoEncoder_Encoder_7(nn.Module):
    # in  1, 1, 7, 32
    # out 1, 1, 1, 1
    def __init__(self):
        super().__init__()
        self.hidden_layer_1 = nn.Conv2d(in_channels= 1 , out_channels= 3 , kernel_size= 3 , stride= 2 , padding= 1 )
        self.hidden_layer_2 = nn.Conv2d(in_channels= 3 , out_channels= 4 , kernel_size= 2 , stride= 2 , padding= 2 )
        self.hidden_layer_3 = nn.Conv2d(in_channels= 4 , out_channels= 3 , kernel_size= 3 , stride= 1 , padding= (1,2) )
        self.hidden_layer_4 = nn.Conv2d(in_channels= 3 , out_channels= 2 , kernel_size= (2,3) , stride= 1 , padding= (0,1) )
        self.hidden_layer_5 = nn.Conv2d(in_channels= 2 , out_channels= 1 , kernel_size= 2 , stride= (1,2) , padding= 0 )

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

class GEN_ConvAutoEncoder_Decoder_7(nn.Module):
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


####################################


class GEN_autoEncoder_6k(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_6k()
        self.decoder = GEN_autoEncoder_Decoder_6k()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_6k(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=5943, out_features=5000)
        self.hidden_layer_2 = nn.Linear(in_features=5000, out_features=4500)
        self.hidden_layer_3 = nn.Linear(in_features=4500, out_features=4000)
        self.hidden_layer_4 = nn.Linear(in_features=4000, out_features=3000)
        self.hidden_layer_5 = nn.Linear(in_features=3000, out_features=2000)
        self.hidden_layer_6 = nn.Linear(in_features=2000, out_features=1200)
        self.hidden_layer_7 = nn.Linear(in_features=1200, out_features=750)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

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
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_6k(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=750, out_features=1200)
        self.hidden_layer_2 = nn.Linear(in_features=1200, out_features=2000)
        self.hidden_layer_3 = nn.Linear(in_features=2000, out_features=3000)
        self.hidden_layer_4 = nn.Linear(in_features=3000, out_features=4000)
        self.hidden_layer_5 = nn.Linear(in_features=4000, out_features=4500)
        self.hidden_layer_6 = nn.Linear(in_features=4500, out_features=5000)
        self.hidden_layer_7 = nn.Linear(in_features=5000, out_features=5943)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
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
    
    
##########

class GEN_autoEncoder_05k(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_05k()
        self.decoder = GEN_autoEncoder_Decoder_05k()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

class GEN_autoEncoder_Encoder_05k(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=200, out_features=400)
        self.hidden_layer_2 = n# The code appears to be a Python script that may be related to a
        # Generative Adversarial Network (GAN) using Graph Convolutional
        # Networks (GCN) with linear layers, possibly utilizing a pretrained
        # ResNet-0128 model. The script may be designed for a specific task or
        # project related to image generation or manipulation. The mention of
        # "CHENGDU" and "bt" is not clear from the code snippet provided.
        n.Linear(in_features=400, out_features=300)
        self.hidden_layer_3 = nn.Linear(in_features=300, out_features=250)
        self.hidden_layer_4 = nn.Linear(in_features=250, out_features=200)
        self.hidden_layer_5 = nn.Linear(in_features=200, out_features=150)
    
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(12, affine=True)

    def forward(self, x):
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_5(layer_nn)
        #x_out = self.batch_norm_1(layer_nn)
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_05k(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_3 = nn.Linear(in_features=150, out_features=200)
        self.hidden_layer_4 = nn.Linear(in_features=200, out_features=250)
        self.hidden_layer_5 = nn.Linear(in_features=250, out_features=300)
        self.hidden_layer_6 = nn.Linear(in_features=300, out_features=400)
        self.hidden_layer_7 = nn.Linear(in_features=400, out_features=200)

    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_3(x)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)
        x_out = self.hidden_layer_7(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}
    
    


    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_64()
        self.decoder = GEN_autoEncoder_Decoder_64()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        enc_summary = torchsummary.summary(self.encoder, (1, 1, 64), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchsummary.summary(self.decoder, (1, 1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

#---------------------------

#---------------------------
# MODEL EXOGENOUS  
class GEN_autoEncoder_Encoder_exogenous_7(nn.Module):
    def __init__(self,):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=7, out_features=5)
        self.hidden_layer_2 = nn.Linear(in_features=5, out_features=3)
        
        self.act_1 = nn.LeakyReLU(0.2)        
        self.dp_1 = nn.Dropout(p=0.2)
       
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}
#---------------------------

#---------------------------
# MODEL 32  
class GEN_autoEncoder_32(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_32()
        self.decoder = GEN_autoEncoder_Decoder_32()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 32), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 30), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_32(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=32, out_features=28)
        self.hidden_layer_2 = nn.Linear(in_features=28, out_features=22)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_32(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=22, out_features=28)
        self.hidden_layer_2 = nn.Linear(in_features=28, out_features=32)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)    
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 48
class GEN_autoEncoder_48(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_48()
        self.decoder = GEN_autoEncoder_Decoder_48()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 48), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 44), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_48(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=48, out_features=96)
        #self.hidden_layer_2 = nn.Linear(in_features=64, out_features=96)        
        #self.hidden_layer_3 = nn.Linear(in_features=96, out_features=128)
        self.hidden_layer_4 = nn.Linear(in_features=96, out_features=64)        
        self.hidden_layer_5 = nn.Linear(in_features=64, out_features=58)
        self.hidden_layer_6 = nn.Linear(in_features=58, out_features=52)
        self.hidden_layer_7 = nn.Linear(in_features=52, out_features=48)
        self.hidden_layer_8 = nn.Linear(in_features=48, out_features=44)
        #BN without any learning associated to it
        #self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)

    def forward(self, x):
        
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.batch_norm_1(layer_nn)
        
        #layer_nn = self.hidden_layer_2(layer_nn)
        #layer_nn = F.tanh(layer_nn)
        #layer_nn = self.hidden_layer_3(layer_nn)
        #layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = F.tanh(layer_nn)        
        layer_nn = self.hidden_layer_8(layer_nn)        
        
        #layer_nn = self.batch_norm_1(layer_nn)
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_48(nn.Module):
           
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=44, out_features=48)
        self.hidden_layer_2 = nn.Linear(in_features=48, out_features=52)
        self.hidden_layer_3 = nn.Linear(in_features=52, out_features=58)
        self.hidden_layer_4 = nn.Linear(in_features=58, out_features=64)
        self.hidden_layer_5 = nn.Linear(in_features=64, out_features=96)
        #self.hidden_layer_6 = nn.Linear(in_features=128, out_features=96)
        #self.hidden_layer_7 = nn.Linear(in_features=96, out_features=64)
        self.hidden_layer_8 = nn.Linear(in_features=96, out_features=48)
                
    def forward(self, x):
        #layer_nn = self.bn_1(x)
        layer_nn = self.hidden_layer_1(x)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_1(layer_nn)
        
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_2(layer_nn)
        
        
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_3(layer_nn)
        
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_4(layer_nn)
        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = F.tanh(layer_nn)
        ##layer_nn = self.dropout_5(layer_nn)
        
        ##layer_nn = self.hidden_layer_6(layer_nn)
        ##layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_6(layer_nn)
        
        ##layer_nn = self.hidden_layer_7(layer_nn)
        ##layer_nn = F.tanh(layer_nn)
        #layer_nn = self.dropout_6(layer_nn)
        
        x_out = self.hidden_layer_8(layer_nn)
        
        
        return {"x_input":x, "x_output":x_out}
   
#---------------------------

#---------------------------
# MODEL 64
class GEN_autoEncoder_64(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_64()
        self.decoder = GEN_autoEncoder_Decoder_64()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 64), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 50), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_64(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=64, out_features=60)
        self.hidden_layer_2 = nn.Linear(in_features=60, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=52)        
        self.hidden_layer_4 = nn.Linear(in_features=52, out_features=50)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_64(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=50, out_features=52)
        self.hidden_layer_2 = nn.Linear(in_features=52, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=60)
        self.hidden_layer_4 = nn.Linear(in_features=60, out_features=64)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)       
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 64GCN
class GEN_autoEncoderGCN_64(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_64(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_64(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 64), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 50), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_64(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=64, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=52)        
        self.hidden_layer_4 = nn.Linear(in_features=52, out_features=50)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_64(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=50, out_features=52)
        self.hidden_layer_2 = nn.Linear(in_features=52, out_features=56)
        self.hidden_layer_3 = nn.Linear(in_features=56, out_features=64)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 128  
class GEN_autoEncoder_128(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_128()
        self.decoder = GEN_autoEncoder_Decoder_128()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 128), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 80), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_128(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=128, out_features=112)
        self.hidden_layer_2 = nn.Linear(in_features=112, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=84)        
        self.hidden_layer_4 = nn.Linear(in_features=84, out_features=80)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_128(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=80, out_features=84)
        self.hidden_layer_2 = nn.Linear(in_features=84, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=112)
        self.hidden_layer_4 = nn.Linear(in_features=112, out_features=128)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)       
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 128
class GEN_autoEncoderGCN_128(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_graph_128(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_128(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 128), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 80), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_graph_128(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=128, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=84)        
        self.hidden_layer_4 = nn.Linear(in_features=84, out_features=80)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_128(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=80, out_features=84)
        self.hidden_layer_2 = nn.Linear(in_features=84, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=128)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
                
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#+++++++++++++++++++++++++++
# MODEL GCN EXO 128
class GEN_autoEncoderGCN_exo_128(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder_exo = GEN_autoEncoder_Encoder_exogenous_7()
        self.encoder_graph = GEN_autoEncoderGCN_Encoder_graph_128(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_exo_128(edge_index)
        

    def forward(self, x_graph, x_exo):
        x_latent_graph = self.encoder_graph(x_graph)
        x_latent_exo = self.encoder_exo(x_exo)
        x_latent = torch.cat((x_latent_graph["x_output"], x_latent_exo["x_output"]), dim=2)
        x_hat = self.decoder(x_latent)
        return {"x_input":x_graph, "x_input_exo":x_exo, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_graph_summary = torchinfo.summary(self.encoder_graph, input_size=[(1, 128), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        enc_exo_summary = torchinfo.summary(self.encoder_exo, input_size=[(1, 7), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 83), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder_graph": enc_graph_summary, "encoder_exo": enc_exo_summary, "decoder": dec_summary}
        return summary_dict
    
class GEN_autoEncoderGCN_Encoder_graph_128(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=128, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=84)        
        self.hidden_layer_4 = nn.Linear(in_features=84, out_features=80)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_exo_128(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=83, out_features=84)
        self.hidden_layer_2 = nn.Linear(in_features=84, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=128)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------

#---------------------------
# MODEL 256
class GEN_autoEncoder_256(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_256()
        self.decoder = GEN_autoEncoder_Decoder_256()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 256), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 80), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_256(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=256, out_features=112)
        self.hidden_layer_2 = nn.Linear(in_features=112, out_features=96)
        self.hidden_layer_3 = nn.Linear(in_features=96, out_features=80)        
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_256(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=80,out_features=96)
        self.hidden_layer_2 = nn.Linear(in_features=96, out_features=112)
        self.hidden_layer_3 = nn.Linear(in_features=112, out_features=256)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 742
class GEN_autoEncoderGCN_742(nn.Module):
    def __init__(self, edge_index, **kwargs):
        super().__init__()
        self.edge_index = edge_index
        self.encoder = GEN_autoEncoderGCN_Encoder_742(edge_index)
        self.decoder = GEN_autoEncoderGCN_Decoder_742(edge_index)
        

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=[(1, 742), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=[(1, 400), self.edge_index[0].shape], batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoderGCN_Encoder_742(nn.Module):
    def __init__(self, edge_index):
       
        super().__init__()
        #self.edge_index = edge_index
        
        
        self.hidden_layer_1 = gm.GCNConv(in_channels=1, out_channels=1)
        self.hidden_layer_2 = nn.Linear(in_features=742, out_features=628)
        self.hidden_layer_3 = nn.Linear(in_features=628, out_features=514)        
        self.hidden_layer_4 = nn.Linear(in_features=514, out_features=400)
        self.edge_index = edge_index
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
       
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_1(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 05  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        
        #== layer OUT ===================
        x_out = layer_nn
        
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoderGCN_Decoder_742(nn.Module):
    
    def __init__(self, edge_index):
       
        super().__init__()
        
        #self.edge_index = edge_index
        self.hidden_layer_1 = nn.Linear(in_features=400, out_features=514)
        self.hidden_layer_2 = nn.Linear(in_features=514, out_features=628)
        self.hidden_layer_3 = nn.Linear(in_features=628, out_features=742)
        self.hidden_layer_4 = gm.GCNConv(in_channels=1, out_channels=1)
        self.edge_index = edge_index
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_3 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        self.act_4 = nn.Tanh()#nn.Tanh()#nn.LeakyReLU(0.2)#
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
                
        #== layer 04  ===================
        layer_nn = layer_nn.permute(0, 2, 1)
        layer_nn = self.hidden_layer_4(layer_nn, self.edge_index)
        layer_nn = layer_nn.permute(0, 2, 1)
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
#---------------------------

#---------------------------
# MODEL 5943  
class GEN_autoEncoder_5943(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = GEN_autoEncoder_Encoder_5943()
        self.decoder = GEN_autoEncoder_Decoder_5943()

    def forward(self, x):
        x_latent = self.encoder(x)
        x_hat = self.decoder(x_latent["x_output"])
        return {"x_input":x, "x_latent":x_latent["x_output"], "x_output":x_hat["x_output"]}

    def get_decoder(self):
        return self.decoder

    def summary(self):
        
        enc_summary = torchinfo.summary(self.encoder, input_size=(1, 5943), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)
        dec_summary = torchinfo.summary(self.decoder, input_size=(1, 2048), batch_dim = 0, col_names = ("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds", "trainable"), verbose = 0)

        summary_dict = {"encoder": enc_summary, "decoder": dec_summary}
        return summary_dict

class GEN_autoEncoder_Encoder_5943(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=5943, out_features=11886)
        self.hidden_layer_2 = nn.Linear(in_features=11886, out_features=23772)
        self.hidden_layer_3 = nn.Linear(in_features=23772, out_features=14334)
        self.hidden_layer_4 = nn.Linear(in_features=14334, out_features=9792)
        self.hidden_layer_5 = nn.Linear(in_features=9792, out_features=4896)
        #self.hidden_layer_6 = nn.Linear(in_features=3383, out_features=2871)
        #self.hidden_layer_7 = nn.Linear(in_features=2871, out_features=2359)
        #self.hidden_layer_8 = nn.Linear(in_features=2359, out_features=2048)
        
        #BN without any learning associated to it
        self.batch_norm_1 = nn.BatchNorm1d(num_features=1, affine=True)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=1, affine=True)
        self.batch_norm_3 = nn.BatchNorm1d(num_features=1, affine=True)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_5 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_6 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_7 = nn.Tanh()
        
        
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        self.dp_5 = nn.Dropout(p=0.2)
        self.dp_6 = nn.Dropout(p=0.2)
        self.dp_7 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        #layer_nn = F.tanh(layer_nn)
        layer_nn = self.act_1(layer_nn)
        layer_nn = self.dp_1(layer_nn)
        
        #== layer BN1 ===================
        layer_nn = self.batch_norm_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        layer_nn = self.dp_3(layer_nn)
        
        #== layer BN2 ===================
        layer_nn = self.batch_norm_2(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = self.act_4(layer_nn)
        layer_nn = self.dp_4(layer_nn)
        
        #== layer 05  ===================
        layer_nn = self.hidden_layer_5(layer_nn)
        ##layer_nn = self.act_5(layer_nn)
        ##layer_nn = self.dp_5(layer_nn)
        
        #== layer 06  ===================
        ##layer_nn = self.hidden_layer_6(layer_nn)
        ##layer_nn = self.act_6(layer_nn)
        ##layer_nn = self.dp_6(layer_nn)
        
        #== layer 07  ===================
        ##layer_nn = self.hidden_layer_7(layer_nn)
        ##layer_nn = self.act_7(layer_nn)
        ##layer_nn = self.dp_7(layer_nn)
        
        #== layer BN3 ===================
        ##layer_nn = self.batch_norm_3(layer_nn)
        
        #== layer 08  ===================
        ##layer_nn = self.hidden_layer_8(layer_nn)
        
        
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}

class GEN_autoEncoder_Decoder_5943(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        
        self.hidden_layer_1 = nn.Linear(in_features=4896, out_features=9792)
        self.hidden_layer_2 = nn.Linear(in_features=9792, out_features=14334)
        self.hidden_layer_3 = nn.Linear(in_features=14334, out_features=23772)
        self.hidden_layer_4 = nn.Linear(in_features=23772, out_features=11886)
        self.hidden_layer_5 = nn.Linear(in_features=11886, out_features=5943)
        ##self.hidden_layer_6 = nn.Linear(in_features=4407, out_features=4919)
        ##self.hidden_layer_7 = nn.Linear(in_features=4919, out_features=5431)
        ##self.hidden_layer_8 = nn.Linear(in_features=5431, out_features=5943)
        
        self.act_1 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_2 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_3 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_4 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_5 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_6 = nn.Tanh()#nn.LeakyReLU(0.2)
        self.act_7 = nn.Tanh()#nn.LeakyReLU(0.2)
        
        self.dp_1 = nn.Dropout(p=0.2)
        self.dp_2 = nn.Dropout(p=0.2)
        self.dp_3 = nn.Dropout(p=0.2)
        self.dp_4 = nn.Dropout(p=0.2)
        self.dp_5 = nn.Dropout(p=0.2)
        self.dp_6 = nn.Dropout(p=0.2)
        self.dp_7 = nn.Dropout(p=0.2)
        
        
    def forward(self, x):
        #== layer IN  ===================
        layer_nn = x       
        
        #== layer 01  ===================
        layer_nn = self.hidden_layer_1(layer_nn)        
        layer_nn = self.act_1(layer_nn)
        #layer_nn = self.dp_1(layer_nn)
        
        #== layer 02  ===================
        layer_nn = self.hidden_layer_2(layer_nn)
        layer_nn = self.act_2(layer_nn)
        #layer_nn = self.dp_2(layer_nn)
        
        #== layer 03  ===================
        layer_nn = self.hidden_layer_3(layer_nn)
        layer_nn = self.act_3(layer_nn)
        #layer_nn = self.dp_3(layer_nn)
        
        #== layer 04  ===================
        layer_nn = self.hidden_layer_4(layer_nn)
        layer_nn = self.act_4(layer_nn)
        layer_nn = self.dp_4(layer_nn)
        
        #== layer 05  ===================        
        layer_nn = self.hidden_layer_5(layer_nn)
        layer_nn = self.act_5(layer_nn)
        layer_nn = self.dp_5(layer_nn)
        
        #== layer 06  ===================
        layer_nn = self.hidden_layer_6(layer_nn)
        layer_nn = self.act_6(layer_nn)
        layer_nn = self.dp_6(layer_nn)
        
         #== layer 07  ===================
        layer_nn = self.hidden_layer_7(layer_nn)
        layer_nn = self.act_7(layer_nn)
        layer_nn = self.dp_7(layer_nn)
        
        #== layer 08  ===================
        layer_nn = self.hidden_layer_8(layer_nn)
        
        
        #== layer OUT ===================
        x_out = layer_nn
        return {"x_input":x, "x_output":x_out}    
 
>>>>>>> 37fbe91 (old)
    
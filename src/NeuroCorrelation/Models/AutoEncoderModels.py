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

class GEN_autoEncoder_Encoder_16(nn.Module):
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=16, out_features=48)
        self.hidden_layer_2 = nn.Linear(in_features=48, out_features=64)
        self.hidden_layer_3 = nn.Linear(in_features=64, out_features=48)
        self.hidden_layer_4 = nn.Linear(in_features=48, out_features=32)
        self.hidden_layer_5 = nn.Linear(in_features=32, out_features=16)
        self.hidden_layer_6 = nn.Linear(in_features=16, out_features=14)
        self.hidden_layer_7 = nn.Linear(in_features=14, out_features=12)
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

class GEN_autoEncoder_Decoder_16(nn.Module):
    
    def __init__(self):
       
        super().__init__()
        self.hidden_layer_1 = nn.Linear(in_features=12, out_features=14)
        self.hidden_layer_2 = nn.Linear(in_features=14, out_features=16)
        self.hidden_layer_3 = nn.Linear(in_features=16, out_features=32)
        self.hidden_layer_4 = nn.Linear(in_features=32, out_features=48)
        self.hidden_layer_5 = nn.Linear(in_features=48, out_features=64)
        self.hidden_layer_6 = nn.Linear(in_features=64, out_features=48)
        self.hidden_layer_7 = nn.Linear(in_features=48, out_features=16)

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
        print("419-----------------")
        print(x.shape)
        print("********************")
        x_latent = self.encoder(x)
        print(x_latent["x_output"].shape)
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
        self.hidden_layer_2 = nn.Linear(in_features=400, out_features=300)
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
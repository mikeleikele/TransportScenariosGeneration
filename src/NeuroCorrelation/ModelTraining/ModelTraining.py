import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
import torch.optim as optim
from torch.nn import BCELoss
from src.NeuroCorrelation.DataLoaders.DataBatchGenerator import DataBatchGenerator
from src.NeuroCorrelation.Analysis.NeuroDistributions import NeuroDistributions
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import json
from matplotlib.ticker import PercentFormatter
import datetime
from torchviz import make_dot
from torch import autograd
from torch.autograd import Variable
from matplotlib.pyplot import cm
from torch.nn.utils import parameters_to_vector
from termcolor import colored, cprint 

class ModelTraining():

    def __init__(self, model, device, loss_obj, epoch, dataset, dataGenerator, path_folder, univar_count_in, univar_count_out, latent_dim, vc_mapping, input_shape, rangeData, model_type="AE", pre_trained_decoder=False, optimization=False, optimization_function=None,optimization_name=None):
        self.loss_obj = loss_obj
        self.epoch = epoch
        self.dataset = dataset
        self.dataGenerator = dataGenerator
        self.path_folder = path_folder
        self.loss_dict = dict()
        self.model_type = model_type
        self.vc_mapping = vc_mapping
        self.rangeData = rangeData
        self.device = device
        self.append_count = 150 #ogni quanti batch aggiungo val della loss
        self.n_critic = 2
        self.opt_scheduler_ae = "ReduceLROnPlateau"
        self.opt_scheduler_gen = "ReduceLROnPlateau"
        self.opt_scheduler_dis = "StepLR"
        cprint(f"PAY ATTENTION: GEN is update every {self.n_critic} batches", "magenta", end="\n")
        self.GAN_loss = "MSELoss"
        self.univar_count_in = univar_count_in 
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        
        self.path_save_model_gradients = Path(self.path_folder, self.model_type,"model_gradients", )
        if not os.path.exists(self.path_save_model_gradients):
            os.makedirs(self.path_save_model_gradients)
        self.chechWeightsUpdate = False    
        if self.chechWeightsUpdate:
            cprint(f"PAY ATTENTION: check weights update is on", "magenta", end="\n")
        if self.model_type=="AE":
            self.model = model()
            self.model.to(device=self.device)
            model_params = self.model.parameters()            
            lr_ae = 0.01
            self.optimizer = optim.Adam(params=model_params, lr=lr_ae)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
            
            if self.opt_scheduler_ae == "ReduceLROnPlateau":
                self.scheduler_ae = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, verbose=True)
            elif self.opt_scheduler_ae == "StepLR":
                self.scheduler_ae = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)

            
        elif self.model_type=="GAN":
            lr_gen = 0.01
            lr_dis = 0.01
            b1_gen = 0.5   #decay of first order momentum of gradient gen
            b1_dis = 0.5   #decay of first order momentum of gradient dis
            b2_gen = 0.999 #decay of first order momentum of gradient gen
            b2_dis = 0.999 #decay of first order momentum of gradient dis
            self.model = model
            
            if pre_trained_decoder:
                self.model_gen = model.get_generator()
            else:
                model_gen = model.get_generator()
                self.model_gen = model_gen()
            self.model_gen.to(device=self.device)
            gen_params = self.model_gen.parameters()
            
            self.optimizer_gen = optim.Adam(gen_params, lr=lr_gen, betas=(b1_gen, b2_gen))
            if self.opt_scheduler_gen == "ReduceLROnPlateau":
                self.scheduler_gen = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_gen, factor=0.1, patience=5, verbose=True)
            elif self.opt_scheduler_gen == "StepLR":
                self.scheduler_gen = optim.lr_scheduler.StepLR(self.optimizer_gen, step_size=20, gamma=0.1)


            model_dis = self.model.get_discriminator()
            
            self.model_dis = model_dis()
            self.model_dis.to(device=self.device)
            dis_params = self.model_dis.parameters()
            self.optimizer_dis = optim.Adam(dis_params, lr=lr_dis, betas=(b1_gen, b2_gen))  
            
            if self.opt_scheduler_dis == "ReduceLROnPlateau":
                self.scheduler_dis = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_dis, factor=0.1, patience=5, verbose=True)
            elif self.opt_scheduler_dis == "StepLR":
                self.scheduler_dis = optim.lr_scheduler.StepLR(self.optimizer_dis, step_size=15, gamma=0.1)
            
            
           
            '''if self.GAN_loss =="BCELoss":
                self.criterion = nn.BCELoss()
            elif self.GAN_loss =="MSELoss":
                self.criterion = nn.MSE_LOSS()'''
                
        if optimization:
            print("optimization!")   
            self.path_opt_results = Path(self.path_folder, self.model_type,"Optimizations") 
            if not os.path.exists(self.path_opt_results):
                os.makedirs(self.path_opt_results)

    def training(self, batch_size=64, noise_size=None, shuffle_data=True, plot_loss=True, model_flatten_in = True, save_model=True, load_model=False, optimization=False, optimization_function=None, optimization_name=None):
        self.dataLoaded= DataBatchGenerator(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle_data)
        if load_model:
            if optimization is not None:
                path_opt_result = Path(self.path_opt_results, f"{optimization_name}")
                if not os.path.exists(path_opt_result):
                    os.makedirs(path_opt_result)

                path_save_model = Path(path_opt_result, f"model_weights_trial_{optimization_name}.pth")
            else:
                path_save_model = Path(self.path_folder, self.model_type, "model_save", 'model_weights.pth')
            print("\tLOAD TRAINED MODEL:\t",path_save_model)
            
            self.model.load_state_dict(torch.load(path_save_model, map_location=self.device))
        else:
            print("\tTRAIN TRAINED MODEL:\t")
            self.getModeModel()
            print("--------------------------------------------------------------------------------------------------------------------------",self.model_type)
            if self.model_type == "AE":
                train_start_time = datetime.datetime.now()

                self.training_AE(plot_loss=plot_loss, batch_size=batch_size, model_flatten_in=model_flatten_in)
                train_end_time = datetime.datetime.now()
                train_time = train_end_time - train_start_time
                print("\tTIME TRAIN MODEL:\t",train_time)
                model_opt_prediction = self.getModel('all')
            elif self.model_type == "GAN":
                train_start_time = datetime.datetime.now()
                self.training_GAN(noise_size=noise_size)
                train_end_time = datetime.datetime.now()
                train_time = train_end_time - train_start_time
                print("\tTIME TRAIN MODEL:\t",train_time)
                #model_opt_prediction = self.getModel('all')
           
            if save_model:
                self.save_model()
            if optimization:
                if self.model_type == "AE":
                
                    modelPrediction = ModelPrediction(model=model_opt_prediction, device=self.device,dataset=self.dataset, vc_mapping= self.vc_mapping, univar_count_in=self.univar_count_in, univar_count_out=self.univar_count_out,latent_dim=self.latent_dim, data_range=self.rangeData,input_shape=input_shape,path_folder=path_opt_result)         
                       
                    prediction_opt = modelPrediction.compute_prediction(experiment_name=f"{optimizar_trial}", remapping_data=True)
                    opt_function_result = prediction_opt['opt_measure']
                    
                else:
                    print("OPTIMIZATION PROCESS NOT IMPLEMENTED FOR:\t",self.model_type)
            else:
                opt_function_result = 1
            return opt_function_result
            
    def save_model(self):
        path_folder_model = Path(self.path_folder, self.model_type, f"model_save_{self.model_type}")
        if not os.path.exists(path_folder_model):
            os.makedirs(path_folder_model)
        
        if self.model_type  == "AE":
            path_save_model = Path(path_folder_model, 'model_weights_AE.pth')
            torch.save(self.model.state_dict(), path_save_model)
            print("\tSAVE TRAINED MODEL AE:\t",path_save_model)
        elif self.model_type=="GAN":
            path_save_model_gen = Path(path_folder_model, f'model_weights_GAN_gen.pth')
            torch.save(self.model_gen.state_dict(), path_save_model_gen)
            print(f"\tSAVE TRAINED MODEL GAN GEN:\t",path_save_model_gen)
            
            path_save_model_dis = Path(path_folder_model, f'model_weights_GAN_dis.pth')            
            torch.save(self.model_dis.state_dict(), path_save_model_dis)
            print(f"\tSAVE TRAINED MODEL GAN DIS:\t",path_save_model_dis)
            
               
    def training_AE(self, model_flatten_in, batch_size, plot_loss=True, save_trainingTime=True):
        self.loss_dict = dict()
        self.model.train()
        epoch_train = list()
        print("\tepoch:\t",0,"/",self.epoch['AE'],"\t -")
        for epoch in range(self.epoch['AE']):
            epoch_start_time = datetime.datetime.now()
            
            dataBatches = self.dataLoaded.generate()
            loss_batch = list()
            loss_batch_partial = dict()
            for batch_num, dataBatch in enumerate(dataBatches):
                item_batch = len(dataBatch)
                if True: #item_batch == batch_size:
                    loss = torch.zeros([1])                    
                    self.optimizer.zero_grad()
                    y_hat_list = list()
                    sample_list = list()
                    for i, item in enumerate(dataBatch):
                        samplef = item['sample']
                        #noisef = item['noise']
                        sample = samplef.type(torch.float32)
                        sample_list.append(sample)
                        #noise = noisef.type(torch.float32)
                    x_in = torch.Tensor(1, item_batch, self.univar_count_in).to(device=self.device)
                    
                    
                    torch.cat(sample_list, out=x_in) 
                    
                    if model_flatten_in:
                        x_in = x_in.view(-1,self.univar_count_in)
                        x_in.unsqueeze_(1)
                                            
                        

                    
                    y_hat = self.model.forward(x=x_in)
                    
                    y_hat_list = list()


                    for i in range(item_batch):
                        item_dict = {"x_input": y_hat['x_input'][i][0], "x_latent":y_hat['x_latent'][i][0], "x_output":y_hat['x_output'][i][0]}
                        y_hat_list.append(item_dict)
                    #print(y_hat.shape)
                        
                    #y_hat_list.append(y_hat)
                    print(y_hat_list)
                    loss_dict = self.loss_obj.computate_loss(y_hat_list)

                    loss = loss_dict['loss_total']
                    if batch_num%self.append_count == 0:
                        loss_batch.append(loss.detach().cpu().numpy())
                    for loss_part in loss_dict:
                        if loss_part not in loss_batch_partial:
                            loss_batch_partial[loss_part] = list()
                        if batch_num%self.append_count == 0:
                            loss_part_value = loss_dict[loss_part].detach().cpu().numpy()
                            loss_batch_partial[loss_part].append(loss_part_value)
                    
                    loss.backward()
                    self.optimizer.step()
                
               
            self.loss_dict[epoch] = {"GLOBAL_loss": np.mean(loss_batch), "values_list": loss_batch}
            for loss_part in loss_dict:
                self.loss_dict[epoch][loss_part] = np.mean(loss_batch_partial[loss_part])
            epoch_end_time = datetime.datetime.now()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_train.append({"epoch":epoch,"time":epoch_time})
            if self.opt_scheduler_ae == "ReduceLROnPlateau":
                self.scheduler_ae.step(np.mean(loss_batch))
            elif self.opt_scheduler_ae == "StepLR":
                self.scheduler_ae.step() 
            
            
            
            self.plot_grad_flow(named_parameters = self.model.named_parameters(), epoch= f"{epoch+1}", model_section="AE")
            
            print("\tepoch:\t",epoch+1,"/",self.epoch['AE'],"\t - time tr epoch: ", epoch_time ,"\tloss: ",np.mean(loss_batch),"\tlr: ",self.optimizer.param_groups[0]['lr'])
        if plot_loss:
            self.loss_plot(self.loss_dict)
        if save_trainingTime:
            self.save_training_time(epoch_train)

    def training_GAN(self, noise_size, plot_loss=True, train4batch=True, is_WGAN=True, WGAN_coef=10, save_trainingTime=True):
        
        real_label = 1.
        fake_label = 0.
        epoch_train = list()
        self.loss_dict = dict()
        self.model_gen.train()
        self.model_dis.train()
        #self.model_gen = self.model_gen.apply(self.weights_init_uniform)
        #self.model_dis = self.model_dis.apply(self.weights_init_uniform)
        if self.chechWeightsUpdate:
            layer_gen_notrained = dict()
            for lay in range(len(list(self.model_gen.parameters()))):                
                lay_tens = list(self.model_gen.parameters())[lay].clone().detach().cpu().numpy()
                layer_gen_notrained[f"lay_{lay}"] = lay_tens
                
        print("\tepoch:\t",0,"/",self.epoch['GAN']," - ")
        for epoch in range(self.epoch['GAN']):
            first = False
            
            dataBatches = self.dataLoaded.generate()
            loss_batch = list()

            err_D_list = list()
            err_D_r_list = list()
            err_D_f_list = list()
            wgan_gp_list = list()

            err_G_list = list()
            
            loss_batch_partial = dict()
            epoch_start_time = datetime.datetime.now()
            
            for batch_num, dataBatch in enumerate(dataBatches):
                if not train4batch:
                    for i, item in enumerate(dataBatch):
                        samplef = item['sample']
                        sample = samplef.type(torch.float32)
                        
                        ########################### 
                        # 1  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                        ###########################
                        ###########################
                        ##  A Update D with real data
                        ###########################
                        self.model_dis.zero_grad()
                        self.optimizer_dis.zero_grad()     
                        label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device)      
                        output = self.model_dis(sample)['x_output'].view(-1)                    
                        item_err = self.criterion(output, label)
                        
                        self.optimizer_dis.step()
                        
                        ###########################
                        ##  B Update D with fake data
                        ###########################
                        self.optimizer_dis.zero_grad()                
                        noise = torch.randn(1, 1, noise_size[0], noise_size[1]).to(device=self.device) 
                        fake = self.model_gen(noise)['x_output'] 
                        label = torch.full((1,), fake_label, dtype=torch.float32).to(device=self.device)                    
                        output = self.model_dis(fake.detach())['x_output'].view(-1)
                        item_err = self.criterion(output, label)
                        item_err.backward()
                        self.optimizer_dis.step() 
                    
                        ###########################
                        # 2 Update G network: maximize log(D(x)) + log(1 - D(G(z)))
                        ###########################
                        self.optimizer_gen.zero_grad()
                        noise = torch.randn(1, 1, noise_size[0], noise_size[1]).to(device=self.device) 
                        fake = self.model_gen(noise)['x_output']                    
                        
                        if first:
                            print("NOISE: (epoch",epoch,")\t",noise)
                            print("")
                            print("FAKE: (epoch",epoch,")\t",fake)
                            print("================================================================================================================================================\n\n")
                            first = False
                            
                        label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device) 
                        output = self.model_dis(fake)['x_output'].view(-1)                    
                        item_err = self.criterion(output, label)
                        item_err.backward()
                        self.optimizer_gen.step() 
                #------------------
                elif train4batch:
                    ###########################
                    # 1  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    
                    ###########################
                    ##  A Update D with real data
                    ###########################
                    #self.model_dis.zero_grad()
                    self.optimizer_dis.zero_grad()
                    batch_real_err_D = torch.zeros(1).type(torch.float32).to(device=self.device) 
                    x_real = list()
                    
                    for i, item in enumerate(dataBatch):
                        samplef = item['sample']
                        sample = samplef.type(torch.float32)
                        x_real.append(sample)
                        label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device)      
                        output = self.model_dis(sample)['x_output'].view(-1)   
                        item_err = output #self.criterion(output, label)
                        batch_real_err_D += item_err
                    
                    #maximize batch_real_err_D using minus sign
                    batch_real_err_D__mean = -batch_real_err_D/len(dataBatch)
                    if batch_num%self.append_count == 0:
                        err_D_r_list.append(batch_real_err_D__mean.detach().cpu().numpy()[0])
                    #batch_real_err_D.backward()
                    #self.optimizer_dis.step()
                    
                
                    ###########################
                    ##  make noiseData
                    ###########################
                    noise_batch = list()
                    for i, item in enumerate(dataBatch):
                        noise = torch.randn(1, 1, noise_size[0], noise_size[1]).to(device=self.device) 
                        noise_batch.append(noise)
                
                    ###########################
                    ##  B Update D with fake data
                    ###########################
                    
                    batch_fake_err_D = torch.zeros(1).to(device=self.device) 
                    x_fake = list()
                    for i, item in enumerate(noise_batch):
                        fake = self.model_gen(item)['x_output'] 
                        x_fake.append(sample)
                        label = torch.full((1,), fake_label, dtype=torch.float32).to(device=self.device)                    
                        output = self.model_dis(fake.detach())['x_output'].view(-1)
                        item_err = output #self.criterion(output, label)
                        batch_fake_err_D += item_err
                    
                    #maximize batch_real_err_D using minus sign
                    batch_fake_err_D__mean = batch_fake_err_D/len(dataBatch)
                    if batch_num%self.append_count == 0:
                        err_D_f_list.append(batch_fake_err_D__mean.detach().cpu().numpy()[0])
                    
                    if is_WGAN:
                        xarr_real = torch.stack(x_real).to(device=self.device) 
                        xarr_fake = torch.stack(x_fake).to(device=self.device) 
                        batch_size = len(noise_batch)
                        
                        wgan_grad_penality = self.wasser_gradient_penalty(self.model_dis, xr=xarr_real, xf=xarr_fake, batch_size=batch_size)
                        batch_err_D = batch_real_err_D__mean + batch_fake_err_D__mean + (WGAN_coef * wgan_grad_penality)
                        if batch_num%30 == 0:
                            wgan_gp_list.append(wgan_grad_penality.detach().cpu().numpy())
                    else: 
                        batch_err_D = batch_real_err_D__mean + batch_fake_err_D__mean                    
                    
                    batch_err_D.backward()
                    self.optimizer_dis.step() 
                
                    ###########################               
                    if batch_num%self.append_count == 0:
                        err_D_list.append(batch_err_D.detach().cpu().numpy())
                    
                    ###########################
                    # 2 Update G network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    
                    if batch_num%self.n_critic == 0:
                    
                        batch_fake_err_G = torch.zeros(1).to(device=self.device) 
                        self.optimizer_gen.zero_grad()
                        self.model_gen.train()

                        for i, item in enumerate(noise_batch):
                            fake = self.model_gen(item)['x_output']                    
                            label = torch.full((1,), real_label, dtype=torch.float32).to(device=self.device) 
                            output = self.model_dis(fake)['x_output'].view(-1)                    
                            item_err = output #self.criterion(output, label)
                            batch_fake_err_G += item_err 
                        
                        
                        #maximize batch_real_err_D using minus sign
                        batch_fake_err_G__mean = -batch_fake_err_G/len(dataBatch)
                        
                        batch_err_G = batch_fake_err_G__mean
                        if batch_num%self.append_count == 0:
                            err_G_list.append(batch_fake_err_G__mean.detach().cpu().numpy()[0])
                        
                        batch_err_G.backward()
                        self.optimizer_gen.step() 
                
            
            if self.chechWeightsUpdate:
                print(len(list(self.model_gen.parameters())))
                layer_gen_trained = dict()
                for lay in range(len(list(self.model_gen.parameters()))):                
                    lay_tens = list(self.model_gen.parameters())[lay].clone().detach().cpu().numpy()
                    layer_gen_trained[f"lay_{lay}"] = lay_tens
                
                print("START EQ ---")
                for key in layer_gen_trained:
                    n_train = layer_gen_notrained[key]
                    y_train = layer_gen_trained[key]
                    ten_eq = np.array_equal(n_train,y_train)
                    if ten_eq:
                        print(f"{key} are equals")
                print("END EQ -----")
            
            
            self.plot_grad_flow(named_parameters = self.model_gen.named_parameters(), epoch= f"{epoch+1}", model_section="GAN_gen")
            self.plot_grad_flow(named_parameters = self.model_dis.named_parameters(), epoch= f"{epoch+1}", model_section="GAN_dis")
            
            self.loss_dict[epoch] = {"loss_Dis": np.mean(err_D_list), "loss_Gen": np.mean(err_G_list),"loss_Dis_real": np.mean(err_D_r_list), "loss_Dis_fake": np.mean(err_D_f_list)}
            epoch_end_time = datetime.datetime.now()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_train.append({"epoch":epoch,"time":epoch_time})
            print("\tepoch:\t",epoch,"/",self.epoch['GAN'],"\t")
            print("\t\t\t-LOSS D\tall",np.mean(err_D_list),"\tD(real)",np.mean(err_D_r_list),"\tD(fake)",np.mean(err_D_f_list),"\tG",np.mean(err_G_list))
            print("\t\t\t-LeRt D",self.optimizer_dis.param_groups[0]['lr'],"\tG",self.optimizer_gen.param_groups[0]['lr'])
            if is_WGAN:
                 print("\t\t\t-gradient penalty\t",np.mean(wgan_gp_list))
            
            
            if self.opt_scheduler_gen == "ReduceLROnPlateau":
                self.scheduler_gen.step(np.mean(err_G_list))
            elif self.opt_scheduler_gen == "StepLR":
                self.scheduler_gen.step() 
            
            if self.opt_scheduler_dis == "ReduceLROnPlateau":
                self.scheduler_dis.step(np.mean(err_D_list))
            elif self.opt_scheduler_dis == "StepLR":
                self.scheduler_dis.step() 

        if plot_loss:
            self.loss_plot(self.loss_dict)
        
        if save_trainingTime:
            self.save_training_time(epoch_train)    
        
   
    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 1.0)
            m.bias.data.fill_(0)
    
            
    def get_tensor_info(tensor):
        info = []
        for name in ['requires_grad', 'is_leaf', 'retains_grad', 'grad_fn', 'grad']:
            info.append(f'{name}({getattr(tensor, name, None)})')
        return ' '.join(info)

    def getBack(self, var_grad_fn):
        print(var_grad_fn)
        for n in var_grad_fn.next_functions:
            if n[0]:
                try:
                    tensor = getattr(n[0], 'variable')
                    print(n[0])
                    print('Tensor with grad found:', tensor)
                    print(' - gradient :', tensor.grad)
                    print()
                except AttributeError as e:
                    self.getBack(n[0])

    
    def getModel(self, selection='all', eval=False, train=False):
        if self.model_type == "AE":
            if selection=='all':
                model_selected = self.model
            if selection=='encoder':
                raise Exception("Encoder not implemented")
            if selection=='decoder':
                model_selected = self.model.get_decoder()
                if train:
                    model_selected = model_selected.train()
                elif eval:
                    model_selected = model_selected.eval()

        elif self.model_type == "GAN":
            if selection=='gen':
                if train:
                    model_selected = self.model_gen.train()
                elif eval:
                    model_selected = self.model_gen.eval()
            elif selection=='dis':
                model_selected = self.model_dis
        return model_selected

    def eval(self):
        if self.model_type == "AE":
            self.model.eval()
        elif self.model_type == "GAN":
            self.model_gen.eval() 
            self.model_dis.eval() 
            # = self.model_gen
    
    def trainMode(self):
        if self.model_type == "AE":
            self.model.train()
        elif self.model_type == "GAN":
            self.model_gen.train() 
            self.model_dis.train()

    def getModeModel(self):
        if self.model_type == "AE":
            print("AE mode training:\t", self.model.training)
        elif self.model_type == "GAN":
            print("GAN gen mode training:\t", self.model_gen.training)
            print("GAN dis mode training:\t", self.model_dis.training)
    
    
    def wasser_gradient_penalty(self, dis, xr, xf, batch_size):
        
        #  Random weight term for interpolation between real and fake samples
        alpha  = torch.rand(batch_size, 1).to(device=self.device) 
        alpha = alpha.expand_as(xr)
        
        # interpolation between real and fake samples
        interpolates = alpha  * xr + (1 - alpha) * xf
        # set it to require grad info
        
        interpolates.requires_grad_(True)
        d_interpolates = dis(interpolates)['x_output']
        
        real_grad_out = Variable(Tensor(xr.shape[0], 1).fill_(1.0).to(device=self.device), requires_grad=False)
        
        real_grad = autograd.grad(outputs=d_interpolates,
                            inputs=interpolates,
                            grad_outputs=real_grad_out,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
        real_grad = real_grad.view(real_grad.size(0), -1)    
        gradient_penalty = torch.pow(real_grad.norm(2, dim=1) - 1, 2).mean()
        
        '''
        #Compute W-div gradient penalty
        # REAL
        real_grad_out = Variable(Tensor(xr.shape[0], 1).fill_(1.0).to(device=self.device), requires_grad=False)
        real_grad = autograd.grad(
            real_validity, xr, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)



        fake_grad_out = Variable(Tensor(fake_imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake_grad = autograd.grad(
            fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
        '''
        
        return gradient_penalty

    
    
    def loss_plot(self, loss_dict):
        path_fold_lossplot = Path(self.path_folder, self.model_type, "loss_plot")
        if not os.path.exists(path_fold_lossplot):
            os.makedirs(path_fold_lossplot)
            
        color = cm.rainbow(np.linspace(0, 1, len(loss_dict[0])))

        loss_plot_dict = dict()
        i = 0
        for key in loss_dict[0]:
            if key != "values_list" and key!='loss_total':
                loss_list = list()
                for mean_val in loss_dict:
                    loss_list.append(loss_dict[mean_val][key])
                loss_plot_dict[key] = loss_list
                plt.figure(figsize=(12,8))  
                plt.plot(loss_list, color=color[i], marker='o',mfc=color[i])
                filename = Path(path_fold_lossplot,"loss_training_"+str(key)+".png")
                plt.savefig(filename)
                i += 1
        i = 0
        plt.figure(figsize=(12,8))
        for key in loss_plot_dict:  
            if key != "allLosses":
                plt.plot(loss_plot_dict[key], color=color[i], marker='o',mfc=color[i], label=str(key))
                i += 1
        plt.legend(loc="upper right")
        filename = Path(path_fold_lossplot,"loss_training_allLosses.png")
        plt.savefig(filename)
        df = pd.DataFrame(loss_plot_dict).T.reset_index()
        
    def save_training_time(self, list_training_time):
        path_fold_lossplot = Path(self.path_folder, self.model_type, "loss_plot")
        if not os.path.exists(path_fold_lossplot):
            os.makedirs(path_fold_lossplot)
        dftrainingTime = pd.DataFrame(columns=['epoch',"time"]) 
        for item in list_training_time:
            dftrainingTime = dftrainingTime.append({'epoch': item["epoch"], 'time': item["time"] }, ignore_index=True)
        time_file = Path(path_fold_lossplot ,"epochs_time.csv")
        dftrainingTime.to_csv(time_file, sep='\t')  
          
              
    def plot_grad_flow(self, named_parameters, epoch, model_section):
        
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                if p.grad is not None:
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean().cpu())
                    max_grads.append(p.grad.abs().max().cpu())
            
        plt.figure(figsize=(12,15))
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4)], 
            ['max-gradient', 'mean-gradient', 'zero-gradient'])
        
        epoch_str = f'{epoch}'.zfill(len(str(self.epoch[self.model_type])))
        path_save_gradexpl = Path(self.path_save_model_gradients,f"{model_section}_gradient_epoch_{epoch_str}.png")
        plt.savefig(path_save_gradexpl)

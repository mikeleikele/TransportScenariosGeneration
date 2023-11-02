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
        
        self.univar_count_in = univar_count_in 
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        
        self.path_save_model_gradients = Path(self.path_folder, self.model_type,"model_gradients", )
        if not os.path.exists(self.path_save_model_gradients):
            os.makedirs(self.path_save_model_gradients)
            
        if self.model_type=="AE":
            self.model = model()
            self.model.to(device=self.device)
            model_params = self.model.parameters()            
            self.optimizer = optim.Adam(params=model_params, lr=0.01)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=15, gamma=0.1)
            
            
        elif self.model_type=="GAN":
            beta1 = 0.5
            lr = 0.1

            if pre_trained_decoder:
                self.model_gen = model.get_generator()
            else:
                model_gen = model.get_generator()
                self.model_gen = model_gen()
            self.model_gen.to(device=self.device)
            self.optimizer_gen = optim.Adam(self.model_gen.parameters(), lr=lr)
            self.scheduler_gen = optim.lr_scheduler.StepLR(self.optimizer_gen, step_size=10, gamma=0.1)

            model_dis = model.get_discriminator()
            self.model_dis = model_dis()
            self.model_dis.to(device=self.device)
            self.optimizer_dis = optim.Adam(self.model_dis.parameters(), lr=lr)   
            self.scheduler_dis = optim.lr_scheduler.StepLR(self.optimizer_dis, step_size=10, gamma=0.1)
            
            self.criterion = nn.BCELoss()
        
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
            if self.model_type == "AE":
                train_start_time = datetime.datetime.now()

                self.training_AE(plot_loss=plot_loss, model_flatten_in=model_flatten_in)
                train_end_time = datetime.datetime.now()
                train_time = train_end_time - train_start_time
                print("\tTIME TRAIN MODEL:\t",train_time)
                model_for_prediction = self.getModel('all')
            elif self.model_type == "GAN":
                self.training_GAN(noise_size=noise_size)
            if save_model and self.model_type != "GAN":
                self.save_model()
            if optimization:
                if self.model_type == "AE":
                
                    modelPrediction = ModelPrediction(model=model_for_prediction, device=self.device,dataset=self.dataset, vc_mapping= self.vc_mapping, univar_count_in=self.univar_count_in, univar_count_out=self.univar_count_out,latent_dim=self.latent_dim, data_range=self.rangeData,input_shape=input_shape,path_folder=path_opt_result)         
                       
                    prediction_opt = modelPrediction.compute_prediction(experiment_name=f"{optimizar_trial}", remapping_data=True)
                    opt_function_result = prediction_opt['opt_measure']
                    
                else:
                    print("OPTIMIZATION PROCESS NOT IMPLEMENTED FOR:\t",self.model_type)
            else:
                opt_function_result = 1
            print(opt_function_result)
            return opt_function_result
            
    def save_model(self):
        path_folder_model = Path(self.path_folder, self.model_type, "model_save")
        if not os.path.exists(path_folder_model):
            os.makedirs(path_folder_model)
        path_save_model = Path(path_folder_model, 'model_weights.pth')
        print("\tSAVE TRAINED MODEL:\t",path_save_model)
        torch.save(self.model.state_dict(), path_save_model)
   
    def training_AE(self, model_flatten_in, plot_loss=True):
        self.loss_dict = dict()
        self.model.train()
        
        print("\tepoch:\t",0,"/",self.epoch['AE'],"\t -")
        for epoch in range(self.epoch['AE']):
            epoch_start_time = datetime.datetime.now()            
            dataBatches = self.dataLoaded.generate()
            loss_batch = list()
            loss_batch_partial = dict()
            for batch_num, dataBatch in enumerate(dataBatches):
                loss = torch.zeros([1])
                item_batch = len(dataBatch)
                self.optimizer.zero_grad()
                y_hat_list = list()
                sample_list = list()
                for i, item in enumerate(dataBatch):
                    samplef = item['sample']
                    #noisef = item['noise']
                    sample = samplef.float()
                    sample_list.append(sample)
                    #noise = noisef.float()
                    
                x_in = torch.Tensor(item_batch, self.univar_count_in).to(device=self.device)
                
                
                torch.cat(sample_list, out=x_in) 
                
                if model_flatten_in:
                    x_in = x_in.view(-1,self.univar_count_in)
                    
                
                
                y_hat = self.model.forward(x=x_in)
                y_hat_list = list()

                for i in range(item_batch):
                    item_dict = {"x_input": y_hat['x_input'][i], "x_latent":y_hat['x_latent'][i], "x_output":y_hat['x_output'][i]}
                    y_hat_list.append(item_dict)
                #print(y_hat.shape)
                    
                #y_hat_list.append(y_hat)
                
                loss_dict = self.loss_obj.computate_loss(y_hat_list)

                loss = loss_dict['loss_total']
                
                loss_batch.append(loss.detach().cpu().numpy())
                for loss_part in loss_dict:
                    if loss_part not in loss_batch_partial:
                        loss_batch_partial[loss_part] = list()
                    loss_part_value = loss_dict[loss_part].detach().cpu().numpy()
                    loss_batch_partial[loss_part].append(loss_part_value)
                
                loss.backward()
                self.optimizer.step()
                
            self.scheduler.step()    
            self.loss_dict[epoch] = {"mean_all": np.mean(loss_batch), "values_list": loss_batch}
            for loss_part in loss_dict:
                self.loss_dict[epoch][loss_part] = np.mean(loss_batch_partial[loss_part])
            epoch_end_time = datetime.datetime.now()
            epoch_time = epoch_end_time - epoch_start_time
            
            self.plot_grad_flow(named_parameters = self.model.named_parameters(), epoch= f"{epoch+1}")
            
            print("\tepoch:\t",epoch+1,"/",self.epoch['AE'],"\t - time tr epoch: ", epoch_time ,"\tloss: ",np.mean(loss_batch),"\tlr: ",self.optimizer.param_groups[0]['lr'])
        if plot_loss:
            self.loss_plot(self.loss_dict)
            

    def training_GAN(self, noise_size, plot_loss=True):
        
        real_label = 1.
        fake_label = 0.

        self.loss_dict = dict()
        self.model_gen.train()
        self.model_dis.train()
        print("model_gen.training\t: ",self.model_gen.training )
        print("model_dis.training\t: ",self.model_dis.training )
        print("\tepoch:\t",0,"/",self.epoch['GAN']," - ")
        for epoch in range(self.epoch['GAN']):
            
            dataBatches = self.dataLoaded.generate()
            loss_batch = list()

            err_D_list = list()
            err_D_r_list = list()
            err_D_f_list = list()

            err_G_list = list()
            
            loss_batch_partial = dict()
            for batch_num, dataBatch in enumerate(dataBatches):
                ############################
                # 1  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                
                ###########################
                ##  A Update D with real data
                ###########################
                
                self.model_dis.zero_grad()
                batch_real_err = torch.zeros(1)
                for i, item in enumerate(dataBatch):
                    samplef = item['sample']
                    sample = samplef.float()
                    label = torch.full((1,), real_label, dtype=torch.float)                    
                    
                    output = self.model_dis(sample)['x_output'].view(-1)
                    
                    item_err = self.criterion(output, label)
                    batch_real_err += item_err
                err_D_r_list.append(batch_real_err.detach().numpy()[0]/len(dataBatch))
                batch_real_err.backward()
                self.optimizer_dis.step()
                
                ###########################
                ##  make noiseData
                ###########################
                
                noise_batch = list()
                for i, item in enumerate(dataBatch):
                    noise = torch.randn(1, 1, noise_size[0], noise_size[1])
                    noise_batch.append(noise)
                ###########################
                ##  B Update D with fake data
                ###########################
                
                batch_fake_err = torch.zeros(1)

                
                for i, item in enumerate(noise_batch):
                    
                    
                    fake = self.model_gen(item)['x_output']
                    label = torch.full((1,), fake_label, dtype=torch.float)                    
                    output = self.model_dis(fake.detach())['x_output'].view(-1)
                    
                    item_err = self.criterion(output, label)
                    batch_fake_err += item_err
                
                err_D_f_list.append(batch_fake_err.detach().numpy()[0]/len(dataBatch))
                batch_fake_err.backward()
                self.optimizer_dis.step()
                
                errD = batch_real_err + batch_fake_err
                err_D_list.append(errD.detach().numpy())
                ############################
                # 2 Update G network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                
                self.model_gen.zero_grad()
                batch_fake_err = torch.zeros(1)
                for i, item in enumerate(noise_batch):
                    
                    fake = self.model_gen(item)['x_output']
                    label = torch.full((1,), real_label, dtype=torch.float)
                    output = self.model_dis(fake.detach())['x_output'].view(-1)
                    item_err = self.criterion(output, label)
                    batch_fake_err += item_err
                batch_fake_err.backward()
                errG = batch_fake_err
                err_G_list.append(errG.detach().numpy())
                self.optimizer_gen.step()
            self.scheduler_dis.step()
            self.scheduler_gen.step()

            self.loss_dict[epoch] = {"loss_Dis": np.mean(err_D_list), "loss_Gen": np.mean(err_G_list),"loss_Dis_real": np.mean(err_D_r_list), "loss_Dis_fake": np.mean(err_D_f_list)}
            print("\tepoch:\t",epoch,"/",self.epoch['GAN'],"\t - loss D:\tall",np.mean(err_D_list),"\tD(real)",np.mean(err_D_r_list),"\tD(fake)",np.mean(err_D_f_list),"\t G: ",np.mean(err_G_list),"\t\t\tlr D: ",self.optimizer_dis.param_groups[0]['lr'],"\t G: ",self.optimizer_gen.param_groups[0]['lr'])
        if plot_loss:
            self.loss_plot(self.loss_dict)

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

    def loss_plot(self, loss_dict):
        path_fold_lossplot = Path(self.path_folder, self.model_type, "loss_plot")
        if not os.path.exists(path_fold_lossplot):
            os.makedirs(path_fold_lossplot)

        loss_plot_dict = dict()
        for key in loss_dict[0]:
            if key != "values_list" and key!='loss_total':
                loss_list = list()
                for mean_val in loss_dict:
                    loss_list.append(loss_dict[mean_val][key])
                loss_plot_dict[key] = loss_list
                plt.figure(figsize=(12,8))  
                plt.plot(loss_list, color='Blue', marker='o',mfc='Blue' )
                filename = Path(path_fold_lossplot,"loss_training_"+str(key)+".png")
                plt.savefig(filename)
        
        plt.figure(figsize=(12,8))
        for key in loss_plot_dict:  
            plt.plot(loss_plot_dict[key], marker='o',mfc='Blue', label=str(key))
        plt.legend(loc="upper right")
        filename = Path(path_fold_lossplot,"loss_training_allLosses.png")
        plt.savefig(filename)
        df = pd.DataFrame(loss_plot_dict).T.reset_index()
        #df.columns = 

    #deprecato
    def compute_correlationMatrix(self):
        correlationList = list()
        correlationList_txt = list()
        for univ_id in range(self.univar_count):

            self.generated_dict[univ_id] = np.array( self.generated_dict[univ_id], dtype = float) 
            correlationList.append(self.generated_dict[univ_id])
            correlationList_txt.append(self.generated_dict[univ_id].tolist())

        corrCoeff_list_gen_Path = Path(self.path_folder, self.model_type, "corrCoeffList_generated.txt")
        with open(corrCoeff_list_gen_Path, 'w') as fp:
            json.dump(correlationList_txt, fp, sort_keys=True, indent=4)

        corrCoeff_matrix_gen = np.corrcoef(correlationList)
        corrCoeff_matrix_gen_Path = Path(self.path_folder, self.model_type, "corrCoeffMatrix_generated.csv")
        np.savetxt(corrCoeff_matrix_gen_Path, corrCoeff_matrix_gen, delimiter=",")
        corrCoeff_matrix_orig_Path = Path(self.path_folder, self.model_type, "corrCoeffMatrix_original.csv")
        corrCoeff_matrix_orig = np.loadtxt(corrCoeff_matrix_orig_Path,delimiter=",")

        sub_correlation_matrix = np.subtract(corrCoeff_matrix_gen,corrCoeff_matrix_orig)
        sub_correlation_matrix_Path = Path(self.path_folder, self.model_type, "corrCoeffMatrix_sub.csv")
        np.savetxt(sub_correlation_matrix_Path, sub_correlation_matrix, delimiter=",")
              
    def plot_grad_flow(self, named_parameters, epoch=None):
        
        
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
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
        
        epoch_str = f'{epoch}'.zfill(len(str(self.epoch)))
        path_save_gradexpl = Path(self.path_save_model_gradients,f"gradient_epoch_{epoch_str}.png")
        plt.savefig(path_save_gradexpl)
  
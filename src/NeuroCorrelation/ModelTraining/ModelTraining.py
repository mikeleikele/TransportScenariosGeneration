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
import os
import json
from matplotlib.ticker import PercentFormatter
import datetime


class ModelTraining():

    def __init__(self, model, device, loss_obj, epoch, dataset, dataGenerator, path_folder, univar_count, model_type="AE", pre_trained_decoder=False, optimization=False):
        self.loss_obj = loss_obj
        self.epoch = epoch
        self.dataset = dataset
        self.dataGenerator = dataGenerator
        self.path_folder = path_folder
        self.loss_dict = dict()
        self.model_type = model_type
        self.univar_count = univar_count 
        self.device = device
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

    def training(self, batch_size=64, noise_size=None, shuffle_data=True, plot_loss=True, model_flatten_in = True, save_model=True, load_model=False, optimizar_trial=None):
        self.dataLoaded= DataBatchGenerator(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle_data)
        if load_model:
            if optimizar_trial is not None:
                path_save_model = Path(self.path_folder,"model_save", f"model_weights_trial_{optimizar_trial}.pth")
            else:
                path_save_model = Path(self.path_folder,"model_save", 'model_weights.pth')
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

            elif self.model_type == "GAN":
                self.training_GAN(noise_size=noise_size)
            if save_model:
                self.save_model()
            opt_function_result = 1
            return opt_function_result
            
    def save_model(self):
        path_folder_model = Path(self.path_folder,"model_save")
        if not os.path.exists(path_folder_model):
            os.makedirs(path_folder_model)
        path_save_model = Path(path_folder_model, 'model_weights.pth')
        print("\tSAVE TRAINED MODEL:\t",path_save_model)
        torch.save(self.model.state_dict(), path_save_model)
        
        
        
    def training_AE(self, model_flatten_in, plot_loss=True):
        self.loss_dict = dict()
        self.model.train()
        print("\tepoch:\t",0,"/",self.epoch,"\t -")
        for epoch in range(self.epoch):
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
                    
                x_in = torch.Tensor(item_batch, self.univar_count).to(device=self.device)
                
                torch.cat(sample_list, out=x_in) 
                if model_flatten_in:
                    x_in = x_in.view(-1,self.univar_count)
    
                #print(x_in)
                #print(x_in.shape)
                #print("self.univar_count ",self.univar_count)
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
            
            print("\tepoch:\t",epoch+1,"/",self.epoch,"\t - time tr epoch: ", epoch_time ,"\tloss: ",np.mean(loss_batch),"\tlr: ",self.optimizer.param_groups[0]['lr'])
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
        print("\tepoch:\t",0,"/",self.epoch," - ")
        for epoch in range(self.epoch):
            
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
            print("\tepoch:\t",epoch,"/",self.epoch,"\t - loss D:\tall",np.mean(err_D_list),"\tD(real)",np.mean(err_D_r_list),"\tD(fake)",np.mean(err_D_f_list),"\t G: ",np.mean(err_G_list),"\t\t\tlr D: ",self.optimizer_dis.param_groups[0]['lr'],"\t G: ",self.optimizer_gen.param_groups[0]['lr'])
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
        path_fold_lossplot = Path(self.path_folder, "loss_plot")
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

    #
    def compute_correlationMatrix(self):
        correlationList = list()
        correlationList_txt = list()
        for univ_id in range(self.univar_count):

            self.generated_dict[univ_id] = np.array( self.generated_dict[univ_id], dtype = float) 
            correlationList.append(self.generated_dict[univ_id])
            correlationList_txt.append(self.generated_dict[univ_id].tolist())

        corrCoeff_list_gen_Path = Path(self.path_folder, "corrCoeffList_generated.txt")
        with open(corrCoeff_list_gen_Path, 'w') as fp:
            json.dump(correlationList_txt, fp, sort_keys=True, indent=4)

        corrCoeff_matrix_gen = np.corrcoef(correlationList)
        corrCoeff_matrix_gen_Path = Path(self.path_folder, "corrCoeffMatrix_generated.csv")
        np.savetxt(corrCoeff_matrix_gen_Path, corrCoeff_matrix_gen, delimiter=",")
        corrCoeff_matrix_orig_Path = Path(self.path_folder, "corrCoeffMatrix_original.csv")
        corrCoeff_matrix_orig = np.loadtxt(corrCoeff_matrix_orig_Path,delimiter=",")

        sub_correlation_matrix = np.subtract(corrCoeff_matrix_gen,corrCoeff_matrix_orig)
        sub_correlation_matrix_Path = Path(self.path_folder, "corrCoeffMatrix_sub.csv")
        np.savetxt(sub_correlation_matrix_Path, sub_correlation_matrix, delimiter=",")
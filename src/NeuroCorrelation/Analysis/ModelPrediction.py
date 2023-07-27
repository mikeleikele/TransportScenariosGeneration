import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
from matplotlib.ticker import PercentFormatter

class ModelPrediction():

    def __init__(self, model, device, univar_count_in, univar_count_out, latent_dim, path_folder, data_range=None, input_shape="vector"):
        self.model = model
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.path_folder = path_folder
        self.device = device
        if data_range is not None:
            self.max_val = data_range['max_val']
            self.min_val = data_range['min_val']
            
        self.inpu_data = list()
        self.late_data = list()
        self.pred_data = list()

        self.inpu_byVar = dict()
        self.pred_byVar = dict()
        self.late_byComp = dict()

        self.input_shape = input_shape
        
    
    def predict(self, input_sample, pred2numpy=True, latent=True, save=True, experiment_name=None, remapping=False):
        self.inpu_data = list()
        self.late_data = list()
        self.pred_data = list()

        for inp in input_sample:            
            with torch.no_grad():
                input_2d = [inp["sample"]]
                x_in = torch.Tensor(1, self.univar_count_in).to(device=self.device)
                torch.cat(input_2d, out=x_in) 
                if self.input_shape == "vector":
                    x_in = x_in.view(-1,self.univar_count_in)
                elif self.input_shape == "matrix":
                    x_in = x_in
                out = self.model(x_in)            
                self.inpu_data.append(out["x_input"][0])
                if "x_latent" in out:
                    self.late_data.append(out["x_latent"][0])
                self.pred_data.append(out["x_output"][0])
        self.predict_sortByUnivar(pred2numpy=pred2numpy)
        if latent:
            self.latent_sortByComponent(pred2numpy=pred2numpy)
        if save:
            self.saveData(experiment_name, latent, remapping)

    def saveData(self, experiment_name, latent, remapping=None):
        if latent:
            columns = ['x_input', 'x_latent', 'x_output']
        else: 
            columns = ['x_input', 'x_output']
        df_export = pd.DataFrame(columns=columns) 

        if remapping :
            diff_minmax = self.max_val - self.min_val
        print("latent::: ",latent)
        if latent:
            print("latent::: _0")
            for x,z,y in zip(self.inpu_data, self.late_data, self.pred_data):
                if remapping is not None:
                    x_list = [(i*diff_minmax)+self.min_val for i in x.detach().cpu().numpy()]
                    z_list = z.detach().cpu().numpy()
                    y_list = [(i*diff_minmax)+self.min_val for i in y.detach().cpu().numpy()]
                else:
                    x_list = x.detach().cpu().numpy()
                    z_list = z.detach().cpu().numpy()
                    y_list = y.detach().cpu().numpy()        
                new_row = {'x_input': x_list, 'x_latent': z_list, 'x_output': y_list}
                df_export.loc[len(df_export)] = new_row
        else: 
            print("latent::: _1")
            for x,y in zip(self.inpu_data, self.pred_data):
                if remapping is not None:
                    x_list = [(i*diff_minmax)+self.min_val for i in x.detach().cpu().numpy()]
                    y_list = [(i*diff_minmax)+self.min_val for i in y.detach().cpu().numpy()]
                else:
                    x_list = x.detach().cpu().numpy()
                    y_list = y.detach().cpu().numpy()        
                new_row = {'x_input': x_list, 'x_output': y_list}
                df_export.loc[len(df_export)] = new_row
                
        path_file = Path(self.path_folder,"prediced_instances_"+experiment_name+'.csv')
        df_export.to_csv(path_file)
        
    def predict_sortByUnivar(self, pred2numpy=True):
        self.inpu_byVar = dict()        
        self.pred_byVar = dict()

        for univ_id in range(self.univar_count_in):
            self.inpu_byVar[univ_id] = list()      
        for univ_id in range(self.univar_count_out):      
            self.pred_byVar[univ_id] = list()

        for inp,out in zip(self.inpu_data, self.pred_data):

            if self.input_shape == "vector":
                #input
                for univ_id in range(self.univar_count_in):
                    if pred2numpy:
                        self.inpu_byVar[univ_id].append(inp[univ_id].detach().cpu().numpy())
                    else:
                        self.inpu_byVar[univ_id].append(inp[univ_id])
                #output
                for univ_id in range(self.univar_count_out):
                    if pred2numpy:
                        self.pred_byVar[univ_id].append(out[univ_id].detach().cpu().numpy())    
                    else:
                        self.pred_byVar[univ_id].append(out[univ_id])
            elif self.input_shape == "matrix":
                for univ_id in range(self.univar_count_in):

                    for in_var_instance in inp[0][univ_id]:
                        if pred2numpy:
                            self.inpu_byVar[univ_id].append(in_var_instance.detach().cpu().numpy())
                        else:
                            self.inpu_byVar[univ_id].append(in_var_instance)
                
                for univ_id in range(self.univar_count_out):
                    for out_var_instance in out[0][univ_id]:
                        if pred2numpy:
                            self.pred_byVar[univ_id].append(out_var_instance.detach().numpy())    
                        else:
                            self.pred_byVar[univ_id].append(out_var_instance)
        



    def latent_sortByComponent(self, pred2numpy=True):
        self.late_byComp = dict()

        for id_comp in range(self.latent_dim):
            self.late_byComp[id_comp] = list()
        
        for lat in self.late_data:
            if self.input_shape == "vector":
                for id_comp in range(self.latent_dim):                
                    if pred2numpy:
                        self.late_byComp[id_comp].append(lat[id_comp].detach().cpu().numpy())
                    else:
                        self.late_byComp[id_comp].append(lat[id_comp])
            elif self.input_shape == "matrix":
                for id_comp in range(self.latent_dim):                
                    for lat_varValue_instance in lat[0][id_comp]:
                        if pred2numpy:
                            self.late_byComp[id_comp].append(lat_varValue_instance.detach().cpu().numpy())
                        else:
                            self.late_byComp[id_comp].append(lat_varValue_instance)

    def getPred(self):
        by_univar_dict = {"input":self.inpu_data, "latent": self.late_data, "output":self.pred_data}
        return by_univar_dict


    def getPred_byUnivar(self):
        by_univar_dict = {"input":self.inpu_byVar, "output":self.pred_byVar}
        return by_univar_dict

    def getLat_byComponent(self):
        by_comp_dict = {"latent":self.late_byComp}
        return by_comp_dict

    def getLat(self):
        comp_dict = {"latent":self.late_data}
        return comp_dict
    
    def getLatent2data(self):
        data_latent = list()
        for istance in self.late_data:
            istance_dict = {'sample': istance}
            data_latent.append(istance_dict)
        comp_dict = {"latent":data_latent}
        return comp_dict
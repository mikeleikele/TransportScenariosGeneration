from src.NeuroCorrelation.DataLoaders.DataSynteticGeneration import DataSyntheticGeneration
from src.NeuroCorrelation.DataLoaders.DataMapsLoader import DataMapsLoader

import csv
import math
from pathlib import Path
import os
import torch
from torch import Tensor
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from termcolor import cprint
from colorama import init, Style

class DataLoader:
    
    def __init__(self, mode, seed,  name_dataset, version_dataset, device, dataset_setting, epoch, univar_count, lat_dim, corrCoeff, instaces_size, path_folder, time_performance, timeweather, timeweather_settings, key_value_list, prior_channels, noise_distribution="gaussian", vc_dict=None, univ_limit=150,  time_slot=None):
        
        self.mode = mode
        self.seed = seed
        self.key_value_list = key_value_list
        self.name_dataset = name_dataset
        self.version_dataset = version_dataset
        self.vc_mapping = None
        self.path_folder = path_folder
        self.instaces_size = instaces_size
        self.device = device
        self.univar_count = univar_count
        self.lat_dim = lat_dim
        self.rangeData = None
        self.epoch = epoch
        self.dataGenerator = None
        self.dataset_setting = dataset_setting
        self.starting_sample = self.checkInDict(self.dataset_setting,"starting_sample",20)
        self.train_percentual = self.checkInDict(self.dataset_setting,"train_percentual",0.70)        
        self.train_samples = self.checkInDict(self.dataset_setting,"train_samples", 50)
        self.test_samples = self.checkInDict(self.dataset_setting,"test_samples", 500)
        self.noise_samples = self.checkInDict(self.dataset_setting,"noise_samples", 1000)
        self.corrCoeff = corrCoeff
        self.noise_distribution = noise_distribution
        self.corrCoeff['data'] = dict()
        self.summary_path = Path(self.path_folder,'summary')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.statsData = None
        self.vc_dict = vc_dict
        self.univ_limit = univ_limit
        self.pathMap = None
        self.edge_index = None
        self.timeweather = timeweather
        self.timeweather_settings = timeweather_settings
        self.prior_channels = prior_channels
        self.time_slot = time_slot
        self.time_performance = time_performance
        
    def dataset_load(self, draw_plots=True, save_summary=True, loss=None, draw_correlationCoeff=True):
        self.loss = loss
        if self.mode=="random_var" and self.name_dataset=="3var_defined":
            print("DATASET PHASE: Sample generation")
            self.dataGenerator = DataSyntheticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            
            self.dataGenerator.casualVC_init_3VC(num_of_samples = self.starting_sample, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train_data'] = self.dataGenerator.casualVC_generation(name_data="train", num_of_samples = self.train_samples, draw_plots=draw_plots)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.casualVC_generation(name_data="test", num_of_samples = self.test_samples,  draw_plots=draw_plots)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots)
            self.vc_mapping = ['X', 'Y', 'Z']
            self.pathMap = None
            self.edge_index = None
            timeweather_data = None

        if self.mode=="random_var" and self.name_dataset=="copula":
            print("DATASET PHASE: Sample copula generation")
            self.dataGenerator = DataSyntheticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            
            if self.vc_dict is None:
                self.vc_dict = {"X":{"dependence":None}, "Y":{"dependence":{"X":1.6}}, "Z":{"dependence":{"X":3}}, "W":{"dependence":None},"K":{"dependence":{"W":0.5}}, "L":{"dependence":{"W":5}}, "M":{"dependence":None}}
            self.vc_mapping = list()
            for key_vc in self.vc_dict:
                self.vc_mapping.append(key_vc)
            
            self.dataGenerator.casualVC_init_multi(num_of_samples = self.starting_sample, vc_dict=self.vc_dict, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.casualVC_generation(name_data="train", univar_count=self.univar_count, num_of_samples = self.train_samples, draw_plots=draw_plots, instaces_size=self.instaces_size)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.casualVC_generation(name_data="test", univar_count=self.univar_count, num_of_samples = self.test_samples,  draw_plots=draw_plots, instaces_size=self.instaces_size)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots, instaces_size=self.instaces_size)
            self.pathMap = None
            self.edge_index = None
            self.timeweather_count = 0
        
        
        
        if self.mode =="graph_roads" or self.mode=="fin_data":
            if self.mode =="graph_roads":
                print("DATASET PHASE: Load maps data")
            elif self.mode=="fin_data":
                print("DATASET PHASE: Load maps data")
            print("draw_plots ",draw_plots)
            
            self.dataGenerator = DataMapsLoader(torch_device=self.device, seed=self.seed, name_dataset=self.name_dataset, version_dataset=self.version_dataset, key_value_list=self.key_value_list, time_performance=self.time_performance, time_slot=self.time_slot,lat_dim=self.lat_dim, univar_count=self.univar_count, path_folder=self.path_folder, univ_limit=self.univ_limit, timeweather=self.timeweather, timeweather_settings=self.timeweather_settings, noise_distribution=self.noise_distribution)
            self.dataGenerator.mapsVC_load( train_percentual=self.train_percentual, draw_plots=draw_plots)
            
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.mapsVC_getData(name_data="train", draw_plots=draw_plots, draw_correlationCoeff=draw_correlationCoeff)
            

            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.mapsVC_getData(name_data="test",  draw_plots=draw_plots, draw_correlationCoeff=draw_correlationCoeff)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", n_channels=self.prior_channels, num_of_samples = self.noise_samples, draw_plots=draw_plots)
            self.vc_mapping = self.dataGenerator.get_vc_mapping()
            self.pathMap = self.dataGenerator.get_pathMap()
            self.edge_index = self.dataGenerator.get_edgeIndex()
            self.timeweather_count = self.dataGenerator.getTimeweatherCount()
            self.copulaData_filename = self.dataGenerator.get_copulaData_filename()
            
        
        self.rangeData = self.dataGenerator.getDataRange()
        self.statsData = self.dataGenerator.getDataStats()
        
        reduced_noise_data = self.generateNoiseReduced(method="percentile", percentile_points=10)
        
       
        
        self.export_datasplit(data=train_data, name_split="train_data", key="sample")
        self.export_datasplit(data=train_data, name_split="train_data", key="sample_timeweather") 
        self.export_datasplit(data=test_data, name_split="test_data", key="sample")
        self.export_datasplit(data=test_data, name_split="test_data", key="sample_timeweather")
         
        
        data_dict = {"train_data":train_data, "test_data":test_data, "noise_data":noise_data, "reduced_noise_data":reduced_noise_data, "edge_index":self.edge_index}
        
        if save_summary:
            self.saveDataset_setting()
        return data_dict
    
    def get_vcMapping(self):
        return self.vc_mapping
    
    def get_statsData(self):
        if self.statsData is None:
            raise Exception("rangeData not defined.")
        return self.statsData
        
    def getDataGenerator(self):
        if self.dataGenerator is None:
            raise Exception("rangeData not defined.")
        return self.dataGenerator
    
    def get_copulaData_filename(self):
        return self.copulaData_filename
    
    def getRangeData(self):
        if self.rangeData is None:
            raise Exception("rangeData not defined.")
        return self.rangeData
    
    def get_pathMap(self):
        return self.pathMap
        
    def get_edgeIndex(self):
        return self.edge_index
    
    def checkInDict(self, dict_obj, key, value_default):
        if key in dict_obj:
            if dict_obj[key] is not None:
                value = dict_obj[key]
            else:
                value = value_default
        else:
            value = value_default
        return value

    def saveDataset_setting(self):
        settings_list = []
        settings_list.append(f"dataset settings") 
        settings_list.append(f"================") 
        settings_list.append(f"mode_dataset:: {self.mode}") 
        settings_list.append(f"name_dataset:: {self.name_dataset}")
        if self.time_slot is not None:
            settings_list.append(f"time_slot:: {self.time_slot}") 
        settings_list.append(f"mode_dataset:: {self.epoch}") 
        
        cprint(Style.BRIGHT + f"| Export settings:" + Style.RESET_ALL, 'black', attrs=["bold"])
        for key in self.dataset_setting:
            print("|\t",key)
            data_summary = self.dataset_setting[key]         
            summary_str = f"{key}:: {data_summary}"
            settings_list.append(summary_str)
            
        

        if self.loss is not None:
            settings_list.append(f" ") 
            settings_list.append(f"loss settings") 
            settings_list.append(f"================") 
            for key in self.loss:
                settings_list.append(f"loss part:: {key} -") 
                loss_terms = self.loss[key].get_lossTerms()
                for item in loss_terms:
                    settings_list.append(f"\t\t:: {item} \t\tcoef:: {loss_terms[item]}") 
        
        setting_str = '\n'.join(settings_list)    
        filename = Path(self.summary_path, "summary_dataset.txt")
        with open(filename, 'w') as file:
            file.write(setting_str)
        cprint(Style.BRIGHT + f"| SETTING PHASE: Summary dataset saved in: {filename}" + Style.RESET_ALL, 'black', attrs=["bold"])
    
    
    
    
    def find_nearest_kde(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    '''        
    methods :
        'all'     : all possible combination of redux_noise_values
        'percentile' : sampling between 2 different percentile
    '''
    def generateNoiseReduced(self, method, percentile_points = 10,draw_plot = True):
        noise_reduced_path_folder_a = Path(self.path_folder,"maps_analysis_"+self.name_dataset)
        if not os.path.exists(noise_reduced_path_folder_a):
            os.makedirs(noise_reduced_path_folder_a)
        noise_red_path_folder = Path(noise_reduced_path_folder_a,"noise_reduced_data_analysis")
        if not os.path.exists(noise_red_path_folder):
            os.makedirs(noise_red_path_folder)
            
        print("\tNoiseReduced method: ",method)
        noise_redux_samples = list()
        if method=='all':
            redux_noise = list()
            redux_noise_values = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
            redux_noise_values = [-1,  -0.5,  0,  0.5,  1]
            high = self.lat_dim
            c = self.generateNoisePercentile(high, redux_noise_values)
            
            for c_item in c:
                noise_redux_samples.append({'sample': torch.Tensor(c_item).to(device=self.device), 'noise': torch.Tensor(c_item).to(device=self.device)})
            print("\tNoiseReduced samples - all: done")
        elif method== 'percentile':
            mu, sigma = 0, math.sqrt(1) # mean and standard deviation
            s = np.random.normal(mu, sigma, 1000)
            k = 0
            window = 100/(percentile_points)
            values = []
            o = 0
            while o < percentile_points:
                o += 1
                l = np.percentile(s, k)
                if  k+window>=100:
                    r = 100
                else:
                    r = np.percentile(s, k+window)
                c = np.random.uniform(l, r, self.lat_dim)
                noise_redux_samples.append({'sample': torch.Tensor(c).to(device=self.device), 'noise': torch.Tensor(c).to(device=self.device)})
                k += window
        print("\tNoiseReduced samples: done")            
        return noise_redux_samples

    def generateNoisePercentile(self, high, redux_noise_values):
        if high == 0:
            return None
        else:    
            recur_list = list()
            recur_values = self.generateNoisePercentile(high-1, redux_noise_values)
            if recur_values is None:
                for item in redux_noise_values:
                    recur_list.append([item])
                return recur_list
            else:
                recur_list = list()
                for item_list in recur_values:
                    for i in range(len(redux_noise_values)):
                        a = item_list.copy()
                        a.append(redux_noise_values[i])
                        recur_list.append(a)
                return recur_list    
  
    
    def export_key_datasplit(self, data, name_split, key='sample', key_value_name=None, remapping=True):
        """
        Estrae per ogni instance la colonna key_value_name (nelle posizioni self.key_value_list),
        converte in liste di float, applica optional remapping e salva CSV con indice e colonna
        x_input contenente la lista formattata come stringa pulita.
        """
        if key_value_name is None:
            raise ValueError("È necessario specificare key_value_name.")

        # controllo NaN (su tutto il primo sample)
        if torch.any(torch.isnan(data[0][key])):
            cprint(Style.BRIGHT + f"| Export failed: {key} - {name_split} - {key_value_name}" + Style.RESET_ALL, 'black', attrs=["bold"])
            return

        # estrai liste per ogni instance
        extracted = []
        var_idx = self.key_value_list.index(key_value_name)
        for instance in data:
            tensor_data = instance[key]                     # shape: (n_rows, n_vars) o (n_rows, D)
            np_data = tensor_data.detach().cpu().numpy()    # ndarray
            column_values = np_data[:, var_idx]             # tutti i valori per questa variabile
            # garantisco float Python
            extracted.append([float(x) for x in column_values.tolist()])

        # remapping se necessario (valori normalizzati -> original scale)
        if remapping and key_value_name in getattr(self, "rangeData", {}):
            min_val = self.rangeData[key_value_name]['min_val']
            max_val = self.rangeData[key_value_name]['max_val']
            diff = max_val - min_val
            extracted = [[float(v * diff + min_val) for v in sublist] for sublist in extracted]

        # costruisco DataFrame e salvo: voglio prima colonna indice vuota e seconda colonna "x_input"
        out_path = Path(self.path_folder, 'datasplit', key_value_name)
        out_path.mkdir(parents=True, exist_ok=True)
        file_path = out_path / f"datasplit_{name_split}_{key}_{key_value_name}.csv"

        # preparo la stringa pulita per ogni riga (numeri formattati)
        def list_to_clean_string(lst):
            return "[" + ", ".join(f"{float(v):.6f}" for v in lst) + "]"

        df_csv = pd.DataFrame({"x_input": [list_to_clean_string(r) for r in extracted]})

        # salvo con indice (index_label="" produce header ",x_input")
        df_csv.to_csv(file_path, index=True, index_label="", header=True, quoting=csv.QUOTE_MINIMAL)

        cprint(Style.BRIGHT + f"| Saved {name_split} {key_value_name} to: {file_path}" + Style.RESET_ALL, 'green', attrs=["bold"])


    def export_datasplit(self, data, name_split, key='sample'):
        """
        Wrapper che chiama export_key_datasplit per ogni key_value_name in self.key_value_list.
        """
        for key_value_name in self.key_value_list:
            self.export_key_datasplit(data=data, name_split=name_split, key=key, key_value_name=key_value_name)

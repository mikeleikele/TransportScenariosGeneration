
from src.NeuroCorrelation.Datasets.DatasetTool import DatasetTool
from src.NeuroCorrelation.Analysis.DataComparison import DataComparison

import numpy as np
import torch
# Visualization libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
from pathlib import Path
import json
import os
from copulas.multivariate import GaussianMultivariate
import math
import matplotlib.pyplot as plt
import statistics
from scipy import stats as stats
import seaborn as sns
import pandas as pd
import random
from random import shuffle
import csv
from termcolor import cprint
from colorama import init, Style

class DataMapsLoader():

    def __init__(self, torch_device, name_dataset, version_dataset, key_value_list, lat_dim, univar_count, path_folder, seed, time_performance, timeweather, timeweather_settings, name_key="ae", noise_distribution = "gaussian", univ_limit=150, time_slot=None):
        self.torch_device = torch_device
        self.key_value_list = key_value_list
        self.lat_dim = lat_dim
        self.name_key = name_key
        self.univar_count = univar_count
        self.name_dataset = name_dataset
        self.version_dataset = version_dataset
        self.seed = seed
        self.min_val = dict()
        self.max_val = dict()
        for key_value_name in self.key_value_list:
            self.min_val[key_value_name] = None
            self.max_val[key_value_name] = None

        self.mean_vc_val = dict()
        self.median_vc_val = dict()
        self.variance_vc_val = dict()

        self.univ_limit = univ_limit
        self.timeweather = timeweather
        self.time_slot = time_slot
        self.time_performance = time_performance
        self.timeweather_settings = timeweather_settings
        self.noise_distribution = noise_distribution
        
        
        datatasetTool = DatasetTool(name_dataset=self.name_dataset, version_dataset = self.version_dataset, time_slot= self.time_slot)
        datatasetdict =datatasetTool.get_dataset_settings()
        
        filename = datatasetdict["filename"]
        pathMap  = datatasetdict["pathMap"]
        edge_path  = datatasetdict["edge_path"]
        timeweather_path = datatasetdict["timeweather_path"]
        self.copula_filename = datatasetdict["copula_filename"]
        self.data_df = dict()
        
        for key_value in filename:
            self.data_df[key_value] = pd.read_csv(filename[key_value], sep=',')
        
        self.pathMap = pathMap
        if edge_path is not None:
            edgeindexNP =  np.loadtxt(edge_path, delimiter=',')
            li = list()
            for row in edgeindexNP:
                l = list()
                for item in row:
                    l.append(int(item))
                li.append(l)
            self.edge_index = torch.tensor(li, dtype=torch.int64).to(self.torch_device)
        else:
            self.edge_index = None
        self.path_folder = Path(path_folder,"maps_analysis_"+self.name_dataset)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        BROWN = '\033[38;5;208m'
        print(f"{Style.BRIGHT}{BROWN}| Loaded timeweather: {self.timeweather}{Style.RESET_ALL}")


        if self.timeweather:
            self.timeweather_df = pd.read_csv(timeweather_path, sep=',')
            if self.timeweather_settings is not None:
                self.timeweather_df = self.timeweather_df[self.timeweather_settings["column_selected"]]


        
            self.timeweather_count = len(self.timeweather_df.columns)
        else:
            self.timeweather_count = 0
            
        print(f"{Style.BRIGHT}{BROWN}| Loaded timeweather count: {self.timeweather_count}{Style.RESET_ALL}")

    def getTimeweatherCount(self):
        return self.timeweather_count
    
    def getDataRange(self, key_name=None):
        if key_name is None:
            range_dict = dict()
            for key_value_name in self.key_value_list:
                range_dict[key_value_name] = {"max_val": self.max_val[key_value_name], "min_val": self.min_val[key_value_name]}
            return range_dict
        else:
            return {"max_val": self.max_val[key_name], "min_val": self.min_val[key_name]}

    
    def getDataStats(self, key_name=None):
        if key_name is None:
            return {"mean_val": self.mean_vc_val, "median_val": self.median_vc_val, "variance_val": self.variance_vc_val}
        else:
            # Filtra le stats per la chiave specifica
            mean_filtered = {k: v for k, v in self.mean_vc_val.items() if k.startswith(key_name)}
            median_filtered = {k: v for k, v in self.median_vc_val.items() if k.startswith(key_name)}
            variance_filtered = {k: v for k, v in self.variance_vc_val.items() if k.startswith(key_name)}
            return {"mean_val": mean_filtered, "median_val": median_filtered, "variance_val": variance_filtered}
 
    def get_muR(self):
        if self.with_cov:
            return {"mu":self.mu, "r_psd": self.r_psd}
        else:
            return {"mu":None, "r_psd": None}
    
    def get_pathMap(self):
        return self.pathMap
    
    def get_copulaData_filename(self):
        return self.copula_filename
    
    def get_edgeIndex(self):
        return self.edge_index
    
    def mapsVC_load(self, train_percentual=0.70, draw_plots=True, draw_correlationCoeff=False, verbose=False):
        verbose = True

        # --- STEP 1: carica i dati come matrici numpy coerenti tra chiavi ---
        all_values_vc = {}
        shape_check = None

        for key_value_name in self.key_value_list:
            df = self.data_df[key_value_name]
            vc_matrix = []

            for _, row in df.iterrows():
                vals = [float(x) for x in row[key_value_name].strip('[]').replace(' ', '').split(',')]
                vc_matrix.append(vals)

            all_values_vc[key_value_name] = np.array(vc_matrix, dtype=float)

            if shape_check is None:
                shape_check = all_values_vc[key_value_name].shape
            elif shape_check != all_values_vc[key_value_name].shape:
                raise ValueError(f"Inconsistent shape for {key_value_name}: {all_values_vc[key_value_name].shape} != {shape_check}")

        BROWN = '\033[38;5;208m'
        print(f"{Style.BRIGHT}{BROWN}| Loaded {key_value_name}: shape {all_values_vc[key_value_name].shape}{Style.RESET_ALL}")
        
        
        # --- STEP 2: calcola statistiche globali ---
        self.min_val = {}
        self.max_val = {}
        self.mean_vc_val = {}
        self.median_vc_val = {}
        self.variance_vc_val = {}

        for key_value_name in self.key_value_list:
            vals = all_values_vc[key_value_name].flatten()
            self.min_val[key_value_name] = np.min(vals)
            self.max_val[key_value_name] = np.max(vals)
            self.mean_vc_val[key_value_name] = np.mean(vals)
            self.median_vc_val[key_value_name] = np.median(vals)
            self.variance_vc_val[key_value_name] = np.var(vals)

        # --- aggiunta richieste ---
        self.mean_val = None
        self.median_val = None

        mu = {key_name: {'train': list(), 'test': list()} for key_name in self.key_value_list}
        rho_train_dict = {key_name: list() for key_name in self.key_value_list}
        rho_test_dict = {key_name: list() for key_name in self.key_value_list}

        # --- STEP 3: prepara split train/test ---
        first_key = self.key_value_list[0]
        total_values = all_values_vc[first_key].shape[1]
        train_istance = math.floor(total_values * train_percentual)

        shuffle_indexes = [i for i in range(total_values)]
        random.Random(self.seed).shuffle(shuffle_indexes)
        
        BROWN = '\033[38;5;208m'

        print(f"{Style.BRIGHT}{BROWN}| Data stats{Style.RESET_ALL}")

        for key_name in self.key_value_list:
            line = (
                f"| {key_name:<8} → "
                f"min: {self.min_val[key_name]:<10.2f} | "
                f"max: {self.max_val[key_name]:<10.2f} | "
                f"mean: {self.mean_vc_val[key_name]:<10.2f} | "
                f"median: {self.median_vc_val[key_name]:<10.2f} | "
                f"var: {self.variance_vc_val[key_name]:<12.2f}"
            )
            print(f"{Style.BRIGHT}{BROWN}{line}{Style.RESET_ALL}")
        
        
        # --- STEP 4: costruisci train/test normalizzati per ogni chiave ---
        train_values_vc = {key_name: dict() for key_name in self.key_value_list}
        test_values_vc = {key_name: dict() for key_name in self.key_value_list}

        self.vc_mapping_list = self.data_df[self.key_value_list[0]]['ref'].values.tolist()

        for key_value_name in self.key_value_list:
            for idx_ref, _ in enumerate(self.vc_mapping_list):
                full_values = all_values_vc[key_value_name][idx_ref, :]
                norm_values = (full_values - self.min_val[key_value_name]) / (self.max_val[key_value_name] - self.min_val[key_value_name])

                train_idxs = shuffle_indexes[:train_istance]
                test_idxs = shuffle_indexes[train_istance:]

                train_values = norm_values[train_idxs]
                test_values = norm_values[test_idxs]

                train_values_vc[key_value_name][idx_ref] = {
                    'values': train_values,
                    'mean': np.mean(train_values),
                    'std': np.std(train_values)
                }

                test_values_vc[key_value_name][idx_ref] = {
                    'values': test_values,
                    'mean': np.mean(test_values),
                    'std': np.std(test_values)
                }

                mu[key_value_name]['train'] = train_values_vc[key_value_name][idx_ref]['mean']
                mu[key_value_name]['test'] = test_values_vc[key_value_name][idx_ref]['mean']
                rho_train_dict[key_value_name].append(train_values)
                rho_test_dict[key_value_name].append(test_values)

                self.train_samples = len(train_values)
                self.test_samples = len(test_values)

        # --- STEP 5: salva i dati su file ---
        for key_value_name in self.key_value_list:
            key_folder = Path(self.path_folder, key_value_name)
            os.makedirs(key_folder, exist_ok=True)

            filename_train = Path(key_folder, "samples_train.csv")
            filename_test = Path(key_folder, "samples_test.csv")

            all_values_train = [ train_values_vc[key_value_name][key_vc]['values'] for key_vc in self.vc_mapping_list]
            all_values_test = [ test_values_vc[key_value_name][key_vc]['values'] for key_vc in self.vc_mapping_list]
            df_values_train = pd.DataFrame(all_values_train)
            df_values_test = pd.DataFrame(all_values_test)

            df_values_train.to_csv(filename_train, sep='\t', index=True, header=False)
            df_values_test.to_csv(filename_test, sep='\t', index=True, header=False)

        # --- STEP 6: salva mapping e indici shuffle ---
        tw_filename_train = Path(self.path_folder, "timeweather_train.csv")
        tw_filename_test = Path(self.path_folder, "timeweather_test.csv")
        idx_filename_train = Path(self.path_folder, "indexes_train.csv")
        idx_filename_test = Path(self.path_folder, "indexes_test.csv")
        filename_vc_mapping = Path(self.path_folder, "vc_mapping.csv")

        df_vc_mapping = pd.DataFrame([str(v) for v in self.vc_mapping_list], columns=['vc_name'])
        df_vc_mapping.to_csv(filename_vc_mapping, sep='\t')

        train_idx = shuffle_indexes[:train_istance]
        test_idx = shuffle_indexes[train_istance:]

        with open(idx_filename_train, mode='w', encoding='utf-8') as file:
            file.write(','.join(map(str, train_idx)) + '\n')

        with open(idx_filename_test, mode='w', encoding='utf-8') as file:
            file.write(','.join(map(str, test_idx)) + '\n')

        # --- STEP 7: correlazioni ---
        ticks_list = np.concatenate([[''], self.vc_mapping_list])

        for key_value_name in self.key_value_list:
            rho_train = np.corrcoef(rho_train_dict[key_value_name])
            rho_test = np.corrcoef(rho_test_dict[key_value_name])

            key_folder = Path(self.path_folder, key_value_name)

            if draw_correlationCoeff:
                self.plot_correlation(rho_corr=rho_train, ticks_list=ticks_list,
                                    name_plot=f"train_{key_value_name}",
                                    key_value_name=key_value_name,
                                    path_fold=key_folder,
                                    draw_plots=draw_plots)
                

                self.plot_correlation(rho_corr=rho_test, ticks_list=ticks_list,
                                    name_plot=f"test_{key_value_name}",
                                    key_value_name=key_value_name,
                                    path_fold=key_folder,
                                    draw_plots=draw_plots)
                
        # --- STEP 8: dataframe finali ---
        self.train_data_vc = {key_name: pd.DataFrame() for key_name in self.key_value_list}
        self.test_data_vc = {key_name: pd.DataFrame() for key_name in self.key_value_list}

        for key_value_name in self.key_value_list:
            for idx_ref in train_values_vc[key_value_name]:
                self.train_data_vc[key_value_name][idx_ref] = train_values_vc[key_value_name][idx_ref]['values']
            for idx_ref in test_values_vc[key_value_name]:
                self.test_data_vc[key_value_name][idx_ref] = test_values_vc[key_value_name][idx_ref]['values']

        if draw_plots:
            for key_value_name in self.key_value_list:
                key_folder = Path(self.path_folder, key_value_name)
                os.makedirs(key_folder, exist_ok=True)
                self.comparison_plot = DataComparison(univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=None, path_folder=key_folder, name_key=self.name_key, key_value_list=self.key_value_list)

                if draw_correlationCoeff:
                    self.comparison_plot.plot_vc_analysis(self.train_data_vc[key_value_name], plot_name=f"mapsTrain_{key_value_name}")
                    self.comparison_plot.plot_vc_analysis(self.test_data_vc[key_value_name], plot_name=f"mapsTest_{key_value_name}")
                    
        
    def mapsVC_getData(self, name_data="train", draw_plots=True, instaces_size=1, draw_correlationCoeff=True):
        path_fold_Analysis = Path(self.path_folder, name_data+"_data_analysis")
        if not os.path.exists(path_fold_Analysis):
            os.makedirs(path_fold_Analysis)
        
        if name_data == "train":
            data = self.train_data_vc
            n_istances = self.train_samples
            if self.timeweather:
                tw_data = self.timeweather_df_train
                
        elif name_data == "test":
            data = self.test_data_vc
            n_istances = self.test_samples
            if self.timeweather:
                tw_data = self.timeweather_df_test
        

        dataset_couple = []
        for i in range(n_istances):
            if self.timeweather:
                dataset_couple.append({"sample": self.getSample(data, i), "sample_timeweather": self.getSample(tw_data, i)})
            else:
                dataset_couple.append({"sample": self.getSample(data, i), "sample_timeweather": torch.tensor(np.nan, dtype=torch.float)})
        
        maps_data_vc = dict()
        for id_var in range(self.univar_count):
            maps_data_vc[id_var] = list()
        
        for item in dataset_couple:
            sample = item['sample']  
            for j in range(self.univar_count): 
                value = sample[j].detach().cpu().numpy()
                maps_data_vc[j].append(value)
        
        self.comparison_datamaps = DataComparison(univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.lat_dim, key_value_list=self.key_value_list, path_folder= path_fold_Analysis, name_key=self.name_key)
        if draw_correlationCoeff:
            df_data = pd.DataFrame(maps_data_vc)
            rho = self.comparison_datamaps.correlationCoeff(df_data)
        else:
            rho = None
        return dataset_couple, rho


    def getSample(self, data, key_sample):
        channel_samples = []
    
        num_columns = len(data[self.key_value_list[0]].columns)
    
        for col_idx in range(num_columns):
            column_values = []
            for key in self.key_value_list:
                value = data[key].iloc[key_sample, col_idx]
                column_values.append(value)
            channel_samples.append(column_values)
        
        sample_ch = torch.tensor(channel_samples, dtype=torch.float32).to(self.torch_device)
        return sample_ch

        
    def getSample_synthetic(self, data, key_sample):
        sample = []
        for ed in data:    
            sample.append(ed[0][key_sample])  
        return torch.from_numpy(np.array(sample)).type(torch.float32).to(self.torch_device)
    
    def get_synthetic_noise_data(self, n_channels, name_data=None, num_of_samples=5000,  draw_plots=True, draw_correlationCoeff=False):
        path_fold_noiseAnalysis = Path(self.path_folder, name_data + "_data_analysis")
        if not os.path.exists(path_fold_noiseAnalysis):
            os.makedirs(path_fold_noiseAnalysis)
        
        cprint(Style.BRIGHT + f"| Synthetic Noise data sampling by {self.noise_distribution} distribution" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        
        # Genera valori random per ogni dimensione latente
        random_values = [self.getRandom(dim=num_of_samples, distribution=self.noise_distribution) for i in range(self.lat_dim)]
        
        dataset_couple = []
        for s_id in range(num_of_samples):
            random_sampled = []
            for lat_id in range(self.lat_dim):
                random_sampled.append(random_values[lat_id][0][s_id])
            dataset_couple.append({"sample": torch.stack(random_sampled), "noise": torch.stack(random_sampled)})

        if draw_plots:
            noise_data_vc = dict()
            noise_data_vc['noise'] = dict()
            noise_data_vc['noise']['data'] = dict()
            
            # Inizializza per ogni timestep/variabile
            for id_var in range(self.lat_dim):
                noise_data_vc['noise']['data'][id_var] = list()
            
            # Popola i dati - ora ogni valore è una lista di lunghezza n_channels
            for item in dataset_couple:
                for id_var in range(self.lat_dim):
                    # Estrai il valore per questa variabile
                    value = item['sample'][id_var].tolist()
                    
                    
                    if n_channels > 1:
                        multi_channel_value = [value + np.random.normal(0, 0.01) for _ in range(n_channels)]
                    else:
                        multi_channel_value = [value]
                    
                    noise_data_vc['noise']['data'][id_var].append(multi_channel_value)
            
            self.comparison_plot_noise = DataComparison(
                univar_count_in=self.lat_dim, 
                univar_count_out=self.lat_dim, 
                latent_dim=self.lat_dim, 
                path_folder=path_fold_noiseAnalysis, 
                key_value_list=self.key_value_list, 
                name_key=self.name_key
            )
            
            noise_data_vc['noise']['color'] = 'green'
            noise_data_vc['noise']['alpha'] = 1
            
            self.comparison_plot_noise.data_comparison_plot_nochannels(noise_data_vc, plot_name="normal_noise", mode="in", is_npArray=False)
            
            if draw_correlationCoeff:
                self.comparison_plot_noise.plot_vc_analysis(noise_data_vc['noise']['data'], plot_name=name_data, color_data="green")

        return dataset_couple

    def data2Copula(self, data_in):
        df_data = None
        for i, istance in enumerate(data_in):
            tensor_list = list()
            for var in istance['sample']:
                
                if df_data is None:
                    col = [i for i in range(len(istance['sample']))]
                    df_data = pd.DataFrame(columns=col)
                tensor_list.append(var.detach().cpu().numpy().tolist())
            df_data.loc[i] = tensor_list        
        return df_data  
    

    def casualVC_generation(self, real_data=None, toPandas=True, univar_count=None, name_data="train", num_of_samples = 5000, draw_plots=True, color_data='blue', draw_correlationCoeff=True):
        time_key_fit = name_data+"_copula_fit"
        time_key_gen = name_data+"_copula_gen"
        
        path_fold_copulagenAnalysis = Path(self.path_folder,name_data+"_copulagen_data_analysis")
        if not os.path.exists(path_fold_copulagenAnalysis):
            os.makedirs(path_fold_copulagenAnalysis)
        
        
        if real_data is None:
            real_data = self.real_data_vc
            vc_mapping = self.vc_mapping
        else:
            if toPandas:
                real_data = self.data2Copula(real_data)
            else:
                real_data = real_data
            vc_mapping =list(real_data.columns.values)
        if univar_count == None:
            univar_count = self.univar_count

        instaces_size=1
        copula = GaussianMultivariate()
        copula.fit(real_data)

        print("\t"+name_data+"\tcopula.sample : start")
        synthetic_data = copula.sample(num_of_samples)
        print("\t"+name_data+"\tcopula.sample : end")
        
        copula = GaussianMultivariate()
        print("\t"+name_data+"\tcopula.fit : start")
        self.time_performance.start_time(time_key_fit)
        copula.fit(real_data)
        self.time_performance.stop_time(time_key_fit)
        self.time_performance.compute_time(time_key_fit, fun = "sum") 
        print("\t"+name_data+"\tcopula.fit : end")

        print("\t"+name_data+"\tcopula.sample : start")
        sample_to_generate = num_of_samples * instaces_size
        self.time_performance.start_time(time_key_gen)
        synthetic_data = copula.sample(sample_to_generate)
        self.time_performance.stop_time(time_key_gen)
        self.time_performance.compute_time(time_key_gen, fun = "sum") 
        print("\t"+name_data+"\tcopula.sample : end")
        
        t_copula_fit = self.time_performance.get_time(time_key_fit, fun = "mean")
        t_copula_gen = self.time_performance.get_time(time_key_gen, fun = "mean")

        print(f"{Style.BRIGHT}\033[38;2;121;212;242m| time \tfit gaussian copula:\t{t_copula_fit}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}\033[38;2;121;212;242m| time \tgen gaussian copula:\t{t_copula_gen}{Style.RESET_ALL}")

        
        self.sample_synthetic =  [ [[]]  for i in range(univar_count)]
        for i, row in synthetic_data.iterrows():
            for id_univar in range(univar_count):
                name_var = vc_mapping[id_univar]
                value_vc = row[name_var]
                self.sample_synthetic[id_univar][0].append(value_vc)
                
        samples_origin_path = Path(self.path_folder, "samples_copula_"+name_data+".txt")
        with open(samples_origin_path, 'w+') as fp:
            json.dump(self.sample_synthetic, fp, sort_keys=True, indent=4)

        dataset_couple = []
        
        for i in range(num_of_samples):
            dataset_couple.append({"sample":self.getSample_synthetic(self.sample_synthetic, i), "noise":self.getRandom(dim=univar_count, distribution = self.noise_distribution)})
        
        

        if draw_plots:
            noise_data_vc = dict()
            for id_var in range(univar_count):
                noise_data_vc[id_var] = list()
            for item in dataset_couple:
                for id_var in range(univar_count):
                    noise_data_vc[id_var].append(item['sample'][id_var].tolist())
            self.comparison_plot_noise = DataComparison(univar_count_in=self.lat_dim, univar_count_out=self.lat_dim, latent_dim=self.lat_dim, path_folder= path_fold_copulagenAnalysis, name_key=self.name_key)

            
            if draw_correlationCoeff:
                self.comparison_plot_noise.plot_vc_analysis(noise_data_vc,plot_name=name_data, color_data=color_data)
                rho = self.comparison_plot_noise.correlationCoeff(noise_data_vc)
            else:
                rho = None
        return dataset_couple, rho
        
    def get_vc_mapping(self):
        return self.vc_mapping_list

    def getRandom(self, dim, distribution):
        if distribution == "gaussian":
            randomNoise = torch.randn(1, dim).to(self.torch_device)
        elif distribution == "uniform":
            randomNoise = torch.rand(1, dim).to(self.torch_device)
        #randomNoise = torch.randn(1, dim).uniform_(0,1).to(self.torch_device)
        return randomNoise.type(torch.float32)

    
    def plot_dataDist(self, key_value_name=None):
        # Se non specificato, usa la prima chiave
        if key_value_name is None:
            key_value_name = self.key_value_list[0]
            
        path_fold_dist = Path(self.path_folder, key_value_name, "univar_distribution_synthetic")
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
            
        for univ_id in range(len(self.mu)):
            mean_val = self.mu[univ_id]
            data_vals = self.sample_synthetic[univ_id][0]
            plt.figure(figsize=(12,8))
            plt.axvline(x=mean_val, color='b', label='mean')
            plt.hist(data_vals, density=True, bins=50)
            mean_plt_txt = f"      mean: {mean_val:.3f}"
            plt.text(mean_val, 0, s=mean_plt_txt, rotation=90)            
            filename = Path(path_fold_dist, f"dist_synthetic_{key_value_name}_{univ_id}.png")
            plt.savefig(filename)



    def plot_correlation(self, rho_corr, ticks_list, name_plot, path_fold, key_value_name=None, draw_plots=True):
        corrcoef_Path = Path(path_fold, name_plot+"_corrcoef_mapsData.csv")
        if key_value_name is None:
            key_value_name = self.key_value_list[0]
       
        path_fold_dist = Path(self.path_folder, key_value_name, "univar_distribution_synthetic")
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        
        corrcoef_Path = Path(path_fold_dist, name_plot+"_corrcoef_mapsData.csv")
        np.savetxt(corrcoef_Path, rho_corr, delimiter=",")
        if draw_plots and self.univar_count<self.univ_limit: 
            fig, ax = plt.subplots(figsize=(14,14))
            im = ax.imshow(rho_corr)
            im.set_clim(-1, 1)
            ax.grid(False)
            
            ax.set_xticklabels(ticks_list)
            ax.set_yticklabels(ticks_list)        
            
            for i in range(self.univar_count):
                for j in range(self.univar_count):
                    lbl_txt = f'{rho_corr[i, j]:.2f}'
                    ax.text(j, i, lbl_txt, ha='center', va='center',color='w')
            cbar = ax.figure.colorbar(im, ax=ax, format='% .2f')
            plt.locator_params(axis='x', nbins=len(ticks_list))
            plt.locator_params(axis='y', nbins=len(ticks_list))
            plt.setp( ax.xaxis.get_majorticklabels(), rotation=-90, ha="left")
            filename = Path(path_fold_dist, name_plot+"_corrcoef_mapsData.png")
            plt.savefig(filename)
            

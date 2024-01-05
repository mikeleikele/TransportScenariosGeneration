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

class DataMapsLoader():

    def __init__(self, torch_device, name_dataset, lat_dim, univar_count, path_folder, seed):
        self.torch_device = torch_device
        self.lat_dim = lat_dim
        self.univar_count = univar_count
        self.name_dataset = name_dataset
        self.seed = seed
        self.min_val = None
        self.max_val = None
        self.mean_vc_val = dict()
        self.median_vc_val = dict()
        self.variance_vc_val = dict()
        if self.name_dataset=="PEMS_16":
            filename = Path("data","neuroCorrelation_data","PEMSBAY_S16_START17_END19_MF.csv")
        elif self.name_dataset=="MetrLA_16":
            filename = Path("data","neuroCorrelation_data","METR_LA_S16_START7_END9.csv")
        elif self.name_dataset=="PEMS_all":
            filename = Path("data","neuroCorrelation_data","PEMSBAY_S325_START17_END19_MF.csv")
        elif self.name_dataset=="MetrLA_all":
            filename = Path("data","neuroCorrelation_data","METRLA_S207_START17_END19_MF.csv")
        elif self.name_dataset=="China_Chengdu":
            filename = Path("data","neuroCorrelation_data","Chengdu_slot_A.csv")
        
        elif self.name_dataset=="China_Chengdu_A0500":
            filename = Path("data","neuroCorrelation_data","Chengdu_slot_A_cut_200.csv")
        elif self.name_dataset=="China_Chengdu_A0016":
            filename = Path("data","neuroCorrelation_data","Chengdu_slot_A_cut_16.csv")
        elif self.name_dataset=="China_Chengdu_A0064":
            filename = Path("data","neuroCorrelation_data","Chengdu_slot_A_cut_64.csv")   
            
        self.data_df = pd.read_csv(filename, sep=',')
        
        self.path_folder = Path(path_folder,"maps_analysis_"+self.name_dataset)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        self.univar_count = len(self.data_df['ref'].values)

    def getDataRange(self):
        rangeData = {"max_val": self.max_val, "min_val":self.min_val}
        return rangeData
    
              
    def getDataStats(self):
        statsData = {"mean_val": self.mean_vc_val, "median_val":self.median_vc_val, "variance_val":self.variance_vc_val}
        return statsData
        
    def get_muR(self):
        if self.with_cov:
            return {"mu":self.mu, "r_psd": self.r_psd}
        else:
            return {"mu":None, "r_psd": None}

    def mapsVC_load(self, train_percentual=0.70, draw_plots=True, verbose=False):
        all_values_vc = dict()
        vc_mapping = list()
        
        
        train_istance = None

        for i, key_vc in enumerate(self.data_df['ref'].values):
            vc_mapping.append(key_vc)
            all_values_vc[key_vc] = dict()        
            vc_values = [float(x) for x in self.data_df['traffic_speed'][i][0:-2].strip('[]').replace('"', '').replace(' ', '').split(',')]
            all_values_vc[key_vc]['values'] = vc_values

            if self.min_val is None:
                self.min_val = min(vc_values)
            else:
                min_vc = min(vc_values)
                self.min_val = min(min_vc, self.min_val)
            
            if self.max_val is None:
                self.max_val = max(vc_values)
            else:
                max_vc = max(vc_values)
                self.max_val = max(max_vc, self.max_val)
            print("-------A---",len(vc_values))
            print("-------B---",vc_values)
            self.mean_vc_val[key_vc] = statistics.mean(vc_values)
            self.median_vc_val[key_vc] = statistics.median(vc_values)
            self.variance_vc_val[key_vc] = statistics.variance(vc_values)
            
            
            
        self.mean_val = None
        self.median_val = None
        
        mu = dict()
        mu['train'] = list()
        rho_train_list = list()

        mu['test'] = list()
        rho_test_list = list()

        train_istance = math.floor(len(vc_values) * train_percentual)
        shuffle_indexes = [i for i in range(len(vc_values))]
        random.Random(self.seed).shuffle(shuffle_indexes)
        
        if verbose:
            print("\tglobal min :\t",self.min_val)
            print("\tglobal max :\t",self.max_val)        
            print("\tvc     mean:\t",self.mean_vc_val)
            print("\tvc     median:\t",self.median_vc_val)
            print("\tvc     variance:\t",self.variance_vc_val)
        
        train_values_vc = dict()
        test_values_vc = dict()
        self.vc_mapping_list = self.data_df['ref'].values.tolist()
        
        for key_vc in self.data_df['ref'].values:

            train_values_vc[key_vc] = dict()
            train_istance_list = [all_values_vc[key_vc]['values'][i] for i in range(len(all_values_vc)) if i in shuffle_indexes[:train_istance]]            
            train_values_vc[key_vc]['values'] = (np.array(train_istance_list) - self.min_val)/(self.max_val - self.min_val)
            train_values_vc[key_vc]['mean'] = np.mean(train_values_vc[key_vc]['values'])
            train_values_vc[key_vc]['std'] = np.std(train_values_vc[key_vc]['values'])
            mu['train'] = train_values_vc[key_vc]['mean']
            rho_train_list.append(train_values_vc[key_vc]['values'])
            self.train_samples = len(train_values_vc[key_vc]['values'])

            test_values_vc[key_vc] = dict()
            test_istance_list = [all_values_vc[key_vc]['values'][i] for i in range(len(all_values_vc)) if i in shuffle_indexes[train_istance:]]
            
            test_values_vc[key_vc]['values'] = (np.array(test_istance_list) - self.min_val)/(self.max_val - self.min_val)
            test_values_vc[key_vc]['mean'] = np.mean(test_values_vc[key_vc]['values'])
            test_values_vc[key_vc]['std'] = np.std(test_values_vc[key_vc]['values'])
            mu['test'] = test_values_vc[key_vc]['mean']
            rho_test_list.append(test_values_vc[key_vc]['values'])
            self.test_samples = len(test_values_vc[key_vc]['values'])
        print("\ttrain samples: done")
        print("\ttest samples: done")
        ticks_list = np.concatenate([[''], self.data_df['ref'].values])
        rho_train = np.corrcoef(rho_train_list)
        self.plot_correlation(rho_corr=rho_train, ticks_list=ticks_list, name_plot="train", path_fold=self.path_folder, draw_plots=draw_plots)
        print("\ttrain correlation: done")
        
        rho_test = np.corrcoef(rho_test_list)
        self.plot_correlation(rho_corr=rho_test, ticks_list=ticks_list, name_plot="test", path_fold=self.path_folder, draw_plots=draw_plots)
        print("\ttest correlation: done")
        
        self.train_data_vc = pd.DataFrame()
        for key_vc in train_values_vc:
            self.train_data_vc[key_vc] = train_values_vc[key_vc]['values']
        
        self.test_data_vc = pd.DataFrame()
        for key_vc in test_values_vc:
            self.test_data_vc[key_vc] = test_values_vc[key_vc]['values']
        
        if draw_plots:         
            
            self.comparison_plot = DataComparison(univar_count_in=self.univar_count, univar_count_out=self.univar_count, dim_latent=None, path_folder= self.path_folder)
            self.comparison_plot.plot_vc_analysis(self.train_data_vc,plot_name="mapsTrain")
            print("\ttrain correlation plot: done")
            self.comparison_plot.plot_vc_analysis(self.test_data_vc,plot_name="mapsTest")
            print("\ttest correlation plot: done")
            data_plot = {"train_data":self.train_data_vc,"test_data":self.test_data_vc}
            #self.comparison_plot_syntetic.plot_vc_real2gen(data_plot, labels=["train","test"], plot_name="test_train")
            

    def mapsVC_getData(self, name_data="train",  draw_plots=True, instaces_size=1):
        path_fold_copulagenAnalysis = Path(self.path_folder,name_data+"_copulagen_data_analysis")
        if not os.path.exists(path_fold_copulagenAnalysis):
            os.makedirs(path_fold_copulagenAnalysis)
            
        if name_data=="train":
            data = self.train_data_vc
            n_istances = self.train_samples
        elif name_data=="test":
            data = self.test_data_vc
            n_istances = self.test_samples
        
        dataset_couple = []
        for i in range(n_istances):
            dataset_couple.append({"sample":self.getSample(data, i)})
        
        maps_data_vc = dict()
        for id_var in range(self.univar_count):
            maps_data_vc[id_var] = list()
        for item in dataset_couple:
            for id_var in range(self.univar_count):
                for j in range(instaces_size):
                    maps_data_vc[id_var].append(item['sample'][id_var].detach().cpu().numpy().tolist())
            
            
        self.comparison_datamaps = DataComparison(univar_count_in=self.univar_count, univar_count_out=self.univar_count, dim_latent=self.lat_dim, path_folder= path_fold_copulagenAnalysis)
        df_data = pd.DataFrame(maps_data_vc)
        rho = self.comparison_datamaps.correlationCoeff(df_data)
        
        return dataset_couple, rho

    def getSample(self, data, key_sample):
        sample = []
        for ed in data:
            sample.append(data[ed][key_sample])  
        return torch.from_numpy(np.array(sample)).float().to(self.torch_device)

    def getSample_synthetic(self, data, key_sample):
        sample = []
        for ed in data:    
            sample.append(ed[0][key_sample])  
        return torch.from_numpy(np.array(sample)).float().to(self.torch_device)
    
    def get_synthetic_noise_data(self, name_data=None, num_of_samples = 5000,  draw_plots=True):
        path_fold_noiseAnalysis = Path(self.path_folder,name_data+"_data_analysis")
        if not os.path.exists(path_fold_noiseAnalysis):
            os.makedirs(path_fold_noiseAnalysis)

        random_values = [self.getRandom(dim=num_of_samples) for i in range(self.lat_dim)] 
        dataset_couple = []
        for s_id in range(num_of_samples):
            random_sampled = []
            for lat_id in range(self.lat_dim):
                random_sampled.append(random_values[lat_id][0][s_id])
            dataset_couple.append({"sample":torch.stack(random_sampled), "noise":torch.stack(random_sampled)})

        if draw_plots:
            noise_data_vc = dict()
            for id_var in range(self.lat_dim):
                noise_data_vc[id_var] = list()
            for item in dataset_couple:
                for id_var in range(self.lat_dim):
                    noise_data_vc[id_var].append(item['sample'][id_var].tolist())
            self.comparison_plot_noise = DataComparison(univar_count_in=self.lat_dim, univar_count_out=self.lat_dim, dim_latent=self.lat_dim, path_folder= path_fold_noiseAnalysis)

            self.comparison_plot_noise.plot_vc_analysis(noise_data_vc,plot_name=name_data, color_data="green")

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
    

    def casualVC_generation(self, real_data=None, toPandas=True, univar_count=None, name_data="train", num_of_samples = 50000, draw_plots=True, color_data='blue'):
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

        
        copula = GaussianMultivariate()
        copula.fit(real_data)

        print("\t"+name_data+"\tcopula.sample : start")
        synthetic_data = copula.sample(num_of_samples)
        print("\t"+name_data+"\tcopula.sample : end")

        
        self.sample_synthetic =  [ [[]]  for i in range(univar_count)]
        for i, row in synthetic_data.iterrows():
            for id_univar in range(univar_count):
                name_var = vc_mapping[id_univar]
                value_vc = row[name_var]
                self.sample_synthetic[id_univar][0].append(value_vc)
                
        samples_origin_path = Path(self.path_folder, "samples_copula_"+name_data+".txt")
        with open(samples_origin_path, 'w') as fp:
            json.dump(self.sample_synthetic, fp, sort_keys=True, indent=4)

        dataset_couple = []
        for i in range(num_of_samples):
            dataset_couple.append({"sample":self.getSample_synthetic(self.sample_synthetic, i), "noise":self.getRandom(dim=univar_count)})
        
        

        if draw_plots:
            noise_data_vc = dict()
            for id_var in range(univar_count):
                noise_data_vc[id_var] = list()
            for item in dataset_couple:
                for id_var in range(univar_count):
                    noise_data_vc[id_var].append(item['sample'][id_var].tolist())
            self.comparison_plot_noise = DataComparison(univar_count_in=self.lat_dim, univar_count_out=self.lat_dim, dim_latent=self.lat_dim, path_folder= path_fold_copulagenAnalysis)

            self.comparison_plot_noise.plot_vc_analysis(noise_data_vc,plot_name=name_data, color_data=color_data)
            rho = self.comparison_plot_noise.correlationCoeff(noise_data_vc)
        return dataset_couple, rho
        
    def get_vc_mapping(self):
        return self.vc_mapping_list

    def getRandom(self, dim):
        randomNoise =  torch.randn(1, dim).to(self.torch_device)
        #torch.randn(dim).uniform_(0,1).to(self.torch_device)
        return randomNoise.float()

    
    def plot_dataDist(self):
        path_fold_dist = Path(self.path_folder,"univar_distribution_synthetic")
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        for univ_id in range(len(self.mu)):
            mean_val = self.mu[univ_id]
            data_vals = self.sample_synthetic[univ_id][0]
            plt.figure(figsize=(12,8))
            plt.axvline(x = mean_val, color = 'b', label = 'mean')
            plt.hist(data_vals, density=True, bins=50)
            mean_plt_txt = f"      mean: {mean_val:.3f}"
            plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90)            
            filename = Path(path_fold_dist,"dist_synthetic_"+str(univ_id)+".png")
            plt.savefig(filename)

    def plot_correlation(self, rho_corr, ticks_list, name_plot, path_fold, draw_plots=True):
        corrcoef_Path = Path(path_fold, name_plot+"_corrcoef_mapsData.csv")
        np.savetxt(corrcoef_Path, rho_corr, delimiter=",")
        if draw_plots: 
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
            filename = Path(path_fold, name_plot+"_corrcoef_mapsData.png")
            plt.savefig(filename)


    


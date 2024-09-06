from src.NeuroCorrelation.Models.CHENGDU_models.CHENGDU_settings import CHENGDU_settings
from src.NeuroCorrelation.Models.PEMS_METR_models.PEMS_METR_settings import PEMS_METR_settings
from src.NeuroCorrelation.Models.autoEncoderModels_CHENGDU.CHENGDU_zone import *
from src.NeuroCorrelation.Models.ESG_models.ESG_models import *
from src.NeuroCorrelation.DataLoaders.DataSynteticGeneration import DataSyntheticGeneration
from src.NeuroCorrelation.Models.AutoEncoderModels import *
from src.NeuroCorrelation.ModelTraining.LossFunctions import LossFunction
from src.NeuroCorrelation.ModelTraining.ModelTraining import ModelTraining
from src.NeuroCorrelation.Analysis.DataComparison import DataComparison, DataComparison_Advanced, CorrelationComparison
from src.NeuroCorrelation.Analysis.DataStatistics import DataStatistics
from src.NeuroCorrelation.Analysis.ScenariosMap import ScenariosMap
from src.NeuroCorrelation.DataLoaders.DataMapsLoader import DataMapsLoader
from src.NeuroCorrelation.Optimization.Optimization import Optimization
from src.NeuroCorrelation.Models.NetworkDetails import NetworkDetails
from src.NeuroCorrelation.DataLoaders.DataLoader import DataLoader
from src.NeuroCorrelation.ModelPrediction.ModelPrediction import ModelPrediction
from src.NeuroCorrelation.ModelPrediction.PerformePrediction import PerformePrediction
from src.NeuroCorrelation.Analysis.TimeAnalysis import TimeAnalysis
import copy
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics
from pathlib import Path
import copy
import os


class NeuralCore():

    def __init__(self, device, path_folder, epoch, case, model_case, dataset_setting, univar_count, lat_dim, instaces_size, input_shape, do_optimization, opt_settings, seed=0, run_mode="all", ):
        device = "cpu"#("cuda:0" if (torch.cuda.is_available()) else "cpu")
        #device = "cpu"
        
        self.seed_torch = seed
        self.seed_data = seed
        self.seed_noise = seed
        print("SETTING PHASE: Seed ")
        print("seed torch:\t",self.seed_torch)
        print("seed data:\t",self.seed_data)
        print("seed noise:\t",self.seed_noise)
        torch.manual_seed(self.seed_torch)

        self.device = device
        print("SETTING PHASE: Device selection")
        print("\tdevice:\t",self.device)
        
        self.univar_count = univar_count        
        self.lat_dim = lat_dim
        self.epoch = epoch
        self.dataset_setting = dataset_setting
        self.batch_size = dataset_setting['batch_size']
        self.path_folder = Path(path_folder)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        self.instaces_size = instaces_size
        self.input_shape = input_shape
        self.model_trained = None
        self.model = dict()
        
        self.loss_obj = dict()
        self.instaces_size_noise = (self.instaces_size, self.lat_dim)
        self.corrCoeff = dict()
        
        
        self.summary_path = Path(self.path_folder,'summary')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.time_performance = TimeAnalysis(folder_path=self.summary_path)
        
        if run_mode=="fast":   
            self.performace_cases = {"AE":['train'],
                                    "GAN":['noise_gaussian', 'noise_gaussian_reduced'],
                                    "WGAN":['noise_gaussian', 'noise_gaussian_reduced'],
            }
            self.draw_plot = True
            self.draw_correlationCoeff = False
            self.draw_scenarios = False
        elif run_mode=="train_only":
            self.performace_cases = {"AE":[],
                                     "GAN":[],
                                     "WGAN":[]}    
            self.draw_plot = True
            self.draw_correlationCoeff = False
            self.draw_scenarios = False
        elif run_mode=="all":
            self.performace_cases = {"AE":['train', 'test', 'noise_gaussian', 'noise_gaussian_reduced', 'noise_copula'],
                                     "GAN":['noise_gaussian', 'noise_gaussian_reduced'],
                                     "WGAN":['noise_gaussian', 'noise_gaussian_reduced']}
            self.draw_plot = True
            self.draw_correlationCoeff = True
            self.draw_scenarios = True
        elif run_mode=="ALL_nocorr":
            self.performace_cases = {"AE":['train', 'test', 'noise_gaussian', 'noise_gaussian_reduced', 'noise_copula'],
                                     "GAN":['noise_gaussian', 'noise_gaussian_reduced'],
                                     "WGAN":['noise_gaussian', 'noise_gaussian_reduced']}
            self.draw_plot = True
            self.draw_correlationCoeff = False
            self.draw_scenarios = True
        elif run_mode=="ALL_nocorr_noscen":
            self.performace_cases = {"AE":['train', 'test', 'noise_gaussian', 'noise_gaussian_reduced', 'noise_copula'],
                                     "GAN":['noise_gaussian', 'noise_gaussian_reduced'],
                                     "WGAN":['noise_gaussian', 'noise_gaussian_reduced']}
            self.draw_plot = True
            self.draw_correlationCoeff = False
            self.draw_scenarios = False
            
        print("SETTING PHASE: Model creation")
        print("\tmodel_case:\t",model_case)
        self.model_case = model_case
        self.case = case
        
        if self.case == "PEMS_METR" or self.case == "CHENGDU":
            if self.case == "PEMS_METR":
                self.case_setting = PEMS_METR_settings(model_case=self.model_case, device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, dataset_setting=self.dataset_setting, epoch=self.epoch, path_folder=self.path_folder, corrCoeff=self.corrCoeff, instaces_size=self.instaces_size)
            elif self.case == "CHENGDU":
                self.case_setting = CHENGDU_settings(model_case=self.model_case, device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, dataset_setting=self.dataset_setting, epoch=self.epoch, path_folder=self.path_folder, corrCoeff=self.corrCoeff, instaces_size=self.instaces_size)
            self.trainingMode = self.case_setting.get_trainingMode()
            self.path_folder_nets = self.case_setting.get_folder_nets()
            dataloader = self.case_setting.get_DataLoader(seed_data=self.seed_data)
            self.graph_topology = self.case_setting.get_graph_topology()
            self.loss_obj = self.case_setting.get_LossFunction()
       
        
        # TO REFACTORY
        
        '''
        # ESG - Environmental Social Governance project
        elif self.model_case=="ESG__GAN_linear_pretrained_35":
            self.graph_topology = False
            dataloader = DataLoader(mode="fin_data", seed=self.seed_data, name_dataset="ESG_35", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.05, "VARIANCE_LOSS":0.05, "SPEARMAN_CORRELATION_LOSS":0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        '''
        
        '''
        if self.model_case=="GAN_linear_pretrained_0016_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0016", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0., "MEDIAN_LOSS_batch":0.0005, "VARIANCE_LOSS":0.5, "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
        elif self.model_case=="GAN_linear_pretrained_0032_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0032", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.0005, "VARIANCE_LOSS":0.05, "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)        
        elif self.model_case=="GAN_linear_pretrained_0064_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0064", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.00005, "VARIANCE_LOSS":0.0005, "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="GAN_GCN_linear_pretrained_0064_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0064", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.00005, "VARIANCE_LOSS":0.0005, "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
        elif self.model_case=="GAN_linear_pretrained_0128_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0128", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":5e-05, "VARIANCE_LOSS":0.0005, "SPEARMAN_CORRELATION_LOSS":0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
        elif self.model_case=="GAN_linear_pretrained_0256_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0256", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":5e-05, "VARIANCE_LOSS":0.0005, "SPEARMAN_CORRELATION_LOSS":0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
        
        elif self.model_case=="GAN_linear_pretrained_5943_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_5943", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
        elif self.model_case=="GAN_GCN_linear_pretrained_RN0128_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_RN_A0128", timeweather=True, device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":2, "MEDIAN_LOSS_batch":5e-03, "VARIANCE_LOSS":0.005, "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
        elif self.model_case=="GAN_GCN_linear_pretrained_RN0742_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_RN_A0742", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.5, "VARIANCE_LOSS":0.1, "SPEARMAN_CORRELATION_LOSS":0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

        elif self.model_case=="GAN_GCN_linear_pretrained_URB_ZONE0_CHENGDU_bt":
            self.graph_topology = True
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_URB_zone0", time_slot="A", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction( {"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":5e-1, "VARIANCE_LOSS":5e-1, "SPEARMAN_CORRELATION_LOSS":1e-3}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
            
        elif self.model_case=="GAN_GCN_linear_pretrained_URB_ZONE1_CHENGDU_bt":
            self.graph_topology = True
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_URB_zone1", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.05, "VARIANCE_LOSS":0.05, "SPEARMAN_CORRELATION_LOSS":0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="GAN_GCN_linear_pretrained_URB_ZONE1-2_CHENGDU_bt":
            self.graph_topology = True
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_URB_zone1-2", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":5e-1, "VARIANCE_LOSS":0.5, "SPEARMAN_CORRELATION_LOSS":0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        
        elif self.model_case=="GAN_linear_pretrained_16_CHENGDU_bt":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0016", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.05}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)      
        elif self.model_case=="AE_conv_vc_copula":
            dataloader = DataLoader(mode="random_var", seed=self.seed_data, name_dataset="copula", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.05, "SPEARMAN_CORRELATION_LOSS":0.3,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 
        elif self.model_case=="autoencoder_6k_Chengdu":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.0005, "SPEARMAN_CORRELATION_LOSS":0.1,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="autoencoder_05k_Chengdu":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0500", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.1, "MEDIAN_LOSS_batch":0.05,  "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="autoencoder_0016_Chengdu":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0016", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8,  "DECORRELATION_LATENT_LOSS":  0.01}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)       
        elif self.model_case=="autoencoder_0064_Chengdu":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0064", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="GAN_linear_pretrained_0064_Chengdu":
            dataloader = DataLoader(mode="graph_roads", seed=self.seed_data, name_dataset="China_Chengdu_A0064", device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_nets["AE"] = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_nets["AE"]):
                os.makedirs(self.path_folder_nets["AE"])
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            self.path_folder_nets["GAN"] = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_nets["GAN"]):
                os.makedirs(self.path_folder_nets["GAN"])
            self.loss_obj['GAN'] = LossFunction({}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)         

        '''
            
        self.modelTrainedAE = None
        self.data_splitted = dataloader.dataset_load(draw_plots=True, save_summary=True, loss=self.loss_obj, draw_correlationCoeff=self.draw_correlationCoeff)
                
        self.dataGenerator = dataloader.getDataGenerator()
        self.vc_mapping = dataloader.get_vcMapping()
        self.rangeData = dataloader.getRangeData()
        print("rangeData:\t",self.rangeData)
        self.statsData = dataloader.get_statsData()
        self.path_map = dataloader.get_pathMap()
        self.edge_index = dataloader.get_edgeIndex()
        self.case_setting.set_edge_index(self.edge_index)
        self.case_setting.deploy_models()
        
        for key in self.loss_obj:
            self.loss_obj[key].set_stats_data(self.statsData, self.vc_mapping)
        
        if do_optimization: 
            self.do_optimization = True
            time_opt_folder = Path(self.summary_path,"time_optimization")
            if not os.path.exists(time_opt_folder):
                os.makedirs(time_opt_folder)
            
            opt_time_analysis = self.time_performance
            self.optimization = Optimization(model=self.model, device=self.device, data_dict=self.data_splitted,
                loss=self.loss_obj, path_folder=self.path_folder, time_performance = opt_time_analysis,
                univar_count=self.univar_count, batch_size=self.batch_size, latent_dim=self.lat_dim, vc_mapping=self.vc_mapping, 
                input_shape=self.input_shape, rangeData=self.rangeData, dataGenerator=self.dataGenerator, 
                instaces_size_noise=self.instaces_size_noise, direction="maximize", timeout=600,
                graph_topology=self.graph_topology, edge_index=self.edge_index)
            self.optimization.set_fromDict(opt_settings)
        else:
            self.do_optimization = False
            self.optimization = None

    def start_experiment(self, load_model=False):
        comparison_corr_list = list()
                
        if self.trainingMode in ["AE>GAN","AE>WGAN"]:
            self.model["AE"] =  self.case_setting.get_model(key="AE")
            trained_obj_ae = self.training_model(data_dict=self.data_splitted, model_type="AE", model=self.model["AE"], loss_obj=self.loss_obj["AE"], epoch=self.epoch, graph_topology = self.graph_topology, optimization=self.do_optimization, optimizer_trial=self.optimization)
            model_ae_trained = trained_obj_ae[0]
            #self.predict_model(model=model_ae_trained, model_type="AE", data=self.data_splitted,path_folder_pred=self.path_folder_nets["AE"], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
            model_ae_decoder, model_ae_decoder_size = model_ae_trained.getModel("decoder", train=True, size=True)
            if self.trainingMode == "AE>GAN":
                model_key = "GAN"
            if self.trainingMode == "AE>WGAN":
                model_key = "WGAN"
            self.model[model_key] = self.case_setting.get_model(key=model_key)
            self.model[model_key].set_partialModel(key="generator", model_net=model_ae_decoder, model_size=model_ae_decoder_size)
            
            trained_obj_gan = self.training_model(self.data_splitted, model_type=model_key, model=self.model[model_key], loss_obj=self.loss_obj[model_key], pre_trained_decoder=True,epoch=self.epoch)
            model_gan_trained = trained_obj_gan[0]
            self.predict_model(model=model_gan_trained, model_type=model_key, data=self.data_splitted, path_folder_pred=self.path_folder_nets[model_key], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
        
        ''' 
        elif self.model_case=="GAN_GCN_linear_pretrained_URB_ZONE0_CHENGDU_bt":            
            self.model['AE'] = GEN_autoEncoderGCN_zone0
            self.graph_topology = True
            trained_obj_ae = self.training_model(data_dict=self.data_splitted, model_type="AE", model=self.model['AE'], loss_obj=self.loss_obj['AE'], epoch=self.epoch, graph_topology = self.graph_topology, optimization=self.do_optimization, optimizer_trial=self.optimization)
            model_ae_trained = trained_obj_ae[0]
            self.predict_model(model=model_ae_trained, model_type="AE", data=self.data_splitted,path_folder_pred=self.path_folder_nets["AE"], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)            
            self.model['GAN'] = GAN_neural_mixed_ZONE0(generator=model_ae_decoder)
            trained_obj_gan = self.training_model(self.data_splitted, model_type="GAN", model=self.model['GAN'], loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=self.epoch)
            model_gan_trained = trained_obj_gan[0]
            self.predict_model(model=model_gan_trained, model_type="GAN", data=self.data_splitted, path_folder_pred=self.path_folder_nets["GAN"], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
        
        elif self.model_case=="GAN_GCN_linear_pretrained_URB_ZONE1_CHENGDU_bt":            
            self.model['AE'] = GEN_autoEncoderGCN_zone1            
            self.graph_topology = True
            model_ae_trained = self.training_model(self.data_splitted, model_type="AE", model=self.model['AE'], loss_obj=self.loss_obj['AE'], epoch=self.epoch, graph_topology = self.graph_topology)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=self.data_splitted, path_folder_pred=self.path_folder_nets["AE"], input_shape="vector")   
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)            
            self.model['GAN'] = GAN_neural_mixed_64(generator=model_ae_decoder)
            model_gan_trained = self.training_model(self.data_splitted, model_type="GAN", model=self.model['GAN'], loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=self.epoch)
            self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=self.data_splitted, path_folder_pred=self.path_folder_nets["GAN"], input_shape="vector")
        elif self.model_case=="GAN_GCN_linear_pretrained_URB_ZONE1-2_CHENGDU_bt":            
            self.model['AE'] = GEN_autoEncoderGCN_zones_1_2
            self.graph_topology = True
            trained_obj_ae = self.training_model(data_dict=self.data_splitted, model_type="AE", model=self.model['AE'], loss_obj=self.loss_obj['AE'], epoch=self.epoch, graph_topology = self.graph_topology, optimization=self.do_optimization, optimizer_trial=self.optimization)
            model_ae_trained = trained_obj_ae[0]
            self.predict_model(model=model_ae_trained, model_type="AE", data=self.data_splitted,path_folder_pred=self.path_folder_nets["AE"], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)            
            self.model['GAN'] = GAN_neural_mixed_0437(generator=model_ae_decoder)
            trained_obj_gan = self.training_model(self.data_splitted, model_type="GAN", model=self.model['GAN'], loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=self.epoch)
            model_gan_trained = trained_obj_gan[0]
            self.predict_model(model=model_gan_trained, model_type="GAN", data=self.data_splitted, path_folder_pred=self.path_folder_nets["GAN"], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
        
             
        #ESG
        elif self.model_case=="ESG__GAN_linear_pretrained_35":
            self.model['AE'] = ESG__GEN_autoEncoder_35
            self.graph_topology = False
            trained_obj_ae = self.training_model(data_dict=self.data_splitted, model_type="AE", model=self.model['AE'], loss_obj=self.loss_obj['AE'], epoch=self.epoch, graph_topology = self.graph_topology, optimization=self.do_optimization, optimizer_trial=self.optimization)
            model_ae_trained = trained_obj_ae[0]
            self.predict_model(model=model_ae_trained, model_type="AE", data=self.data_splitted,path_folder_pred=self.path_folder_nets["AE"], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)            
            self.model['GAN'] = ESG__GAN_neural_mixed_35(generator=model_ae_decoder)
            trained_obj_gan = self.training_model(self.data_splitted, model_type="GAN", model=self.model['GAN'], loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=self.epoch)
            model_gan_trained = trained_obj_gan[0]
            self.predict_model(model=model_gan_trained, model_type="GAN", data=self.data_splitted, path_folder_pred=self.path_folder_nets["GAN"], path_folder_data=self.path_folder, noise_samples=1000, input_shape="vector", draw_plot=self.draw_plot, draw_scenarios=self.draw_scenarios, draw_correlationCoeff=self.draw_correlationCoeff)   
        ''' 
        
        if self.graph_topology:
            net_details = NetworkDetails(model=self.model, loss=self.loss_obj, path=self.summary_path, edge_index = self.data_splitted['edge_index'])
        else:
            net_details = NetworkDetails(model=self.model, loss=self.loss_obj, path=self.summary_path)
        net_details.saveModelParams()
        
        corr_comp = CorrelationComparison(self.corrCoeff, self.path_folder)
        corr_comp.compareMatrices(comparison_corr_list)        
    
    def training_model(self, data_dict, model_type, optimization=False, optimizer_trial=None,  model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False, graph_topology=False):
        if loss_obj is None:
            loss_obj = self.loss_obj
        if model is None:
            model = self.model
        if epoch is None:
            epoch=self.epoch
        print("\t\tGRAPH TOPOLOGY:\t",self.graph_topology)
        if optimization:
            print("\tOPTIMIZATION:\tTrue")
            self.optimization.optimization()
            self.optimization.setValuesOptimized(loss_obj= loss_obj, model_type=model_type)
            print(self.loss_obj[model_type].get_lossTerms())
        else:
            print("\tOPTIMIZATION:\tFalse")
            
        print("TRAINING PHASE: Training data - ", model_type)
        train_data = data_dict['train_data']   
        test_data = data_dict['test_data'] 
        edge_index = data_dict['edge_index']
            
        training_obj = ModelTraining(model=model, device=self.device, loss_obj=loss_obj, epoch=epoch, train_data=train_data, test_data=test_data, dataGenerator=self.dataGenerator, path_folder=self.path_folder, univar_count_in = self.univar_count, univar_count_out = self.univar_count, latent_dim=self.lat_dim, model_type=model_type, pre_trained_decoder=pre_trained_decoder, vc_mapping = self.vc_mapping,input_shape=self.input_shape, rangeData=self.rangeData,batch_size=self.batch_size, optimization=False, graph_topology=graph_topology, edge_index=edge_index, time_performance=self.time_performance)
        if model_type =="AE":
            optim_score = training_obj.training(training_name=f"MAIN_",model_flatten_in=model_flatten_in,load_model=load_model)
        elif model_type in ["GAN","WGAN"]:
            optim_score = training_obj.training(training_name=f"MAIN_",noise_size=self.instaces_size_noise, load_model=load_model)
        training_obj.eval()
        
        if optimizer_trial is None:
            return training_obj, None
        else:
            return training_obj, optim_score

    def predict_model(self, model, model_type, data, input_shape, path_folder_data, path_folder_pred, noise_samples=1000, draw_plot=True, draw_scenarios=True, draw_correlationCoeff=True):
        predMod = PerformePrediction(model=model, device=self.device,  model_type=model_type, univar_count=self.univar_count, latent_dim=self.lat_dim, data=data, dataGenerator=self.dataGenerator, input_shape = input_shape, rangeData=self.rangeData, vc_mapping=self.vc_mapping, draw_plot=draw_plot, draw_scenarios=draw_scenarios, draw_correlationCoeff= draw_correlationCoeff, noise_samples=noise_samples, path_folder_pred=path_folder_pred, path_folder_data= path_folder_data, path_map=self.path_map, time_performance=self.time_performance)
        predMod.predict_model(cases_list = self.performace_cases[model_type])
        self.time_performance.save_time()
from src.NeuroCorrelation.DataLoaders.DataSynteticGeneration import DataSynteticGeneration
from src.NeuroCorrelation.Models.AutoEncoderModels import *
from src.NeuroCorrelation.Models.GAN_neural import *
from src.NeuroCorrelation.ModelTraining.LossFunctions import LossFunction
from src.NeuroCorrelation.ModelTraining.ModelTraining import ModelTraining
from src.NeuroCorrelation.Analysis.ModelPrediction import ModelPrediction
from src.NeuroCorrelation.Analysis.DataComparison import DataComparison, CorrelationComparison
from src.NeuroCorrelation.Analysis.DataStatistics import DataStatistics
from src.NeuroCorrelation.DataLoaders.DataMapsLoader import DataMapsLoader
from src.NeuroCorrelation.Optimization.Optimization import Optimization
from src.NeuroCorrelation.Models.NetworkDetail import NetworkDetails
from src.NeuroCorrelation.DataLoaders.DataLoader import DataLoader



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

    def __init__(self, device, path_folder, epoch, batch_size,  model_case, dataset_setting, univar_count, lat_dim, instaces_size, input_shape):
        device = ("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.device = device
        print("SETTING PHASE: Device selection")
        print("\tdevice:\t",self.device)
        
        self.univar_count = univar_count        
        self.lat_dim = lat_dim
        self.epoch = epoch
        self.dataset_setting = dataset_setting
        self.batch_size = batch_size
        self.path_folder = Path('data','neuroCorrelation',path_folder)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        self.instaces_size = instaces_size
        self.input_shape = input_shape
        self.model_trained = None
        self.loss_obj = dict()
        self.instaces_size_noise = (self.instaces_size, self.lat_dim)
        self.corrCoeff = dict()
        
        summary_path = Path(self.path_folder,'summary')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        
        print("SETTING PHASE: Model creation")
        print("\tmodel_case:\t",model_case)
        self.model_case = model_case
        
        if self.model_case=="autoencoder_3_copula_optimization":            
            dataloader = DataLoader(mode="random_var", name_dataset="copula", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_3
            self.loss_obj['AE'] = LossFunction({"MSE_LOSS":  1, "SPEARMAN_CORRELATION_LOSS":1, "COVARIANCE_LOSS": 1, "DECORRELATION_LATENT_LOSS":  0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_3_defined":
            dataloader = DataLoader(mode="random_var", name_dataset="3var_defined", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_3
            self.loss_obj['AE'] = LossFunction({"MSE_LOSS":  1, "SPEARMAN_CORRELATION_LOSS":1, "COVARIANCE_LOSS": 1, "DECORRELATION_LATENT_LOSS":  0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_16_PEMS":
            dataloader = DataLoader(mode="graph_roads", name_dataset="PEMS_16", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_16
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":0.8}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_16_MetrLa":
            dataloader = DataLoader(mode="graph_roads", name_dataset="MetrLA_16", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_16
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)      
        
        elif self.model_case=="autoencoder_ALL_PEMS":
            dataloader = DataLoader(mode="graph_roads", name_dataset="PEMS_all", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_325
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
        elif self.model_case=="autoencoder_ALL_MetrLa":
            dataloader = DataLoader(mode="graph_roads", name_dataset="MetrLA_all", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_207
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.5}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)      
            
        elif self.model_case=="GAN_linear_16_PEMS":
            dataloader = DataLoader(mode="graph_roads", name_dataset="PEMS_16", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GAN_neural_16()
            self.loss_obj['AE'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 
            
        elif self.model_case=="GAN_linear_vc_copula":
            dataloader = DataLoader(mode="random_var", name_dataset="copula", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GAN_LinearNeural_7()
            self.loss_obj['AE'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)      

        elif self.model_case=="GAN_conv_vc_7_copula":
            dataloader = DataLoader(mode="random_var", name_dataset="copula", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GAN_Conv_neural_7()
            self.loss_obj['AE'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

        elif self.model_case=="GAN_linear_pretrained_vc_copula":
            dataloader = DataLoader(mode="random_var", name_dataset="copula", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_ae = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_ae):
                os.makedirs(self.path_folder_ae)
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.05, "SPEARMAN_CORRELATION_LOSS":0.3,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

            self.path_folder_gan = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_gan):
                os.makedirs(self.path_folder_gan)
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 
        
        elif self.model_case=="GAN_linear_pretrained_16_PEMS":
            dataloader = DataLoader(mode="graph_roads", name_dataset="PEMS_16", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_ae = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_ae):
                os.makedirs(self.path_folder_ae)
            self.model = GEN_autoEncoder_16
            #self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.05, "SPEARMAN_CORRELATION_LOSS":0.3}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
            
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_dataset":0.005, "SPEARMAN_CORRELATION_LOSS":0.3}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
            self.path_folder_gan = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_gan):
                os.makedirs(self.path_folder_gan)
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 

        elif self.model_case=="AE_conv_vc_copula":
            dataloader = DataLoader(mode="random_var", name_dataset="copula", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_ae = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_ae):
                os.makedirs(self.path_folder_ae)
            
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.05, "SPEARMAN_CORRELATION_LOSS":0.3,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

            self.path_folder_gan = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_gan):
                os.makedirs(self.path_folder_gan)
            self.loss_obj['GAN'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 


        elif self.model_case=="autoencoder_6k_Chengdu":
            dataloader = DataLoader(mode="graph_roads", name_dataset="China_Chengdu", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_6k
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.0005, "SPEARMAN_CORRELATION_LOSS":0.1,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_05k_Chengdu":
            dataloader = DataLoader(mode="graph_roads", name_dataset="China_Chengdu_A0500", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_05k
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.1, "MEDIAN_LOSS_batch":0.05,  "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

        elif self.model_case=="autoencoder_0016_Chengdu":
            dataloader = DataLoader(mode="graph_roads", name_dataset="China_Chengdu_A0016", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_16
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8,  "DECORRELATION_LATENT_LOSS":  0.01}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_0064_Chengdu":
            dataloader = DataLoader(mode="graph_roads", name_dataset="China_Chengdu_A0064", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.model = GEN_autoEncoder_64
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

        elif self.model_case=="GAN_linear_pretrained_0064_Chengdu":
            dataloader = DataLoader(mode="graph_roads", name_dataset="China_Chengdu_A0064", device=self.device, dataset_setting=self.dataset_setting, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
            self.path_folder_ae = Path(self.path_folder,'AE')
            if not os.path.exists(self.path_folder_ae):
                os.makedirs(self.path_folder_ae)
            self.model = GEN_autoEncoder_64
            
            self.loss_obj['AE'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS_batch":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
            self.path_folder_gan = Path(self.path_folder,'GAN')
            if not os.path.exists(self.path_folder_gan):
                os.makedirs(self.path_folder_gan)
            self.loss_obj['GAN'] = LossFunction({}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)         

        self.modelTrainedAE = None
        
        
        net_details = NetworkDetails(model=self.model, loss=self.loss_obj, path=summary_path)
        net_details.saveModelParams()
        
                
        self.data_splitted = dataloader.dataset_load(draw_plots=True, save_summary=True)
        self.dataGenerator = dataloader.getDataGenerator()
        self.vc_mapping = dataloader.get_vcMapping()
        self.rangeData = dataloader.getRangeData()
        self.statsData = dataloader.get_statsData()
        
        for key in self.loss_obj:
            self.loss_obj[key].set_stats_data(self.statsData, self.vc_mapping)

    def start_experiment(self, load_model=False):
        comparison_corr_list = list()
        
        if self.model_case=="autoencoder_3_copula_optimization":
            self.optimization = Optimization(model=self.model, device=self.device, data_dict=self.data_splitted, model_type="AE", 
                                            epoch=self.epoch, loss=self.loss_obj, path_folder=self.path_folder, 
                                            univar_count=self.univar_count, batch_size=self.batch_size, 
                                            latent_dim=self.lat_dim, vc_mapping=self.vc_mapping, input_shape=self.input_shape, rangeData=self.rangeData,
                                            dataGenerator=self.dataGenerator, instaces_size_noise=None, direction="maximize", 
                                            timeout=600)
            search_space = [{"type":"Categorical","min":0,"max":1, "values_list":[0,1,2], "name":"cat"},{"type":"Integer","min":0,"max":1, "values_list":[], "name":"int"}]
            
            search_space = [
                {"type":"Real","min":0.5,"max":1, "values_list":None, "name":"MSE_LOSS"},
                {"type":"Real","min":0.5,"max":1, "values_list":None, "name":"COVARIANCE_LOSS"},
                {"type":"Real","min":0.5,"max":1, "values_list":None, "name":"DECORRELATION_LATENT_LOSS"}]
            self.optimization.set_searchSpace(search_space)
            self.optimization.set_optimizer(base_estimator="GP", n_initial_points=10)            
            
            
            
            self.optimization.optimization(n_trials=2, network_key = "AE")
            #model_ae = self.training_model(self.data_splitted, model_type="AE", load_model=load_model)
            #self.predict_model(model=self.model_trained, model_type="AE", data_dict=self.data_splitted, input_shape="vector")
        
        elif self.model_case=="autoencoder_3_defined":
            model_ae = self.optimization_model(data_dict, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted)
            
        elif self.model_case=="autoencoder_16_PEMS":
            model_ae = self.training_model(self.data_splitted, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted, input_shape="vector")
            
        elif self.model_case=="autoencoder_16_MetrLa":
            model_ae = self.training_model(self.data_splitted, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted)
            
        elif self.model_case=="autoencoder_ALL_PEMS":
            model_ae = self.training_model(self.data_splitted, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted)
            
        elif self.model_case=="autoencoder_ALL_MetrLa":
            model_ae = self.training_model(self.data_splitted, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted)
            
        elif self.model_case=="GAN_linear_16_PEMS":
            model_gan = self.training_model(self.data_splitted, model_type="GAN")
            self.predict_model(model=model_gan, model_type="GAN", data_dict=self.data_splitted)
            
        elif self.model_case=="GAN_linear_vc_copula":
           model_gan = self.training_model(self.data_splitted, model_type="GAN")
           self.predict_model(model=model_gan, model_type="GAN", data_dict=self.data_splitted)
        
        elif self.model_case=="GAN_conv_vc_7_copula":
            model_gan = self.training_model(self.data_splitted, model_type="GAN")
            self.predict_model(model=model_gan, model_type="GAN", data_dict=self.data_splitted, input_shape="matrix")
            comparison_corr_list = [
                # original data
                [('data','train'),('data','train')],
                [('data','train'),('data','test')],
                
                # GAN
                [('data','train'),('GAN_noise','output')],
                
            ]

        elif self.model_case=="GAN_linear_pretrained_vc_copula":
            self.model_ae = GEN_autoEncoder_3
            
            model_ae_trained = self.training_model(self.data_splitted, model_type="AE", model=self.model_ae, loss_obj=self.loss_obj['AE'], epoch=10)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=self.data_splitted, path_folder_pred=self.path_folder_ae, input_shape="vector")
            
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)
            
            self.model_gan = GAN_neural_mixed_7(generator=model_ae_decoder)
            
            model_gan_trained = self.training_model(self.data_splitted, model_type="GAN", model=self.model_gan, loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=30)
            self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=self.data_splitted, path_folder_pred=self.path_folder_gan, input_shape="vector")
            comparison_corr_list = [
                # original data
                [('data','train'),('data','train')],
                [('data','train'),('data','test')],
                # AE
                [('AE_train','input'),('AE_train','output')],
                [('AE_train','input'),('AE_noise','output')],
                [('AE_train','input'),('AE_copulaLat','output')],                
                [('AE_noise','output'),('AE_copulaLat','output')],
                [('AE_train','output'),('AE_noise','output')],
                [('AE_train','output'),('AE_copulaLat','output')],
                # GAN
                [('data','train'),('GAN_noise','output')],
                # AE vs GAN
                [('AE_train','output'),('GAN_noise','output')],
                [('AE_noise','output'),('GAN_noise','output')],
                [('AE_copulaLat','output'),('GAN_noise','output')],
            ]


        elif self.model_case=="GAN_linear_pretrained_16_PEMS":            
            self.model_ae = GEN_autoEncoder_16
            model_ae_trained = self.training_model(self.data_splitted, model_type="AE", model=self.model_ae, loss_obj=self.loss_obj['AE'], epoch=self.epoch)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=self.data_splitted, path_folder_pred=self.path_folder_ae, input_shape="vector")
            
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)
            
            self.model_gan = GAN_neural_mixed_16(generator=model_ae_decoder)
            model_gan_trained = self.training_model(self.data_splitted, model_type="GAN", model=self.model_gan, loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=self.epoch)
            self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=self.data_splitted, path_folder_pred=self.path_folder_gan, input_shape="vector")
        
        elif self.model_case=="AE_conv_vc_copula":
            self.model_ae = GEN_ConvAutoEncoder_7
            
            model_ae_trained = self.training_model(self.data_splitted, model_type="AE", model=self.model_ae, loss_obj=self.loss_obj['AE'], epoch=10, model_flatten_in=False)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=self.data_splitted, path_folder_pred=self.path_folder_ae, input_shape="vector")
            
            #model_ae_decoder = model_ae_trained.getModel("decoder",train=True)
            
            #self.model_gan = GAN_neural_mixed_7(generator=model_ae_decoder)
            #

            #model_gan_trained = self.training_model(self.data_splitted, model_type="GAN", model=self.model_gan, loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=30)
            #self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=self.data_splitted, path_folder_pred=self.path_folder_gan, input_shape="vector")
            #comparison_corr_list = [['train_data','train_data'],['train_data','test_data'],['train_data','noise_out_data_GAN'], ['train_in_data_AE','train_out_data_AE'],['train_in_data_AE','noise_out_data_AE'],['train_in_data_AE','copula_out_data_AE'],['noise_out_data_AE', 'copula_out_data_AE'],['train_data','noise_out_data_GAN'],['noise_out_data_AE','noise_out_data_GAN'],['copula_out_data_AE','noise_out_data_GAN']]
     
        elif self.model_case=="autoencoder_6k_Chengdu":
            model_ae = self.training_model(self.data_splitted, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted, input_shape="vector")
        
        elif self.model_case=="autoencoder_05k_Chengdu":
            model_ae = self.training_model(self.data_splitted, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted, input_shape="vector")
            comparison_corr_list = [                # original data
                [('data','train'),('data','train')],
                [('data','train'),('data','test')],
                # AE
                [('AE_train','input'),('AE_train','output')],
                [('AE_train','input'),('AE_noise','output')],
                [('AE_train','input'),('AE_copulaLat','output')],
                [('AE_noise','output'),('AE_copulaLat','output')],
                [('AE_train','output'),('AE_noise','output')],
                [('AE_train','output'),('AE_copulaLat','output')],
            ]
            
        elif self.model_case=="autoencoder_0016_Chengdu":
            model_ae = self.training_model(self.data_splitted, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted, input_shape="vector")
            comparison_corr_list = [
                # original data
                [('data','train'),('data','train')],
                [('data','train'),('data','test')],
                # AE
                [('AE_train','input'),('AE_train','output')],
                [('AE_train','input'),('AE_noise','output')],
                [('AE_train','input'),('AE_copulaLat','output')],                
                [('AE_noise','output'),('AE_copulaLat','output')],
                [('AE_train','output'),('AE_noise','output')],
                [('AE_train','output'),('AE_copulaLat','output')],
                
            ]
        elif self.model_case=="autoencoder_0064_Chengdu":
            model_ae = self.training_model(self.data_splitted, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=self.data_splitted, input_shape="vector")
            comparison_corr_list = [
                # original data
                [('data','train'),('data','train')],
                [('data','train'),('data','test')],
                # AE
                [('AE_train','input'),('AE_train','output')],
                [('AE_train','input'),('AE_noise','output')],
                [('AE_train','input'),('AE_copulaLat','output')],                
                [('AE_noise','output'),('AE_copulaLat','output')],
                [('AE_train','output'),('AE_noise','output')],
                [('AE_train','output'),('AE_copulaLat','output')],
                
            ]    
        elif self.model_case=="GAN_linear_pretrained_0064_Chengdu":            
            self.model_ae = GEN_autoEncoder_64
            model_ae_trained = self.training_model(self.data_splitted, model_type="AE", model=self.model_ae, loss_obj=self.loss_obj['AE'], epoch=100)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=self.data_splitted, path_folder_pred=self.path_folder_ae, input_shape="vector", draw_plot=True)
            
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)
            
            
            self.model_gan = GAN_neural_mixed_64(generator=model_ae_decoder)
            model_gan_trained = self.training_model(self.data_splitted, model_type="GAN", model=self.model_gan, loss_obj=self.loss_obj['GAN'], pre_trained_decoder=True,epoch=25)
            self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=self.data_splitted, path_folder_pred=self.path_folder_gan, input_shape="vector") 
            comparison_corr_list = [
                # original data
                [('data','train'),('data','train')],
                [('data','train'),('data','test')],
                # AE
                [('AE_train','input'),('AE_train','output')],
                [('AE_train','input'),('AE_noise','output')],
                [('AE_train','input'),('AE_copulaLat','output')],                
                [('AE_noise','output'),('AE_copulaLat','output')],
                [('AE_train','output'),('AE_noise','output')],
                [('AE_train','output'),('AE_copulaLat','output')],                
            ]
        
        corr_comp = CorrelationComparison(self.corrCoeff, self.path_folder)
        corr_comp.compareMatrices(comparison_corr_list)        
    
     
    
    def optimization_model(self, data_dict, model_type, model_trained, model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False):
        self.optimization.optimize(n_trials=2, objective=self.training_model(self.data_splitted,  model_type="AE"))
        
        #training_result = self.training_model(self.data_splitted, model_type, trial=None, model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False)
        #model_trained = copy.deepcopy(training_result[0])
        #return training_result[1]
    
    def training_model(self, data_dict, model_type, optimization=None, optimizar_trial=None,  model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False):
        if optimization:
            print("tOPTIMIZATION:\tTrue")
            optimization_name = optimizar_trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            print("\tOPTIMIZATION NAME\t",optimization_name)
            optimization_function = None
            
        else:
            print("\t\tOPTIMIZATION:\tFalse")
            optimization = False 
            optimization_name = None
            optimization_function = None
        print("TRAINING PHASE: Training data - ", model_type)
        train_data = data_dict['train_data']
        if loss_obj is None:
            loss_obj = self.loss_obj
        if model is None:
            model = self.model
        if epoch is None:
            epoch=self.epoch
            
        training_obj = ModelTraining(model=model, device=self.device, loss_obj=loss_obj, epoch=epoch, dataset=train_data, dataGenerator=self.dataGenerator, path_folder=self.path_folder, univar_count_in = self.univar_count, univar_count_out = self.univar_count, latent_dim=self.lat_dim, model_type=model_type, pre_trained_decoder=pre_trained_decoder, vc_mapping = self.vc_mapping,input_shape=self.input_shape, rangeData=self.rangeData, optimization=optimization, optimization_function=optimization_function, optimization_name=optimization_name)
        if model_type =="AE":
            optim_score = training_obj.training(batch_size=self.batch_size, model_flatten_in=model_flatten_in,load_model=load_model)
        elif model_type =="GAN":
            optim_score = training_obj.training(batch_size=self.batch_size, noise_size=self.instaces_size_noise, load_model=load_model)
        training_obj.eval()
        print("optim_score:",optim_score)
        
        if optimizar_trial is None:
            return training_obj
        else:
            return training_obj, optim_score

    def predict_model(self, model, model_type, data_dict,  input_shape, noise_samples=1000, draw_plot=True, path_folder_pred=None):
        if path_folder_pred is None:
            path_folder_pred = self.path_folder
        
        train_data = data_dict['train_data']
        test_data = data_dict['test_data']
        noise_data = data_dict['noise_data']
        
        if model_type =="AE":
            print("PHASE: AutoEncoder")
            if model is not None:
                modelAE = model.getModel("all")
                print("PREDICT PHASE: Training data")
                plot_name = "AE_train"
                train_analysis_folder = Path(path_folder_pred,"train_analysis")
                if not os.path.exists(train_analysis_folder):
                    os.makedirs(train_analysis_folder)
                train_modelPrediction = ModelPrediction(model=modelAE, device=self.device, dataset=train_data, vc_mapping= self.vc_mapping, univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.lat_dim, data_range=self.rangeData, input_shape=input_shape, path_folder=train_analysis_folder)                
                train_predict = train_modelPrediction.compute_prediction(experiment_name="train_test_data", remapping_data=True)
                
                print("\tSTATS PHASE:  Correlation and distribution")
                datastats_train = DataStatistics(univar_count_in=self.univar_count, univar_count_out=self.univar_count, dim_latent=self.lat_dim, data=train_predict, path_folder=train_analysis_folder)
                datastats_latent = True
                self.corrCoeff[plot_name] = datastats_train.get_corrCoeff(latent=datastats_latent)
                
                if draw_plot:
                    print("\tSTATS PHASE:  Plots")
                    plot_colors = {"input":"blue", "latent":"green", "output":"orange"}
                    distribution_compare = {"train_input":{'data':train_predict['prediction_data_byvar']['input'], 'color':plot_colors['input']}, "train_reconstructed":{'data':train_predict['prediction_data_byvar']['output'], 'color':plot_colors['output']}}
                    datastats_train.plot(plot_colors=plot_colors, plot_name=plot_name, distribution_compare=distribution_compare, latent=datastats_latent)

                #
                print("PREDICT PHASE: Testing data")
                plot_name = "AE_test"
                test_analysis_folder = Path(path_folder_pred,"test_data_analysis")
                if not os.path.exists(test_analysis_folder):
                    os.makedirs(test_analysis_folder)
                test_modelPrediction = ModelPrediction(model=modelAE, device=self.device, dataset=test_data, vc_mapping= self.vc_mapping, univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.lat_dim, data_range=self.rangeData, input_shape=input_shape, path_folder=test_analysis_folder)                
                test_predict = test_modelPrediction.compute_prediction(experiment_name="testing_test_data", remapping_data=True)
                
                print("\t\t Statistics data")
                datastats_test = DataStatistics(univar_count_in=self.univar_count, univar_count_out=self.univar_count, dim_latent=self.lat_dim, data=test_predict, path_folder=test_analysis_folder)
                datastats_latent = True
                self.corrCoeff[plot_name] = datastats_test.get_corrCoeff(latent=datastats_latent)
                
                
                if draw_plot:
                    print("\tSTATS PHASE:  Plots")
                    plot_colors = {"input":"blue", "latent":"green", "output":"orange"}
                    distribution_compare = {"train_input":{'data':train_predict['prediction_data_byvar']['input'],'color':'cornflowerblue', 'alpha':0.5}, "test_input":{'data':test_predict['prediction_data_byvar']['input'],'color':plot_colors['input']}, "test_reconstructed":{'data':test_predict['prediction_data_byvar']['output'],'color':plot_colors['output']}}
                    datastats_test.plot(plot_colors=plot_colors, plot_name=plot_name, distribution_compare=distribution_compare, latent=datastats_latent)
                #

                print("PREDICT PHASE: Noised data generation")
                plot_name = "AE_noise"
                noise_analysis_folder = Path(path_folder_pred,"noise_data_analysis")
                if not os.path.exists(noise_analysis_folder):
                    os.makedirs(noise_analysis_folder)
                modelTrainedDecoder = modelAE.get_decoder()
                
                noise_modelPrediction = ModelPrediction(model=modelTrainedDecoder, device=self.device, dataset=noise_data, vc_mapping= self.vc_mapping, univar_count_in=self.lat_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=input_shape, path_folder=noise_analysis_folder)                
                noise_predict = noise_modelPrediction.compute_prediction(experiment_name="noise_test_data", remapping_data=True)
                
                print("\tSTATS PHASE:  Correlation and distribution")
                datastats_noiseAE = DataStatistics(univar_count_in=self.lat_dim, univar_count_out=self.univar_count, dim_latent=None, data=noise_predict, path_folder=noise_analysis_folder)
                datastats_latent=False
                self.corrCoeff[plot_name] = datastats_noiseAE.get_corrCoeff(latent=datastats_latent)
                
                if draw_plot:
                    print("\tSTATS PHASE:  Plots")
                    plot_colors = {"input":"green","output":"m"}
                    distribution_compare = {"train_input":{'data':train_predict['prediction_data_byvar']['input'],'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data':noise_predict['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}
                    datastats_noiseAE.plot(plot_colors=plot_colors, plot_name=plot_name, distribution_compare=distribution_compare, latent=datastats_latent)


                print("PREDICT PHASE: Copula Latent data")
                plot_name = "AE_copulaLat"
                copulaLat_analysis_folder = Path(path_folder_pred,"copulaLat_data_analysis")
                if not os.path.exists(copulaLat_analysis_folder):
                    os.makedirs(copulaLat_analysis_folder)
                copulaLat_samples_starting = train_predict['latent_data_input']['latent']
                copulaLat_data, self.corrCoeff['copulaLatent_data_AE'] = self.dataGenerator.casualVC_generation(name_data="copulaLatent", real_data=copulaLat_samples_starting, univar_count=self.lat_dim, num_of_samples = noise_samples,  draw_plots=True)
                modelTrainedDecoder = modelAE.get_decoder()
                
                copulaLat_modelPrediction = ModelPrediction(model=modelTrainedDecoder, device=self.device, dataset=copulaLat_data, vc_mapping= self.vc_mapping, univar_count_in=self.lat_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=input_shape, path_folder=copulaLat_analysis_folder)                
                copulaLat_predict = copulaLat_modelPrediction.compute_prediction(experiment_name="copula_test_data", remapping_data=True)
                
                print("\tSTATS PHASE:  Correlation and distribution")
                datastats_copLatent = DataStatistics(univar_count_in=self.lat_dim, univar_count_out=self.univar_count, dim_latent=None, data=copulaLat_predict, path_folder=copulaLat_analysis_folder)
                datastats_latent=False
                self.corrCoeff[plot_name] = datastats_copLatent.get_corrCoeff(latent=datastats_latent)
                        
                if draw_plot:
                    print("\tSTATS PHASE:  Plots")
                    plot_colors = {"input":"green", "output":"darkviolet"}                    
                    distribution_compare = {"train_input":{'data':train_predict['prediction_data_byvar']['input'],'color':'cornflowerblue', 'alpha':0.5}, "copulaLatent_generated":{'data':copulaLat_predict['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}
                    datastats_copLatent.plot(plot_colors=plot_colors, plot_name=plot_name, distribution_compare=distribution_compare, latent=datastats_latent)
        
        
        elif model_type =="GAN":
            print("PHASE: Generative Adversarial Network")
            modelGEN = model.getModel(selection="gen", eval=True)
            modelDIS = model.getModel(selection="dis", eval=True)
            
            if modelGEN is not None and modelDIS is not None:
                noise_data = data_dict['noise_data']
                #
                print("PREDICT PHASE: Noised data generation")
                plot_name = "GAN_noise"
                noise_analysis_folder = Path(path_folder_pred,"noise_data_analysis")
                if not os.path.exists(noise_analysis_folder):
                    os.makedirs(noise_analysis_folder)
                traindata_input = self.dataset2var(train_data)
                
                noise_modelPrediction = ModelPrediction(model=modelGEN, device=self.device, dataset=noise_data, vc_mapping= self.vc_mapping, univar_count_in=self.lat_dim, univar_count_out=self.univar_count, latent_dim=None, data_range=self.rangeData, input_shape=input_shape, path_folder=noise_analysis_folder)
                noise_predict = noise_modelPrediction.compute_prediction(experiment_name="noise_test_data", remapping_data=True)
                
                print("\tSTATS PHASE:  Correlation and distribution")
                datastats_noiseGAN = DataStatistics(univar_count_in=self.lat_dim, univar_count_out=self.univar_count, dim_latent=None, data=noise_predict, path_folder=noise_analysis_folder)
                datastats_latent=False
                self.corrCoeff[plot_name] = datastats_noiseGAN.get_corrCoeff(latent=datastats_latent)
                
                if draw_plot:    
                    plot_colors = {"input":"green", "output":"m"}                    
                    distribution_compare = {"train_input":{'data':traindata_input,'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data':noise_predict['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}
                    datastats_noiseGAN.plot(plot_colors=plot_colors, plot_name=plot_name, distribution_compare=distribution_compare, latent=datastats_latent)

        
  

    def dataset2var(self, data, pred2numpy=True):
        var_byComp = dict()

        for id_var in range(self.univar_count):
            var_byComp[id_var] = list()
        
        for count, item in enumerate(data):
            
            if pred2numpy:
                item_np = item['sample'].detach().numpy()
            else:
                item_np = item['sample'][0][0]
            
            for id_var in range(self.univar_count):
                if pred2numpy:
                    var_byComp[id_var].append(item_np[id_var])
                else:
                    var_byComp[id_var].append(item_np[id_var][0])
        
        return var_byComp
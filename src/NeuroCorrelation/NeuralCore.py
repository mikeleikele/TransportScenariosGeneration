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

    def __init__(self, device, path_folder, epoch = 3, batch_size=20,  model_case="autoencoder_3", univar_count=7, lat_dim=4):
        device = ("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.device = device
        print("SETTING PHASE: Device selection")
        print("\tdevice:\t",self.device)
        
        self.univar_count = univar_count        
        self.lat_dim = lat_dim
        self.epoch = epoch
        self.batch_size = batch_size
        self.path_folder = Path('data','neuroCorrelation',path_folder)
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
        
        self.model_trained = None
        
        print("SETTING PHASE: Model creation")
        print("\tmodel_case:\t",model_case)
        self.model_case = model_case
        if self.model_case=="autoencoder_78":
            self.model = GEN_autoEncoder_78
            self.loss_obj = LossFunction(["MSE_LOSS", "VARIANCE_LOSS"], univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="autoencoder_3_copula_optimization" or self.model_case=="autoencoder_3_defined" :
            self.model = GEN_autoEncoder_3
            self.loss_obj = LossFunction({"MSE_LOSS":  1, "SPEARMAN_CORRELATION_LOSS":1, "COVARIANCE_LOSS": 1, "DECORRELATION_LATENT_LOSS":  0.001}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_16_PEMS":
            self.model = GEN_autoEncoder_16
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.005, "SPEARMAN_CORRELATION_LOSS":0.8}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="autoencoder_16_MetrLa":
            self.model = GEN_autoEncoder_16
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.00005, "SPEARMAN_CORRELATION_LOSS":0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)      
        elif self.model_case=="autoencoder_ALL_PEMS":
            self.model = GEN_autoEncoder_325
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.00005, "SPEARMAN_CORRELATION_LOSS":0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        elif self.model_case=="autoencoder_ALL_MetrLa":
            self.model = GEN_autoEncoder_207
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.5}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)      
        elif self.model_case=="GAN_linear_16_PEMS":
            self.model = GAN_neural_16()
            self.loss_obj = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 
        elif self.model_case=="GAN_linear_vc_copula":
            self.model = GAN_LinearNeural_7()
            self.loss_obj = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)      

        elif self.model_case=="GAN_conv_vc_7_copula":
            self.model = GAN_Conv_neural_7()
            self.loss_obj = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

        elif self.model_case=="GAN_linear_pretrained_vc_copula":
            self.path_folder_ae = Path(self.path_folder,'ae')
            if not os.path.exists(self.path_folder_ae):
                os.makedirs(self.path_folder_ae)
            self.loss_obj = dict()
            self.loss_obj['ae'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.05, "SPEARMAN_CORRELATION_LOSS":0.3,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

            self.path_folder_gan = Path(self.path_folder,'gan')
            if not os.path.exists(self.path_folder_gan):
                os.makedirs(self.path_folder_gan)
            self.loss_obj['gan'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 
        
        elif self.model_case=="GAN_linear_pretrained_16_PEMS":
            self.path_folder_ae = Path(self.path_folder,'ae')
            if not os.path.exists(self.path_folder_ae):
                os.makedirs(self.path_folder_ae)
            self.model = GEN_autoEncoder_16
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.05, "SPEARMAN_CORRELATION_LOSS":0.3}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
            self.loss_obj = dict()
            self.loss_obj['ae'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.005, "SPEARMAN_CORRELATION_LOSS":0.3}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
            
            self.path_folder_gan = Path(self.path_folder,'gan')
            if not os.path.exists(self.path_folder_gan):
                os.makedirs(self.path_folder_gan)
            self.loss_obj['gan'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 

        elif self.model_case=="AE_conv_vc_copula":
            self.path_folder_ae = Path(self.path_folder,'ae')
            if not os.path.exists(self.path_folder_ae):
                os.makedirs(self.path_folder_ae)
            self.loss_obj = dict()
            self.loss_obj['ae'] = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.05, "SPEARMAN_CORRELATION_LOSS":0.3,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

            self.path_folder_gan = Path(self.path_folder,'gan')
            if not os.path.exists(self.path_folder_gan):
                os.makedirs(self.path_folder_gan)
            self.loss_obj['gan'] = LossFunction(dict(), univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device) 


        elif self.model_case=="autoencoder_6k_Chengdu":
            self.model = GEN_autoEncoder_6k
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS":0.0005, "SPEARMAN_CORRELATION_LOSS":0.1,  "DECORRELATION_LATENT_LOSS":  0.1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_05k_Chengdu":
            self.model = GEN_autoEncoder_05k
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.1, "MEDIAN_LOSS":0.05,  "SPEARMAN_CORRELATION_LOSS":1}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)

        elif self.model_case=="autoencoder_0016_Chengdu":
            self.model = GEN_autoEncoder_16
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8,  "DECORRELATION_LATENT_LOSS":  0.01}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        
        elif self.model_case=="autoencoder_0064_Chengdu":
            self.model = GEN_autoEncoder_64
            self.loss_obj = LossFunction({"JENSEN_SHANNON_DIVERGENCE_LOSS":0.6, "MEDIAN_LOSS":0.00005, "SPEARMAN_CORRELATION_LOSS":0.8}, univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)


        self.modelTrainedAE = None
        self.corrCoeff = dict()
        
        summary_path = Path(self.path_folder,'summary')
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        net_details = NetworkDetails(model=self.model, loss=self.loss_obj, path=summary_path)
        net_details.saveModelParams()

    def start_experiment(self, load_model=False):
        comparison_corr_list = list()
        
        if self.model_case=="autoencoder_3_copula_optimization":
            self.instaces_size = 1
            
            data_dict = self.dataset_load(mode="vc_copula")

            self.optimization = Optimization(model=self.model, device=self.device, data_dict=data_dict, model_type="AE", epoch=self.epoch, loss=self.loss_obj, path_folder=self.path_folder, univar_count=self.univar_count, batch_size=self.batch_size, dataGenerator=self.dataGenerator, instaces_size_noise=None, direction="maximize", timeout=600)
            search_space = [{"type":"Categorical","min":0,"max":1, "values_list":[0,1,2], "name":"cat"},{"type":"Integer","min":0,"max":1, "values_list":[], "name":"int"}]
            
            search_space = [
                {"type":"Real","min":0.5,"max":1, "values_list":None, "name":"MSE_LOSS"},
                {"type":"Real","min":0.5,"max":1, "values_list":None, "name":"COVARIANCE_LOSS"},
                {"type":"Real","min":0.5,"max":1, "values_list":None, "name":"DECORRELATION_LATENT_LOSS"}]
            self.optimization.set_searchSpace(search_space)
            self.optimization.set_optimizer(base_estimator="GP", n_initial_points=10)            
            
            
            
            self.optimization.optimization(n_trials=2)
            #model_ae = self.training_model(data_dict, model_type="AE", load_model=load_model)
            #self.predict_model(model=self.model_trained, model_type="AE", data_dict=data_dict, input_shape="vector")
        elif self.model_case=="autoencoder_3_defined":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="vc_defined")
            model_ae = self.optimization_model(data_dict, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict)
        elif self.model_case=="autoencoder_16_PEMS":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.5, name_dataset="PEMS_16")
            model_ae = self.training_model(data_dict, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict, input_shape="vector")
        elif self.model_case=="autoencoder_16_MetrLa":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.5, name_dataset="MetrLA_16")
            model_ae = self.training_model(data_dict, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict)
        elif self.model_case=="autoencoder_ALL_PEMS":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.5, name_dataset="PEMS_all")
            model_ae = self.training_model(data_dict, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict)
        elif self.model_case=="autoencoder_ALL_MetrLa":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.5, name_dataset="MetrLA_all")
            model_ae = self.training_model(data_dict, model_type="AE")
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict)
        elif self.model_case=="GAN_linear_16_PEMS":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=1, name_dataset="PEMS_16")
            model_gan = self.training_model(data_dict, model_type="GAN")
            self.predict_model(model=model_gan, model_type="GAN", data_dict=data_dict)
        elif self.model_case=="GAN_linear_vc_copula":
            self.instaces_size = 1
            self.instaces_size_noise = (1, 7)
            data_dict = self.dataset_load(mode="vc_copula", starting_sample=50, train_sample=500000, test_samples = 1, noise_samples=100000)
            model_gan = self.training_model(data_dict, model_type="GAN")
            self.predict_model(model=model_gan, model_type="GAN", data_dict=data_dict)
        elif self.model_case=="GAN_conv_vc_7_copula":
            self.instaces_size = 32
            self.instaces_size_noise = (2, 6)
            data_dict = self.dataset_load(mode="vc_copula", starting_sample=50, train_sample=5000, test_samples = 1000, noise_samples=1000, instaces_size=self.instaces_size)
            model_gan = self.training_model(data_dict, model_type="GAN")
            self.predict_model(model=model_gan, model_type="GAN", data_dict=data_dict, input_shape="matrix")
            comparison_corr_list = [
                # original data
                [('data','train'),('data','train')],
                [('data','train'),('data','test')],
                
                # GAN
                [('data','train'),('GAN_noise','output')],
                
            ]

        elif self.model_case=="GAN_linear_pretrained_vc_copula":
            
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="vc_copula", starting_sample=50, train_sample=5000, test_samples = 1000, noise_samples=1000)
            
            self.model_ae = GEN_autoEncoder_3
            
            model_ae_trained = self.training_model(data_dict, model_type="AE", model=self.model_ae, loss_obj=self.loss_obj['ae'], epoch=10)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=data_dict, path_folder_pred=self.path_folder_ae, input_shape="vector")
            
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)
            
            self.model_gan = GAN_neural_mixed_7(generator=model_ae_decoder)
            self.instaces_size_noise = (1, 4)

            model_gan_trained = self.training_model(data_dict, model_type="GAN", model=self.model_gan, loss_obj=self.loss_obj['gan'], pre_trained_decoder=True,epoch=30)
            self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=data_dict, path_folder_pred=self.path_folder_gan, input_shape="vector")
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
            self.instaces_size = 1
            #data_dict = self.dataset_load(mode="vc_copula", starting_sample=50, train_sample=50000, test_samples = 1, noise_samples=25000)
            #
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.5, name_dataset="PEMS_16")

            self.model_ae = GEN_autoEncoder_3
            model_ae_trained = self.training_model(data_dict, model_type="AE", model=self.model_ae, loss_obj=self.loss_obj['ae'], epoch=50)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=data_dict, path_folder_pred=self.path_folder_ae)
            
            model_ae_decoder = model_ae_trained.getModel("decoder",train=True)
            
            self.model_gan = GAN_neural_mixed_7(generator=model_ae_decoder)
            model_gan_trained = self.training_model(data_dict, model_type="GAN", model=self.model_gan, loss_obj=self.loss_obj['gan'], pre_trained_decoder=True,epoch=100)
            self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=data_dict, path_folder_pred=self.path_folder_gan)
        
        elif self.model_case=="AE_conv_vc_copula":
            self.instaces_size = 32
            self.batch_size = 64
            data_dict = self.dataset_load(mode="vc_copula", starting_sample=50, train_sample=500, test_samples = 100, noise_samples=100, instaces_size=self.instaces_size)
            
            self.model_ae = GEN_ConvAutoEncoder_7
            
            model_ae_trained = self.training_model(data_dict, model_type="AE", model=self.model_ae, loss_obj=self.loss_obj['ae'], epoch=10, model_flatten_in=False)
            self.predict_model(model=model_ae_trained, model_type="AE", data_dict=data_dict, path_folder_pred=self.path_folder_ae, input_shape="vector")
            
            #model_ae_decoder = model_ae_trained.getModel("decoder",train=True)
            
            #self.model_gan = GAN_neural_mixed_7(generator=model_ae_decoder)
            #self.instaces_size_noise = (1, 4)

            #model_gan_trained = self.training_model(data_dict, model_type="GAN", model=self.model_gan, loss_obj=self.loss_obj['gan'], pre_trained_decoder=True,epoch=30)
            #self.predict_model(model=model_gan_trained, model_type="GAN", data_dict=data_dict, path_folder_pred=self.path_folder_gan, input_shape="vector")
            #comparison_corr_list = [['train_data','train_data'],['train_data','test_data'],['train_data','noise_out_data_GAN'], ['train_in_data_AE','train_out_data_AE'],['train_in_data_AE','noise_out_data_AE'],['train_in_data_AE','copula_out_data_AE'],['noise_out_data_AE', 'copula_out_data_AE'],['train_data','noise_out_data_GAN'],['noise_out_data_AE','noise_out_data_GAN'],['copula_out_data_AE','noise_out_data_GAN']]
     
        elif self.model_case=="autoencoder_6k_Chengdu":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.4, name_dataset="China_Chengdu", instaces_size=1, draw_plots=False)
            model_ae = self.training_model(data_dict, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict, input_shape="vector")
        
        elif self.model_case=="autoencoder_05k_Chengdu":
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.2, name_dataset="China_Chengdu_A0500", instaces_size=1, draw_plots=True)
            model_ae = self.training_model(data_dict, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict, input_shape="vector")
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
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.4, name_dataset="China_Chengdu_A0016", instaces_size=1, draw_plots=True)
            model_ae = self.training_model(data_dict, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict, input_shape="vector")
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
            self.instaces_size = 1
            data_dict = self.dataset_load(mode="graph_roads", train_percentual=0.99, name_dataset="China_Chengdu_A0064", instaces_size=1, draw_plots=True)
            model_ae = self.training_model(data_dict, model_type="AE", load_model=load_model)
            self.predict_model(model=model_ae, model_type="AE", data_dict=data_dict, input_shape="vector")
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

    def dataset_load(self, mode="vc_copula", train_percentual=0.70, starting_sample=20, train_sample=50, test_samples = 5000, noise_samples=10000, name_dataset=None, vc_dict=None, instaces_size=1, draw_plots=True):
        
        if mode=="vc_defined":
            print("DATASET PHASE: Sample generation")
            self.dataGenerator = DataSynteticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            self.rangeData = self.dataGenerator.getDataRange()
            self.dataGenerator.casualVC_init_3VC(num_of_samples = starting_sample, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train_data'] = self.dataGenerator.casualVC_generation(name_data="train", num_of_samples = train_sample, draw_plots=draw_plots)
            test_data, self.corrCoeff['test_data'] = self.dataGenerator.casualVC_generation(name_data="test", num_of_samples = test_samples,  draw_plots=draw_plots)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = noise_samples, draw_plots=draw_plots)
            self.vc_mapping = ['X', 'Y','Z']

        if mode=="vc_copula":
            print("DATASET PHASE: Sample copula generation")
            self.dataGenerator = DataSynteticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            self.rangeData = self.dataGenerator.getDataRange()
            if vc_dict == None:
                self.vc_dict = {"X":{"dependence":None}, "Y":{"dependence":{"X":1.6}}, "Z":{"dependence":{"X":3}}, "W":{"dependence":None},"K":{"dependence":{"W":0.5}}, "L":{"dependence":{"W":5}}, "M":{"dependence":None}}
            self.vc_mapping = list()
            for key_vc in self.vc_dict:
                self.vc_mapping.append(key_vc)
            self.corrCoeff['data'] = dict()
            self.dataGenerator.casualVC_init_multi(num_of_samples = starting_sample, vc_dict=self.vc_dict, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.casualVC_generation(name_data="train", univar_count=self.univar_count, num_of_samples = train_sample, draw_plots=draw_plots, instaces_size=self.instaces_size)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.casualVC_generation(name_data="test", univar_count=self.univar_count, num_of_samples = test_samples,  draw_plots=draw_plots, instaces_size=self.instaces_size)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = noise_samples, draw_plots=draw_plots, instaces_size=self.instaces_size)
        
        if mode=="graph_roads":
            print("DATASET PHASE: Load maps data")
            self.corrCoeff['data'] = dict()
            self.dataGenerator = DataMapsLoader(torch_device=self.device, name_dataset=name_dataset, lat_dim=self.lat_dim, univar_count=self.univar_count, path_folder=self.path_folder)
            self.dataGenerator.mapsVC_load(train_percentual=train_percentual, draw_plots=draw_plots)
            self.corrCoeff['data'] = dict()
            self.rangeData = self.dataGenerator.getDataRange()
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.mapsVC_getData(name_data="train", draw_plots=draw_plots)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.mapsVC_getData(name_data="test",  draw_plots=draw_plots)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = noise_samples, draw_plots=draw_plots)
            self.vc_mapping = self.dataGenerator.get_vc_mapping()

        if mode=="graph_statics":
            print("to implement")
            #self.train_data = self.dataGenerator.graphGen(num_of_samples = train_sample, with_cov=True)

        data_dict = {"train_data":train_data, "test_data":test_data, "noise_data":noise_data}
        return data_dict

    def optimization_model(self, data_dict, model_type, model_trained, model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False):
        self.optimization.optimize(n_trials=2, objective=self.training_model(data_dict,  model_type="AE"))
        
        #training_result = self.training_model(data_dict, model_type, trial=None, model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False)
        #model_trained = copy.deepcopy(training_result[0])
        #return training_result[1]
    
    def training_model(self, data_dict, model_type, trial=None, model=None, loss_obj=None, pre_trained_decoder=False, epoch=None, model_flatten_in=True, load_model=False):
        if trial is None:
            print("\t\tOPTIMIZATION:\tFalse")
        else:
            print("tOPTIMIZATION:\tTrue")
            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
            print("\tOPTIMIZATION NAME\t",optimizer_name)
        
        print("TRAINING PHASE: Training data - ", model_type)
        train_data = data_dict['train_data']
        if loss_obj is None:
            loss_obj = self.loss_obj
        if model is None:
            model = self.model
        if epoch is None:
            epoch=self.epoch

        training_obj = ModelTraining(model=model, device=self.device, loss_obj=loss_obj, epoch=epoch, dataset=train_data, dataGenerator=self.dataGenerator, path_folder=self.path_folder, univar_count = self.univar_count, model_type=model_type, pre_trained_decoder=pre_trained_decoder)
        if model_type =="AE":
            optim_score = training_obj.training(batch_size=self.batch_size, model_flatten_in=model_flatten_in,load_model=load_model)
        elif model_type =="GAN":
            optim_score = training_obj.training(batch_size=self.batch_size, noise_size=self.instaces_size_noise, load_model=load_model)
        training_obj.eval()
        if trial is None:
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
            if model is not None:
                modelAE = model.getModel("all")
                print("PREDICT PHASE: Training data")
                plot_name = "AE_train"
                train_analysis_folder = Path(path_folder_pred,"train_analysis")
                if not os.path.exists(train_analysis_folder):
                    os.makedirs(train_analysis_folder)
                train_predict = self.compute_prediction(model=modelAE, dataset=train_data, univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.lat_dim, experiment_name="train_test_data", remapping_data=True, folder=train_analysis_folder, data_range=self.rangeData, input_shape=input_shape)
                
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
                test_predict = self.compute_prediction(model=modelAE, dataset=test_data, univar_count_in=self.univar_count, univar_count_out=self.univar_count, latent_dim=self.lat_dim, experiment_name="testing_test_data", remapping_data=True, folder=test_analysis_folder, data_range=self.rangeData, input_shape=input_shape)
                
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
                noise_predict = self.compute_prediction(model=modelTrainedDecoder, dataset=noise_data, univar_count_in=self.lat_dim, univar_count_out=self.univar_count, latent_dim=None, experiment_name="noise_test_data", remapping_data=True,folder=noise_analysis_folder, data_range=self.rangeData, input_shape=input_shape)
                
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
                
                copulaLat_predict = self.compute_prediction(model=modelTrainedDecoder, dataset=copulaLat_data, univar_count_in=self.lat_dim, univar_count_out=self.univar_count, latent_dim=None, folder=copulaLat_analysis_folder, experiment_name="copula_test_data", remapping_data=True,  data_range=self.rangeData, input_shape=input_shape)

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
                noise_predict = self.compute_prediction(model=modelGEN, dataset=noise_data, univar_count_in=self.lat_dim, univar_count_out=self.univar_count, latent_dim=None, experiment_name="noise_test_data", remapping_data=True,folder=noise_analysis_folder, data_range=self.rangeData, input_shape=input_shape)
                
                print("\tSTATS PHASE:  Correlation and distribution")
                datastats_noiseGAN = DataStatistics(univar_count_in=self.lat_dim, univar_count_out=self.univar_count, dim_latent=None, data=noise_predict, path_folder=noise_analysis_folder)
                datastats_latent=False
                self.corrCoeff[plot_name] = datastats_noiseGAN.get_corrCoeff(latent=latent)
                
                if draw_plot:    
                    plot_colors = {"input":"green", "output":"m"}                    
                    distribution_compare = {"train_input":{'data':traindata_input,'color':'cornflowerblue', 'alpha':0.5}, "noise_generated":{'data':noise_predict['prediction_data_byvar']['output'],'color':plot_colors['output'], 'alpha':0.5}}
                    datastats_noiseGAN.plot(plot_colors=plot_colors, plot_name=plot_name, distribution_compare=distribution_compare, latent=latent)

        print(self.corrCoeff)

    def compute_prediction(self, model, dataset, univar_count_in, univar_count_out, latent_dim, experiment_name, folder, input_shape ,remapping_data=False, data_range=None):
        resultDict = dict()
        modelPrediction = ModelPrediction(model, device=self.device, univar_count_in=univar_count_in, univar_count_out=univar_count_out, latent_dim=latent_dim, data_range=data_range, input_shape=input_shape, path_folder=folder)
        if latent_dim is not None:
            modelPrediction.predict(dataset, latent=True,  experiment_name=experiment_name, remapping=remapping_data)
        else:
            modelPrediction.predict(dataset, latent=False, experiment_name=experiment_name, remapping=remapping_data)

        prediction_data = modelPrediction.getPred()
        prediction_data_byvar = modelPrediction.getPred_byUnivar()     
        resultDict["prediction_data"] = prediction_data
        resultDict["prediction_data_byvar"] = prediction_data_byvar
        
        inp_data_vc = pd.DataFrame()
        for id_univar in range(univar_count_in):
            var_name = self.vc_mapping[id_univar]
            inp_data_vc[var_name] = [a.tolist() for a in prediction_data_byvar['input'][id_univar]]
        resultDict["inp_data_vc"] = inp_data_vc

        out_data_vc = pd.DataFrame()
        for id_univar in range(univar_count_out):
                var_name = self.vc_mapping[id_univar]
                out_data_vc[var_name] = [a.tolist() for a in prediction_data_byvar['output'][id_univar]]
        resultDict["out_data_vc"] = out_data_vc

        if latent_dim is not None:
            lat_data = modelPrediction.getLat()
            resultDict["latent_data"] = lat_data
            lat_data_bycomp = modelPrediction.getLat_byComponent()
            resultDict["latent_data_bycomp"] = lat_data_bycomp        
            lat2dataInput = modelPrediction.getLatent2data()
            resultDict["latent_data_input"] = lat2dataInput  
        return resultDict

    def dataset2var(self, data, pred2numpy=True):
        var_byComp = dict()

        for id_var in range(self.univar_count):
            var_byComp[id_var] = list()
        
        for count, item in enumerate(data):
            if pred2numpy:
                item_np = item['sample'][0][0].detach().numpy()
            else:
                item_np = item['sample'][0][0]
            
            for id_var in range(self.univar_count):
                
                var_byComp[id_var].append(item_np[id_var][0])
        
        return var_byComp
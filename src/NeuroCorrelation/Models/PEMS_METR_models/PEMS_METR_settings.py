import os
from pathlib import Path
from src.NeuroCorrelation.Models.AutoEncoderModels import *
from src.NeuroCorrelation.DataLoaders.DataLoader import DataLoader
from src.NeuroCorrelation.Models.PEMS_METR_models.PEMS_METR_models import *
from src.NeuroCorrelation.ModelTraining.LossFunctions import LossFunction

class PEMS_METR_settings():
    
    def __init__(self, model_case, device, univar_count, lat_dim, dataset_setting, epoch, path_folder, corrCoeff, instaces_size, time_slot=None):
        self.model_case = model_case
        self.device = device
        self.dataset_setting = dataset_setting
        self.epoch = epoch
        self.univar_count = univar_count
        self.lat_dim = lat_dim
        self.corrCoeff = corrCoeff
        self.instaces_size = instaces_size
        self.path_folder = path_folder
        self.time_slot = time_slot
        self.model = dict()
        self.setting_model_case()
    
    def setting_model_case(self):
        if   self.model_case == "AE>GAN_linear_pretrained_METR_16":
            self.mode = "graph_roads"
            self.name_dataset = "METR_LA"
            self.version_dataset = "S16"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_16.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_16
            
        elif self.model_case == "AE>GAN_linear_pretrained_METR_32":
            self.mode = "graph_roads"
            self.name_dataset = "METR_LA"
            self.version_dataset = "S32"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_32.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_32
            
        elif self.model_case == "AE>GAN_linear_pretrained_METR_48":
            self.mode = "graph_roads"
            self.name_dataset = "METR_LA"
            self.version_dataset = "S48"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_48.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_48
            
        elif self.model_case == "AE>GAN_linear_pretrained_METR_64":
            self.mode = "graph_roads"
            self.name_dataset = "METR_LA"
            self.version_dataset = "S64"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_64.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_64
            
        if   self.model_case == "AE>GAN_linear_pretrained_PEMS_16":
            self.mode = "graph_roads"
            self.name_dataset = "PEMS_BAY"
            self.version_dataset = "S16"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_16.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_16
            
        elif self.model_case == "AE>GAN_linear_pretrained_PEMS_32":
            self.mode = "graph_roads"
            self.name_dataset = "PEMS_BAY"
            self.version_dataset = "S32"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_32.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_32
            
        elif self.model_case == "AE>GAN_linear_pretrained_PEMS_48":
            self.mode = "graph_roads"
            self.name_dataset = "PEMS_BAY"
            self.version_dataset = "S48"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_48.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_48
            
        elif self.model_case == "AE>GAN_linear_pretrained_PEMS_64":
            self.mode = "graph_roads"
            self.name_dataset = "PEMS_BAY"
            self.version_dataset = "S64"
            self.nets = ['AE', 'GAN']
            self.graph_topology = False
            self.loss_dict = {
                'AE':{"JENSEN_SHANNON_DIVERGENCE_LOSS":1, "MEDIAN_LOSS_batch":0.005, "SPEARMAN_CORRELATION_LOSS":1},
                'GAN': dict()
                }
            self.trainingMode = "AE>GAN"
            self.model['AE'] = AutoEncoderModels(load_from_file =True, json_filepath=Path('src','NeuroCorrelation','Models','PEMS_METR_models','PEMS_METR_64.json'), edge_index=None)
            self.model['GAN'] = PEMS_METR_GAN_64
            
        self.path_folder_nets = dict()
        for key in self.nets:
            self.path_folder_nets[key] = Path(self.path_folder, key)
            if not os.path.exists(self.path_folder_nets[key]):
                os.makedirs(self.path_folder_nets[key])
    
    def get_trainingMode(self):
        return self.trainingMode

    def get_model(self, key):
        return self.model[key]
        
    def get_time_slot(self):
        return self.time_slot
    
    def get_folder_nets(self):
        return self.path_folder_nets
    
    def get_graph_topology(self):
        return self.graph_topology
    
    def get_DataLoader(self, seed_data):      
        dataloader = DataLoader(mode="graph_roads", seed=seed_data, name_dataset=self.name_dataset, version_dataset=self.version_dataset, time_slot=self.time_slot, device=self.device, dataset_setting=self.dataset_setting, epoch = self.epoch, univar_count=self.univar_count, lat_dim=self.lat_dim, corrCoeff = self.corrCoeff, instaces_size=self.instaces_size, path_folder=self.path_folder)
        return dataloader
    
    def get_LossFunction(self):
        loss_obj = dict()
        for key in self.nets:    
            loss_obj[key] = LossFunction(self.loss_dict[key], univar_count=self.univar_count, latent_dim=self.lat_dim, device=self.device)
        return loss_obj

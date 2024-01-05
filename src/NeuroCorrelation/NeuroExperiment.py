from src.NeuroCorrelation.NeuralCore import NeuralCore
import os
from pathlib import Path

class NeuroExperiment():

    def __init__(self, num_case, main_folder, experiment_name, seed, load_model=None):
        id_experiments = int(num_case)
        experiments_list = [
            {"id":  0, "model_case":"autoencoder_3_copula_optimization",  "epoch":{'AE':   3,'GAN':   2}, "univar_count": 7, "lat_dim": 3, "dataset_setting":{"batch_size":  32, "train_percentual":None,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},
            {"id":  1, "model_case":"GAN_linear_pretrained_16_PEMS_bt",   "epoch":{'AE':  50,'GAN':  50}, "univar_count":16, "lat_dim":12, "dataset_setting":{"batch_size": 128, "train_percentual":0.5,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},
            {"id":  2, "model_case":"GAN_linear_pretrained_16_METR_bt",   "epoch":{'AE':  150,'GAN':  150}, "univar_count":16, "lat_dim":12, "dataset_setting":{"batch_size": 32, "train_percentual":0.4,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},            
            {"id":  3, "model_case":"GAN_linear_pretrained_32_METR_bt",   "epoch":{'AE':  50,'GAN':  50}, "univar_count":32, "lat_dim":12, "dataset_setting":{"batch_size": 128, "train_percentual":0.5,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},            
            {"id":  4, "model_case":"GAN_linear_pretrained_16_CHENGDU_bt","epoch":{'AE':  50,'GAN':  50}, "univar_count":16, "lat_dim":12, "dataset_setting":{"batch_size": 128, "train_percentual":0.5,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},
            {"id":  5, "model_case":"GAN_linear_pretrained_0064_Chengdu", "epoch":{'AE':  50,'GAN':  50}, "univar_count":64, "lat_dim":48, "dataset_setting":{"batch_size":  32, "train_percentual":None,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"}
        ]
        experiments_selected = experiments_list[id_experiments]
        
               
        folder_experiment = Path(main_folder, experiment_name)  

        #, 'autoencoder_05k_Chengdu','autoencoder_0016_Chengdu', 'autoencoder_6k_Chengdu','autoencoder_3_copula_optimization']
        print(f"|------------------------")
        print(f"| Modelcase   : {experiments_selected['model_case']}")
        print(f"|             : {experiments_selected}")
        print(f"|------------------------")
        print(f" ")
        nc = NeuralCore(device=None,epoch=experiments_selected["epoch"], model_case=experiments_selected["model_case"], univar_count=experiments_selected["univar_count"], lat_dim=experiments_selected["lat_dim"], dataset_setting=experiments_selected['dataset_setting'], instaces_size= experiments_selected["instaces_size"], input_shape= experiments_selected["input_shape"], path_folder=folder_experiment, seed=seed)
        if load_model=="--load":
            nc.start_experiment(load_model=True)
        else:
            nc.start_experiment()
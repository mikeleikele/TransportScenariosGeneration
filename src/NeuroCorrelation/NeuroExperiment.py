from src.NeuroCorrelation.NeuralCore import NeuralCore
import os
from pathlib import Path
from src.NeuroCorrelation.Analysis.AnalysisResult import AnalysisResult


class NeuroExperiment():

    
    #print("--neuroD (1)num_case::int  (2)experiment_name_suffix::int (3)main_folder::string (4)repeat::int (5)load_model::--load/None" (6)train_models::yes/no)
    def __init__(self, args):
        self.main(num_case=int(args[1]),experiment_name_suffix=args[2], main_folder=args[3], repeation=args[4], load_model=args[5], train_models=args[6])
        
        
    def main(self, num_case, experiment_name_suffix, main_folder, repeation, load_model=None, train_models="yes"):
        path_folder = Path('data','neuroCorrelation',main_folder)
        experiments_name = f"{experiment_name_suffix}___{num_case}"
        
        #2--metr16 ok
        #3--metr32 ok
        #4--metr48 doto
        
        #5--pems16 ok
        #6--pems32 ok
        #7--pems48 ok
        
        for seed in range(0, int(repeation)):
            experiment_name = f"{experiments_name}_{seed}"
            if train_models=="yes":
                univar_count = self.experiment(num_case=num_case, main_folder=path_folder, seed=seed, experiment_name=experiment_name,                                              load_model=load_model)
        
            id_experiments = int(num_case)
            experiments_list = self.getExperimentsList(seed=0)
            experiments_selected = experiments_list[id_experiments]
            univar_count=experiments_selected["univar_count"]
            
            aResult = AnalysisResult(univar_count)
            aResult.compute_statscomparison(folder=path_folder, exp_name=experiments_name, n_run=[seed])
        
    def getExperimentsList(self,seed):
        experiments_list = [
            {"id":  0, "model_case":"autoencoder_3_copula_optimization",  "epoch":{'AE':   3,'GAN':   2}, "univar_count": 7, "lat_dim": 3, "dataset_setting":{"batch_size":  32, "train_percentual":None,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},
            
            {"id":  1, "model_case":"GAN_linear_pretrained_16_PEMS_bt",   "epoch":{'AE':  50,'GAN':  50}, "univar_count":16, "lat_dim":12, "dataset_setting":{"batch_size": 128, "train_percentual":0.5,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},
            {"id":  2, "model_case":"GAN_linear_pretrained_16_METR_bt",   "epoch":{'AE': 50,'GAN': 2}, "univar_count":16, "lat_dim":12, "dataset_setting":{"batch_size":  32, "train_percentual":0.4,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},            
            {"id":  3, "model_case":"GAN_linear_pretrained_32_METR_bt",   "epoch":{'AE': 50,'GAN': 3}, "univar_count":32, "lat_dim":28, "dataset_setting":{"batch_size":  64, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},            
            {"id":  4, "model_case":"GAN_linear_pretrained_48_METR_bt",   "epoch":{'AE': 50,'GAN': 2}, "univar_count":48, "lat_dim":44, "dataset_setting":{"batch_size":  64, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                    
            
            {"id":  5, "model_case":"GAN_linear_pretrained_16_PEMS_bt",   "epoch":{'AE': 150,'GAN': 50}, "univar_count":16, "lat_dim":12, "dataset_setting":{"batch_size":  32, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":1000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},            
            {"id":  6, "model_case":"GAN_linear_pretrained_32_PEMS_bt",   "epoch":{'AE': 50,'GAN': 2}, "univar_count":32, "lat_dim":28, "dataset_setting":{"batch_size":  32, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},            
            {"id":  7, "model_case":"GAN_linear_pretrained_48_PEMS_bt",   "epoch":{'AE': 2,'GAN': 200}, "univar_count":48, "lat_dim":44, "dataset_setting":{"batch_size":  96, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                    
            
            {"id":  8, "model_case":"GAN_linear_pretrained_64_METR_bt",   "epoch":{'AE': 20,'GAN': 2}, "univar_count":64, "lat_dim":60, "dataset_setting":{"batch_size":  32, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                    
            {"id":  9, "model_case":"GAN_linear_pretrained_64_PEMS_bt",   "epoch":{'AE': 20,'GAN': 2}, "univar_count":64, "lat_dim":60, "dataset_setting":{"batch_size":  32, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                    
            
            {"id":  10, "model_case":"GAN_linear_pretrained_16_CHENGDU_bt","epoch":{'AE':  50,'GAN':  50}, "univar_count":16, "lat_dim":12, "dataset_setting":{"batch_size": 128, "train_percentual":0.5,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},
            {"id":  11, "model_case":"GAN_linear_pretrained_0064_Chengdu", "epoch":{'AE':  50,'GAN':  50}, "univar_count":64, "lat_dim":48, "dataset_setting":{"batch_size":  32, "train_percentual":None,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"}
        ]   
        return experiments_list
        
    def experiment(self, num_case, main_folder, seed, experiment_name, load_model):
        
        id_experiments = int(num_case)
        experiments_list = self.getExperimentsList(seed)
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
        res = {"univar_count": experiments_selected["univar_count"]}
        return res
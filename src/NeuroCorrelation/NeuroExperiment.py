
from src.NeuroCorrelation.NeuralCore import NeuralCore
import os
import argparse
from pathlib import Path
from termcolor import cprint
from colorama import init, Style
import gc
import resource


class NeuroExperiment():

    def __init__(self, args):
        gb_lim = 10
        limit_resource=False
        
        if limit_resource:
            self.limit_resource(gb_lim)
        
        parser = argparse.ArgumentParser(description="Description of your program")
        
        parser.add_argument('--num_case', type=int, required=True, help='Description for num_case')
        parser.add_argument('--experiment_name_suffix', type=str, required=True, help='Description for experiment_name_suffix')
        parser.add_argument('--main_folder', type=str, required=True, help='Description for main_folder')
        parser.add_argument('--repeation', type=int, required=True, help='Description for repeation')
        parser.add_argument('--optimization', type=self.str2bool, required=True, help='Enable or disable optimization')
        parser.add_argument('--load_model', type=self.str2bool, required=True, help='Description for load_model')
        parser.add_argument('--train_models', type=self.str2bool, required=True, help='Description for train_models')
        parser.add_argument('--time_slot', type=str, required=False, default=None, help='Description time slot')
        
        parsed_args = parser.parse_args(args)
        
        if len(args) < 3:
            exps = self.getExperimentsList(0)
            for exp in exps['experiments_list']:
                print("", exp['id'], "\t- ", exp['model_case'])
        else:
            
            self.main(
                num_case = parsed_args.num_case,
                experiment_name_suffix = parsed_args.experiment_name_suffix,
                main_folder = parsed_args.main_folder,
                repeation = parsed_args.repeation,
                optimization = parsed_args.optimization,
                load_model = parsed_args.load_model,
                train_models = parsed_args.train_models,
                time_slot = parsed_args.time_slot
            )
    
    def limit_resource(self, gb_limit):
        memory_limit = gb_limit * 1024 * 1024 * 1024  # 1 GB in byte
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        cprint(Style.BRIGHT + "|************************|" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  LIMIT RESOURCES " + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  ram       : {memory_limit} byte ({gb_limit} gb)" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  device  : {None}" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        cprint(Style.BRIGHT + "|************************|" + Style.RESET_ALL, 'magenta', attrs=["bold"])
        
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        print(f"Memory limit set to {gb_limit} GB.")
        print(f"Current limit (soft): {soft / (1024 * 1024 * 1024):.2f} GB")
        print(f"Current limit (hard): {hard / (1024 * 1024 * 1024):.2f} GB")

    
    def main(self, num_case, experiment_name_suffix, main_folder, repeation, load_model=None, time_slot=None, train_models="yes", optimization=False):
        path_folder = Path('data','neuroCorrelation_experiments',main_folder)
        experiments_name = f"{experiment_name_suffix}___{num_case}"
        
        for seed in range(0, int(repeation)):
            experiment_name = f"{experiments_name}_{seed}"
            if train_models:
                univar_count = self.experiment(num_case=num_case, main_folder=path_folder, seed=seed, experiment_name=experiment_name, optimization=optimization, load_model=load_model, time_slot=time_slot)
        
            id_experiments = int(num_case)
            experiments_list = self.getExperimentsList(seed=0, optimization= optimization)
            experiments_selected = experiments_list['experiments_list'][id_experiments]
            experiments_selected["univar_count"]
            gc.collect()


    def getExperimentsList(self,seed,optimization):
        experiments_list = [
            {"id":   0, "model_case":"autoencoder_3_copula_optimization", "epoch":{'AE':   3,'GAN':   2}, "univar_count": 7, "lat_dim": 3, "dataset_setting":{"batch_size":  32, "train_percentual":None,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":None, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},

            #METR-LA DATASET
            {"id":   1, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_16",  "epoch":{'AE':    2,'GAN' :  1},  "univar_count":16,   "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   2, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_32",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":32,   "lat_dim":28,   "dataset_setting":{"batch_size": {'AE' :  64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   3, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_48",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":48,   "lat_dim":36,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   4, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_METR_64",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":64,   "lat_dim":54,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},

            #PEMS-BAY DATASET
            {"id":   5, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_16",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":16,   "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':  128,'GAN' :  128}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   6, "case":"PEMS_METR", "model_case":"AE>WGAN_linear_pretrained_PEMS_16",  "epoch":{'AE': 150,'WGAN':  150},  "univar_count":16,   "lat_dim":12,   "dataset_setting":{"batch_size": {'AE':   64,'WGAN':   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   7, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_32",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":32,   "lat_dim":28,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   8, "case":"PEMS_METR", "model_case":"AE>WGAN_linear_pretrained_PEMS_32",  "epoch":{'AE':  10,'WGAN' :  1000},  "univar_count":32,   "lat_dim":28,   "dataset_setting":{"batch_size": {'AE':64,'WGAN' : 256}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   9, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_48",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":48,   "lat_dim":36,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   10, "case":"PEMS_METR", "model_case":"AE>GAN_linear_pretrained_PEMS_64",  "epoch":{'AE':  100,'GAN' :  100},  "univar_count":64,   "lat_dim":54,   "dataset_setting":{"batch_size": {'AE':   64,'GAN' :   64}, "train_percentual":0.6,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},                                

            #CHENGDU_SMALLGRAPH DATASET
            {"id":   11, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_16_A_linear",  "epoch":{'AE':  100,'GAN':  100},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   12, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_16_A_graph",   "epoch":{'AE':  100,'GAN':  100},  "univar_count":16,   "lat_dim":6,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   13, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_32_A_linear",  "epoch":{'AE':  100,'GAN':   50},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   14, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_32_A_graph",   "epoch":{'AE':   50,'GAN':   50},  "univar_count":32,   "lat_dim":8,   "dataset_setting":{"batch_size": {'AE':   2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   15, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_48_A_linear",  "epoch":{'AE':  100,'GAN':  100},  "univar_count":48,   "lat_dim":16,   "dataset_setting":{"batch_size": {'AE':  2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   16, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_48_A_graph",   "epoch":{'AE':   50,'GAN':   50},  "univar_count":48,   "lat_dim":16,   "dataset_setting":{"batch_size": {'AE':  2375,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   17, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_64_A_linear",  "epoch":{'AE':  100,'GAN':  100},  "univar_count":64,   "lat_dim":54,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            {"id":   18, "case":"CHENGDU_SMALLGRAPH", "model_case":"AE>GAN_CHENGDU_SMALLGRAPH_64_A_graph",   "epoch":{'AE':  100,'GAN':  100},  "univar_count":64,   "lat_dim":54,   "dataset_setting":{"batch_size": {'AE':   256,'GAN':   256}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": optimization, "input_shape":"vector"},
            
             
            '''
            {"id":  12, "model_case":"GAN_linear_pretrained_0128_CHENGDU_bt",     "epoch":{'AE': 70,'GAN': 20}, "univar_count":128,   "lat_dim":80,   "dataset_setting":{"batch_size":  2375, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":1000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                                
            {"id":  13, "model_case":"GAN_linear_pretrained_0256_CHENGDU_bt",     "epoch":{'AE': 20,'GAN': 50}, "univar_count":256,   "lat_dim":80,   "dataset_setting":{"batch_size":  2375, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":1000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                                
            {"id":  14, "model_case":"GAN_linear_pretrained_0256_CHENGDU_bt"},
            {"id":  15, "model_case":"GAN_linear_pretrained_0256_CHENGDU_bt"},
            {"id":  16, "model_case":"GAN_linear_pretrained_0256_CHENGDU_bt"},
            {"id":  17, "model_case":"GAN_GCN_linear_pretrained_0064_CHENGDU_bt", "epoch":{'AE': 50,'GAN': 50}, "univar_count":64,   "lat_dim":50,   "dataset_setting":{"batch_size":  2375, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":1000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                                
            
            {"id":  18, "model_case":"GAN_GCN_linear_pretrained_0128_CHENGDU_bt", "epoch":{'AE': 50,'GAN': 50}, "univar_count":64,   "lat_dim":50,   "dataset_setting":{"batch_size":  2645, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":1000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                                
            {"id":  19, "model_case":"GAN_GCN_linear_pretrained_RN0128_CHENGDU_bt", "epoch":{'AE': 50,'GAN': 20}, "univar_count":128,   "lat_dim":80,   "dataset_setting":{"batch_size":  2375, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":1000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                                
            {"id":  20, "model_case":"GAN_GCN_linear_pretrained_RN0742_CHENGDU_bt", "epoch":{'AE': 20,'GAN': 2}, "univar_count":742,   "lat_dim":600,   "dataset_setting":{"batch_size":  2375, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":1000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                                
            
            {"id":  21, "model_case":"GAN_GCN_linear_pretrained_URB_ZONE0_CHENGDU_bt",  "epoch":{'AE': 100,'GAN': 50},  "univar_count":248,   "lat_dim":80,   "dataset_setting":{"batch_size": {'AE': 2375,'GAN': 64}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": True, "input_shape":"vector"},                                
            {"id":  22, "model_case":"GAN_GCN_linear_pretrained_URB_ZONE1_CHENGDU_bt", "epoch":{'AE': 50,'GAN': 2}, "univar_count":240,   "lat_dim":90,   "dataset_setting":{"batch_size":  {'AE': 2375,'GAN': 64} , "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1, "input_shape":"vector"},                                
            {"id":  23, "model_case":"GAN_GCN_linear_pretrained_URB_ZONE1-2_CHENGDU_bt", "epoch":{'AE': 2,'GAN': 20}, "univar_count":437,   "lat_dim":90,   "dataset_setting":{"batch_size":  {'AE': 2375,'GAN': 64} , "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1, "optimization": False, "input_shape":"vector"},                                
                        
            {"id":  24, "model_case":"ESG__GAN_linear_pretrained_35",  "epoch":{'AE': 100,'GAN': 2},  "univar_count":35,   "lat_dim":30,   "dataset_setting":{"batch_size": {'AE': 2348,'GAN': 32}, "train_percentual":0.9,"starting_sample":None,"train_samples":None,"test_samples":None,"noise_samples":10000, "seed":seed}, "instaces_size" :1,"optimization": True, "input_shape":"vector"}
            '''
        ]
        
        optimization_settings = {
            "epochs":{'AE': 30,'GAN': 1}, "n_calls":{'AE': 50,'GAN': 1},"modeltype":"AE", "base_estimator":"GP", "n_initial_points":10, "optimization_function":"mahalanobis", 
            "search_space":[
                {"type":"Real","min":5e-4,"max":2, "values_list":None, "name":"loss", "network_part":"AE", "param":"SPEARMAN_CORRELATION_LOSS"},
                {"type":"Real","min":5e-4,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"JENSEN_SHANNON_DIVERGENCE_LOSS"},
                {"type":"Real","min":5e-4,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"MEDIAN_LOSS_batch"},
                {"type":"Real","min":5e-4,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"VARIANCE_LOSS"},
                {"type":"Real","min":5e-4,"max":2, "values_list":None, "name":"loss", "network_part":"AE",  "param":"COVARIANCE_LOSS"},
                #{"type":"Categorical","min":None, "max":None, "values_list":["Adam", "RMSprop", "SGD"], "name":"loss_optimizer", "param":""},
                ]
            }
        
        run_mode = "ALL_noscen"
        return {"experiments_list":experiments_list, "run_mode":run_mode,  "optimization_settings":optimization_settings}
        
    def experiment(self, num_case, main_folder, seed, experiment_name, optimization, load_model, time_slot=None):
        id_experiments = int(num_case)
        experiments_list = self.getExperimentsList(seed, optimization)
        experiments_selected = experiments_list["experiments_list"][id_experiments]
        
        run_mode = experiments_list["run_mode"]
        opt_settings = experiments_list['optimization_settings']
            
        folder_experiment = Path(main_folder, experiment_name)  

        #, 'autoencoder_05k_Chengdu','autoencoder_0016_Chengdu', 'autoencoder_6k_Chengdu','autoencoder_3_copula_optimization']
        cprint(Style.BRIGHT + "|------------------------|" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"| Modelcase   : {experiments_selected['model_case']}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  settings   : {experiments_selected}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  seed       : {seed}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT +f"|  time_slot  : {time_slot}" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        cprint(Style.BRIGHT + "|------------------------|" + Style.RESET_ALL, 'cyan', attrs=["bold"])
        
        
        
        '''try:'''
        nc = NeuralCore(device=None,epoch=experiments_selected["epoch"], case=experiments_selected['case'], model_case=experiments_selected["model_case"], univar_count=experiments_selected["univar_count"], lat_dim=experiments_selected["lat_dim"], dataset_setting=experiments_selected['dataset_setting'], instaces_size= experiments_selected["instaces_size"], input_shape= experiments_selected["input_shape"], do_optimization=experiments_selected["optimization"],path_folder=folder_experiment, seed=seed, run_mode=run_mode, opt_settings=opt_settings, time_slot=time_slot)
        if load_model:
            nc.start_experiment(load_model=True)
        else:
            nc.start_experiment()
        '''except Exception as e:
            print(f"Unexpected error: {e}")
        else:
            print("No exceptions were thrown.")
        finally:
            res = {"univar_count": experiments_selected["univar_count"]}
            return res'''
    
    def str2bool(self, v):
        if v in ['yes', 'true', 't', 'y', '1']:
            return True
        else:
            return False
from src.NeuroCorrelation.DataLoaders.DataSynteticGeneration import DataSynteticGeneration
from src.NeuroCorrelation.DataLoaders.DataMapsLoader import DataMapsLoader


from pathlib import Path
import os


class DataLoader:
    
    def __init__(self, mode, seed,  name_dataset, device, dataset_setting, epoch, univar_count, lat_dim, corrCoeff, instaces_size, path_folder, vc_dict=None):
        
        self.mode = mode
        self.seed = seed
        self.name_dataset = name_dataset
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
        self.corrCoeff['data'] = dict()
        self.summary_path = Path(self.path_folder,'summary')
        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)
        self.statsData = None
        self.vc_dict = vc_dict

    def dataset_load(self, draw_plots=True, save_summary=True, loss=None):
        self.loss = loss
        if self.mode=="random_var" and self.name_dataset=="3var_defined":
            print("DATASET PHASE: Sample generation")
            self.dataGenerator = DataSynteticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            
            self.dataGenerator.casualVC_init_3VC(num_of_samples = self.starting_sample, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train_data'] = self.dataGenerator.casualVC_generation(name_data="train", num_of_samples = self.train_samples, draw_plots=draw_plots)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.casualVC_generation(name_data="test", num_of_samples = self.test_samples,  draw_plots=draw_plots)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots)
            self.vc_mapping = ['X', 'Y','Z']

        if self.mode=="random_var" and self.name_dataset=="copula":
            print("DATASET PHASE: Sample copula generation")
            self.dataGenerator = DataSynteticGeneration(torch_device=self.device, univar_count=self.univar_count, lat_dim=self.lat_dim, path_folder=self.path_folder)
            
            if self.vc_dict is None:
                self.vc_dict = {"X":{"dependence":None}, "Y":{"dependence":{"X":1.6}}, "Z":{"dependence":{"X":3}}, "W":{"dependence":None},"K":{"dependence":{"W":0.5}}, "L":{"dependence":{"W":5}}, "M":{"dependence":None}}
            self.vc_mapping = list()
            for key_vc in self.vc_dict:
                self.vc_mapping.append(key_vc)
            
            self.dataGenerator.casualVC_init_multi(num_of_samples = self.starting_sample, vc_dict=self.vc_dict, draw_plots=draw_plots)
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.casualVC_generation(name_data="train", univar_count=self.univar_count, num_of_samples = self.train_samples, draw_plots=draw_plots, instaces_size=self.instaces_size)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.casualVC_generation(name_data="test", univar_count=self.univar_count, num_of_samples = self.test_samples,  draw_plots=draw_plots, instaces_size=self.instaces_size)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots, instaces_size=self.instaces_size)
        
        if self.mode =="graph_roads":
            print("DATASET PHASE: Load maps data")
            
            self.dataGenerator = DataMapsLoader(torch_device=self.device, seed=self.seed, name_dataset=self.name_dataset, lat_dim=self.lat_dim, univar_count=self.univar_count, path_folder=self.path_folder)
            self.dataGenerator.mapsVC_load(train_percentual=self.train_percentual, draw_plots=draw_plots)
            
            
            train_data, self.corrCoeff['data']['train'] = self.dataGenerator.mapsVC_getData(name_data="train", draw_plots=draw_plots)
            test_data, self.corrCoeff['data']['test'] = self.dataGenerator.mapsVC_getData(name_data="test",  draw_plots=draw_plots)
            noise_data = self.dataGenerator.get_synthetic_noise_data(name_data="noise", num_of_samples = self.noise_samples, draw_plots=draw_plots)
            self.vc_mapping = self.dataGenerator.get_vc_mapping()

        if self.mode=="graph_statics":
            print("to implement")
            #self.train_data = self.dataGenerator.graphGen(num_of_samples = train_samples, with_cov=True)

        self.rangeData = self.dataGenerator.getDataRange()
        self.statsData = self.dataGenerator.getDataStats()
        
        reduced_noise_data = self.generateNoiseRedux(1)
        
        data_dict = {"train_data":train_data, "test_data":test_data, "noise_data":noise_data, "reduced_noise_data":reduced_noise_data}
        
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
    
    def getRangeData(self):
        if self.rangeData is None:
            raise Exception("rangeData not defined.")
        return self.rangeData
        
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
        settings_list.append(f"mode_dataset:: {self.epoch}") 
         
        for key in self.dataset_setting:
            print("saveDataset_setting\t",key)
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
        print("SETTING PHASE: Summary dataset file - DONE")
        
    def generateNoiseRedux(self, high):
        redux_noise = list()
        redux_noise_values = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
        redux_noise_values = [-1,  -0.5,  0,  0.5,  1]
        c = self.generateNoisePercentile(high, redux_noise_values)
        noise_redux_samples = list()
        #for c_item in c:
        #    noise_redux_samples.append({'sample': torch.Tensor(c_item).to(device=self.device), 'noise': torch.Tensor(c_item).to(device=self.device)})
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
        
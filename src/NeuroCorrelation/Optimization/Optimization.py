import os
from pathlib import Path

import optuna
from optuna.trial import TrialState
from src.NeuroCorrelation.ModelTraining import ModelTraining

class Optimization():
    def __init__(self, model, device, data_dict, model_type, epoch, loss, path_folder, univar_count, batch_size, dataGenerator, instaces_size_noise=None, direction="maximize", timeout=600):
        self.model = model
        self.device = device
        self.train_data = data_dict['train_data']
        self.model_type = model_type
        self.epoch = epoch
        self.loss_obj = loss
        self.path_folder = path_folder
        self.univar_count = univar_count
        self.batch_size = batch_size
        self.dataGenerator = dataGenerator
        self.direction = direction
        self.timeout = timeout
        self.load_model = False
        
        self.run = False
        self.study = None
        
        self.training_obj = ModelTraining(model=self.model, device=self.device, loss_obj=self.loss_obj, epoch=self.epoch, dataset=self.train_data, dataGenerator=self.dataGenerator, path_folder=self.path_folder, univar_count = self.univar_count, model_type=self.model_type, pre_trained_decoder=False)
        

    def optimize(self, n_trials):
        print("\tOPTIMIZATION: create_study")
        self.study = optuna.create_study(direction=self.direction)
        print("\tOPTIMIZATION: optimize start")
        self.study.optimize(self.opt_model, n_trials=n_trials, timeout=self.timeout)
        print("\tOPTIMIZATION: optimize end")
        
    
    def opt_model(self, trial):
        print("\tOPTIMIZATION:\tTrue")
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        print("\tOPTIMIZATION NAME\t",optimizer_name)
        
        print("TRAIN")
        return 1

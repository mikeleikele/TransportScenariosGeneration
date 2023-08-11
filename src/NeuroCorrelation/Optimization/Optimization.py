import os
from pathlib import Path
from src.NeuroCorrelation.ModelTraining.ModelTraining import  ModelTraining
from skopt import Optimizer
from skopt.plots import plot_gaussian_process
from skopt.space import Space, Real, Categorical, Integer
from skopt import gp_minimize
from skopt.utils import point_asdict
from skopt.plots import plot_gaussian_process

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
        
        self.opt = None
        
        self.search_space = {"keys":list(),"space":list()}
        print("OPTIMIZATION PHASE:")
        
        

    def set_searchSpace(self, search_space):
        
        for space_vals in search_space:
            space = None
            key = None
            if space_vals["type"]== "Categorical":
                space = Categorical(space_vals["values_list"], name=space_vals["name"])
            elif space_vals["type"]== "Integer":
                space = Integer(space_vals["min"],space_vals["max"], name=space_vals["name"])
            elif space_vals["type"]== "Real":
                space = Real(space_vals["min"],space_vals["max"], name=space_vals["name"])
            else:
                print("space not recornized")
            if space is not None:  
                self.search_space["keys"].append(space_vals["name"])
                self.search_space["space"].append(space)
            
    def set_optimizer(self, base_estimator="GP", n_initial_points=10):
        
        #"GP", "RF", "ET", "GBRT"
        print("\tcreate optimizer")
        self.opt = Optimizer(dimensions=self.search_space["space"], base_estimator=base_estimator, n_initial_points=n_initial_points, acq_optimizer="sampling")
        
        
    
    def optimization(self, n_trials):
        print("\tbegin optimization")
        for trial in range(n_trials):
            print("\t\ttrial -\t#",trial)
            next_x = self.opt.ask()
            print("\t\t\tpoint values: ",next_x)
            for key, val in zip(self.search_space["keys"], next_x): 
                self.loss_obj.loss_change_coefficent(key, val)
            
            self.training_obj = ModelTraining(model=self.model, device=self.device, loss_obj=self.loss_obj, epoch=self.epoch, dataset=self.train_data, dataGenerator=self.dataGenerator, path_folder=self.path_folder, univar_count = self.univar_count, model_type=self.model_type, pre_trained_decoder=False)
            if self.model_type =="AE":
                optim_score = self.training_obj.training(batch_size=self.batch_size, model_flatten_in=True,load_model=False, optimizar_trial=trial)
            elif self.model_type =="GAN":
                optim_score = self.training_obj.training(batch_size=self.batch_size, noise_size=1, load_model=False)
            self.training_obj.eval()
            print("\t\t\tpoint score: ",optim_score)
            self.opt.tell(next_x, optim_score)
        
        print("OPTIMIZATION RESULT:")  
        print(self.opt.get_result())

        #plot_gaussian_process(self.opt.get_result(), **plot_args) 
        print("\t: end optimization")
        
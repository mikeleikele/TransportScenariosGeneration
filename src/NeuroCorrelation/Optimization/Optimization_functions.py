
import numpy as np
import pandas as pd
from numpy.linalg import inv
from termcolor import cprint 

class Optimization_functions():
    
    def __init__(self):
        self.object_list = ['mahalanobis']
        self.name_fun = None
        
    def set_object_fun(self, name_fun):
        if name_fun in self.object_list:
            self.name_fun = name_fun
        else:
            raise Exception("Optimization function not exist.")
        cprint(f"OPTIMIZATION OBJECT FUN: ", "red", end="\n")
        cprint(f"\t{name_fun}", "red", end="\n")
    
    def get_score(self, values):
        score = None
        if self.name_fun is None:
            raise Exception("No optimization function selected")
        elif self.name_fun == "mahalanobis":
            val_real  = values['inp_data_vc'].to_numpy()
            val_gen = values['out_data_vc'].to_numpy()
            dist_mahalanobis = self.mahalanobis(X=val_real, Y=val_gen)
            score = dist_mahalanobis
        else:
            raise Exception(f"Optimization function {self.name_fun} non developed")
        cprint(f"Optimization function score:\t{score}", "green", end="\n")
        return score
    
    def mahalanobis(self, X, Y):
        mu_X = np.mean(X, axis=0)
        mu_Y = np.mean(Y, axis=0)
        
        cov_X = np.cov(X, rowvar=False)
        cov_Y = np.cov(Y, rowvar=False)
        
        cov_combined = (cov_X + cov_Y) / 2
        
        diff = mu_X - mu_Y
        dist_mahalanobis = np.sqrt(diff.T @ inv(cov_combined) @ diff)
        return dist_mahalanobis
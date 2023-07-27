import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from scipy.stats import norm
import statistics
from pathlib import Path
import os
import pandas as pd
import math
import random

import scipy.stats as stats
from matplotlib import cm # for a scatter plot
from src.tool.utils_matplot import SeabornFig2Grid
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from numpy import dot
from numpy.linalg import norm

class DataComparison():

    def __init__(self, univar_count_in, univar_count_out, dim_latent, path_folder):
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out

        self.dim_latent = dim_latent
        self.path_folder = path_folder
    
    def get_idName(self, id_univar, max_val):
        num_digit = len(str(max_val))
        name_univ = f'{id_univar:0{num_digit}d}'
        return name_univ

    def data_comparison_plot(self, data, plot_name=None, mode="in"):
        if plot_name is not None:
            fold_name = f"{plot_name}_distributions_compare"
        else:
            fold_name = f"univar_distribution_compare"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        stats_dict = {"univ_id": []}

        if mode=="in":
            n_var = self.univar_count_in
        elif mode=="out":
            n_var = self.univar_count_out

        for id_univar in range(n_var):
            name_univ = self.get_idName(id_univar, max_val=n_var)
            stats_dict['univ_id'].append(name_univ)
            plt.figure(figsize=(12,8))  
            for key in data:

                list_values = data[key]['data'][id_univar]
                color_data = data[key]['color']
                if 'alpha' in data[key]:
                    alpha_data = data[key]['alpha']
                else:
                    alpha_data = 0.2
                name_data = key
                mean_val = np.mean(list_values)
                var_val = np.var(list_values)
                std_val = np.std(list_values)
                mean_label = f"{name_data}_mean"
                std_label = f"{name_data}_var"
                
                title_txt = f"{plot_name} - vc: {id_univar}"
                plt.title(title_txt)

                plt.axvline(x = mean_val, linestyle="solid", color = color_data, label = mean_label)
                plt.axvline(x = (mean_val-std_val), linestyle="dashed", color = color_data, label = std_label)
                plt.axvline(x = (mean_val+std_val), linestyle="dashed", color = color_data, label = std_label)

                mean_plt_txt = f"      {name_data} mean: {mean_val:.3f}"
                plt.text(mean_val, 0, s=mean_plt_txt, rotation = 90) 

                #histogram
                label_hist_txt = f"{name_data}"
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), histtype='stepfilled', alpha = alpha_data, color= color_data, label=label_hist_txt)
                #histogram border
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), histtype=u'step', edgecolor="gray", fc="None", lw=1)

                plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

                if mean_label not in stats_dict:
                    stats_dict[mean_label]= list()
                stats_dict[mean_label].append(mean_val)


                if std_label not in stats_dict:
                    stats_dict[std_label]= list()
                stats_dict[std_label].append(std_val)
            
            plt.legend(loc='upper right')
            filename = Path(path_fold_dist,"plot_vc_distribution_"+plot_name+"_"+name_univ+".png")
            plt.savefig(filename)

        filename = Path(path_fold_dist,fold_name+"_table.csv")
        stats_dict = pd.DataFrame.from_dict(stats_dict)
        stats_dict.to_csv(filename, sep='\t', encoding='utf-8')

    def latent_comparison_plot(self, data_lat, path_fold_dist, plot_name=None, color_data="green"):
        stats_dict = {"univ_id": []}
        
        for id_comp in range(self.dim_latent):
            list_values = data_lat[id_comp]
            name_comp = self.get_idName(id_comp, max_val=self.dim_latent)
            plt.figure(figsize=(12,8))
            plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), histtype='stepfilled', alpha = 0.2, color= color_data)
            title_txt = f"{plot_name} - component: {id_comp}"
            plt.title(title_txt)
            filename = Path(path_fold_dist, "plot_latent_distribution_"+plot_name+"_"+name_comp+"_latent.png")
            plt.savefig(filename)
    
    def plot_latent_analysis(self, data_lat, plot_name, color_data="green"):    
        _keys = list(data_lat.keys())
        
        
        if plot_name is not None:
            fold_name = f"{plot_name}_latent_distribution"
        else:
            fold_name = f"univar_latent_distribution"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        self.latent_comparison_plot(data_lat, path_fold_dist,plot_name)
        if len(_keys)<=10:
            if len(_keys)>=7:
                marginal_plot = False
            else:
                marginal_plot = True

            fig = plt.figure(figsize=(18,18))
            gs = gridspec.GridSpec(self.dim_latent, self.dim_latent)
            id_sub = 0
            for i,key_i in enumerate(data_lat):
                i_val = [x.tolist() for x in data_lat[i]]
                i_key = f"{i}_comp"
                
                for j,key_j in enumerate(data_lat):
                    j_val = [x.tolist() for x in data_lat[j]]
                    j_key = f"{j}_comp"
                    if  i != j:
                        plt_cor = self.correlation_plot(i_val, j_val, i_key, j_key, color=color_data, marginal_dist=marginal_plot)
                        mg0 = SeabornFig2Grid(plt_cor, fig, gs[id_sub])
                    else:
                        ax_sub = fig.add_subplot(gs[id_sub])
                        plt_variance = self.variance_plot(i_val, i_key, ax_sub, color=color_data)
                    id_sub += 1
                    
            gs.tight_layout(fig)
            filename = Path(path_fold_dist, "plot_lat_correlation_grid_+"+plot_name+".png")
            plt.savefig(filename)

    def plot_vc_correlationCoeff(self, df_data, plot_name, is_latent=False, corrMatrix=None):
        if is_latent:
            if plot_name is not None:
                fold_name = f"{plot_name}_latent_distribution"
            else:
                fold_name = f"univar_latent_distribution"

            path_fold_dist = Path(self.path_folder, fold_name)
            if not os.path.exists(path_fold_dist):
                os.makedirs(path_fold_dist)
        else:
            path_fold_dist = self.path_folder

        fig = plt.figure(figsize=(18,18))
        if corrMatrix is None:
            corrMatrix = self.correlationCoeff(df_data)
        rho = corrMatrix
        
        corrcoef_Path = Path(path_fold_dist, "data_pearson_pearson_"+plot_name+".csv")
        np.savetxt(corrcoef_Path, rho, delimiter=",")

        fig = plt.figure(figsize=(18,18))
        sns.heatmap(rho, annot = True,square=True, vmin=-1, vmax=1)
        filename = Path(path_fold_dist,"plot_pearson_pearson_"+plot_name+".png")
        plt.savefig(filename)
        

    def correlationCoeff(self, df_data):    
        rho_val_list = list()
        for key_vc in df_data:
            rho_val_list.append(df_data[key_vc])
        rho = np.corrcoef(rho_val_list)
        return rho

    
    def plot_vc_analysis(self, df_data, plot_name, mode="in", color_data="blue"):
        _keys = list(df_data.keys())        
        
        fig = plt.figure(figsize=(18,18))
        if mode=="in":
            n_row = self.univar_count_in
        else:
            n_row = self.univar_count_out
        gs = gridspec.GridSpec(n_row, n_row)
        _key = _keys[0]
        if len(_keys)<=10:
            if len(_keys)>=7:
                marginal_plot = False
            else:
                marginal_plot = True
            n_items = len(df_data[_key])
            if n_items>500:
                item_selected = random.choices([i for i in range(500)], k=500)
            else:
                item_selected = [i for i in range(n_items)]

            id_sub = 0
            for i, key_i in enumerate(df_data):
                i_val_all = df_data[key_i]
                i_val = []#
                for l in item_selected:
                    item = i_val_all[l]
                    if isinstance(item, list):
                        i_val.append(item[0])
                    else:
                        i_val.append(item)

                i_key = f"{key_i}"
                for j, key_j in enumerate(df_data):
                    j_val_all = df_data[key_j]
                    j_val = []  
                    for l in item_selected:
                        item = j_val_all[l]
                        if isinstance(item, list):
                            j_val.append(item[0])
                        else:
                            j_val.append(item)

                    
                    j_key = f"{key_j}"
                    print(i_key, j_key)
                    if  i != j:
                        plt_cor = self.correlation_plot(i_val, j_val, i_key, j_key, color=color_data, marginal_dist=marginal_plot)
                        mg0 = SeabornFig2Grid(plt_cor, fig, gs[id_sub])
                    else:
                        ax_sub = fig.add_subplot(gs[id_sub])
                        plt_variance = self.variance_plot(i_val, i_key, ax_sub, color=color_data)
                    id_sub += 1
            print(247)
            gs.tight_layout(fig)
            filename = Path(self.path_folder, "plot_vc_correlation_grid_"+plot_name+".png")
            plt.savefig(filename)

    def correlation_plot(self, rand1, rand2, name1, name2, color="blue", marginal_dist=True):


        if isinstance(rand1, list) and isinstance(rand2, list):
            min1, max1, min2, max2 = min(rand1), max(rand1), min(rand2), max(rand2)

        else:
            min1, max1, min2, max2 = rand1.min(), rand1.max(), rand2.min(), rand2.max()

        h = sns.jointplot(x=rand1, y=rand2, space=0, xlim=(min1, max1), ylim=(min2, max2),color=color, kind="hist")
        if marginal_dist:
            h.set_axis_labels(name1, name2, fontsize=16)
            h.ax_marg_x.set_axis_off()
            h.ax_marg_y.set_axis_off()
        return h

    def variance_plot(self, rand1, name1, ax_sub, color="blue"):
        h = sns.histplot(data=rand1, kde = True, ax=ax_sub, color=color)
        title_txt = f"Histogram of {name1}"
        h.set_title(title_txt, fontsize=16)        
        return h


class CorrelationComparison():

    def __init__(self, correlation_matrices, folder):
        self.dict_matrices = correlation_matrices
        self.path_fold = Path(folder,"correlation_comparison")
        if not os.path.exists(self.path_fold):
            os.makedirs(self.path_fold)
        
    
    def compareMatrices(self, list_comparisons):
        df = pd.DataFrame()
        for (key_a,key_b) in list_comparisons:
            frobenius_val = self.frobenius_norm(key_a,key_b)
            spearmanr_val = self.spearmanr(key_a,key_b)
            cosin_sim_val = self.cosineSimilarity(key_a,key_b)

            new_row = {'matrix_A':self.keyToSting(key_a), 'matrix_B':self.keyToSting(key_b), 'frobenius':frobenius_val,'spearmanr_statistic':spearmanr_val[0],'spearmanr_pvalue':spearmanr_val[1], 'cosineSimilarity':cosin_sim_val}
            df = df.append(new_row, ignore_index=True)
        csv_path = Path(self.path_fold, 'correlation_comparison.csv')
        df.to_csv(csv_path)

    def keyToSting(self, key):
        key_0 = key[0]
        key_1 = key[1]
        return f"{key_0}__{key_1}"

    def get_matrix(self, key):
        key_0 = key[0]
        key_1 = key[1]        

        return self.dict_matrices[key_0][key_1]

    def frobenius_norm(self, key_a, key_b):
        matrix1 = self.get_matrix(key_a)
        matrix2 = self.get_matrix(key_b)
        diff_matrix = matrix1 - matrix2
        squared_diff = np.square(diff_matrix)
        sum_squared_diff = np.sum(squared_diff)
        frobenius_norm = np.sqrt(sum_squared_diff)
        return frobenius_norm

    def spearmanr(self, key_a, key_b):
        matrix1 = self.get_matrix(key_a)
        matrix2 = self.get_matrix(key_b) 
        matrix1_top = self.upper(matrix1)
        matrix2_top = self.upper(matrix2)
        significance = stats.spearmanr(matrix1_top, matrix2_top)
        return significance

    def cosineSimilarity(self, key_a, key_b):
        matrix1 = self.get_matrix(key_a)
        matrix2 = self.get_matrix(key_b)
        matrix1_flat = np.concatenate(matrix1).ravel()
        matrix2_flat = np.concatenate(matrix2).ravel()
        cos_sim = dot(matrix1_flat, matrix2_flat)/(norm(matrix1_flat)*norm(matrix2_flat))
        return cos_sim

    def upper(self, df):
        '''Returns the upper triangle of a correlation matrix.
        You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
        Args:
        df: pandas or numpy correlation matrix
        Returns:
        list of values from upper triangle
        '''
        try:
            assert(type(df)==np.ndarray)
        except:
            if type(df)==pd.DataFrame:
                df = df.values
            else:
                raise TypeError('Must be np.ndarray or pd.DataFrame')
        mask = np.triu_indices(df.shape[0], k=1)
        return df[mask]

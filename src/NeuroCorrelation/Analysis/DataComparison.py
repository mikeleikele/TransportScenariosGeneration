import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
#from scipy.stats import norm
import statistics
from pathlib import Path
import os
import pandas as pd
import math
import random

import torch
import scipy.stats as stats
from matplotlib import cm # for a scatter plot
#from src.tool.utils_matplot import SeabornFig2Grid
import matplotlib.gridspec as gridspec
import seaborn as sns
from numpy import dot
from numpy.linalg import norm

import warnings
from scipy.stats import wasserstein_distance
from copulas.multivariate import GaussianMultivariate
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy.linalg import inv, pinv
from scipy.linalg import sqrtm

import time

class DataComparison():

    def __init__(self, univar_count_in, univar_count_out, latent_dim, path_folder):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.path_folder = path_folder

    
    def get_idName(self, id_univar, max_val):
        num_digit = len(str(max_val))
        name_univ = f'{id_univar:0{num_digit}d}'
        return name_univ

    def data_comparison_plot(self, data, plot_name=None, mode="in", is_npArray=True):
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
                plt.xlim([0, 1])
                plt.ylim([0, 1])

                if is_npArray:
                    list_values = [x.tolist() for x in data[key]['data'][id_univar]]
                else:
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


                bins = np.linspace(0.0, 1.0, 100)

                #histogram
                label_hist_txt = f"{name_data}"
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), bins=bins, histtype='stepfilled', alpha = alpha_data, color= color_data, label=label_hist_txt)
                #histogram border
                plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), bins=bins, histtype=u'step', edgecolor="gray", fc="None", lw=1)

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

    def latent_comparison_distribution_plot(self, data_lat, path_fold_dist, plot_name=None, color_data="green"):
        stats_dict = {"univ_id": []}
        
        for id_comp in range(self.latent_dim):
            list_values = [x.tolist() for x in data_lat[id_comp]]
            name_comp = self.get_idName(id_comp, max_val=self.latent_dim)
            plt.figure(figsize=(12,8))
            list_values_weights = np.ones(len(list_values)) / len(list_values)
            plt.hist(list_values, weights=list_values_weights, histtype='stepfilled', alpha = 0.2, color= color_data)
            title_txt = f"{plot_name} - component: {id_comp}"
            plt.title(title_txt)
            filename = Path(path_fold_dist, "plot_latent_distribution_"+plot_name+"_"+name_comp+"_latent.png")
            plt.savefig(filename)
    
    def plot_latent_analysis(self, data_lat, plot_name, color_data="green"):    
        # The above code snippet is not doing anything. It contains a comment `# Python` followed by
        # an undefined variable `_keys` and then another comment `
        _keys = list(data_lat.keys())
        n_keys = len(_keys)
        n_row = n_keys
        
        if plot_name is not None:
            fold_name = f"{plot_name}_latent_distribution"
        else:
            fold_name = f"univar_latent_distribution"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        self.latent_comparison_distribution_plot(data_lat, path_fold_dist,plot_name)
        
        fig_size_factor = 2 + n_keys
        fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
        
        gs = gridspec.GridSpec(n_row, n_row)
        _key = _keys[0]
        
        if len(_keys)>=7:
            marginal_plot = False
        else:
            marginal_plot = True
        
        n_items = len(data_lat[_key])
        
        if n_items>1000:
            item_selected = random.choices([i for i in range(1000)], k=1000)
        else:
            item_selected = [i for i in range(n_items)]
        
        for i,key_i in enumerate(data_lat):
            i_val = [x.tolist() for x in data_lat[i]]
            i_key = f"{i}_comp"
                
            for j,key_j in enumerate(data_lat):
                j_val = [x.tolist() for x in data_lat[j]]
                j_key = f"{j}_comp"
                if i<=j:
                    id_sub = (i*n_keys)+j
                    ax_sub = fig.add_subplot(gs[id_sub])
                    if  i != j:
                        self.correlation_plot(i_val, j_val, i_key, j_key, ax_sub, color=color_data, marginal_dist=marginal_plot)
                    else:
                        self.variance_plot(i_val, i_key, ax_sub, color=color_data)
                    
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
        
        return
        for key in rho:
            csvFile_Path = Path(path_fold_dist, f"data_{key}_"+plot_name+".csv")
            np.savetxt(csvFile_Path, rho[key], delimiter=",")
            
            fig = plt.figure(figsize=(18,18))
            sns.heatmap(rho[key], annot = True, square=True, vmin=-1, vmax=1, cmap= 'coolwarm')
            filename = Path(path_fold_dist,f"plot_{key}_"+plot_name+".png")
            plt.savefig(filename)
        

    def correlationCoeff(self, df_data, select_subset=True, num_to_select = 200):    
        rho_val_list = list()
        for key_vc in df_data:
            if isinstance(df_data[key_vc][0], (np.ndarray)):
                vc_values = [value.tolist() for value in df_data[key_vc]]
                rho_val_list.append(vc_values)
            else:
                rho_val_list.append(df_data[key_vc])
        
        data_df = pd.DataFrame(rho_val_list).T
        if select_subset:
            num_to_selected = min(num_to_select, data_df.shape[0])
            data_df = data_df.sample(num_to_selected, random_state=0)
        
        rho = dict()
        rho['pearson'] = data_df.corr(method='pearson').values
        rho['spearman'] = data_df.corr(method='spearman').values
        rho['kendall'] = data_df.corr(method='kendall').values
        rho['covar'] = np.cov(data_df.T, bias=False)
        
        return rho

    
    def plot_vc_analysis(self, df_data, plot_name, mode="in", color_data="blue"):
        
        _keys = list(df_data.keys())        
        n_keys = len(_keys)
        
        if mode=="in":
            n_row = self.univar_count_in
        else:
            n_row = self.univar_count_out
        
        fig_size_factor = 2 + n_keys
        fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
        
        gs = gridspec.GridSpec(n_row, n_row)
        _key = _keys[0]
        
        if len(_keys)>=7:
            marginal_plot = False
        else:
            marginal_plot = True
            
        n_items = len(df_data[_key])
        
        if n_items>1000:
                item_selected = random.choices([i for i in range(1000)], k=1000)
        else:
            item_selected = [i for i in range(n_items)]

        for i, key_i in enumerate(df_data):
            i_val = [df_data[key_i][l][0] if isinstance(df_data[key_i][l], list) else df_data[key_i][l] for l in item_selected]
            
            for j, key_j in enumerate(df_data):
                j_val = [df_data[key_j][l][0] if isinstance(df_data[key_j][l], list) else df_data[key_j][l] for l in item_selected]
                if i<=j:
                    id_sub = (i*n_keys)+j
                    ax_sub = fig.add_subplot(gs[id_sub])
                    if  i != j:
                        self.correlation_plot(i_val, j_val, f"{key_i}", f"{key_j}", ax_sub, color=color_data, marginal_dist=marginal_plot)
                    else:
                        self.variance_plot(i_val, f"{key_i}", ax_sub, color=color_data)
        
        gs.tight_layout(fig)
        filename = Path(self.path_folder, "plot_vc_correlation_grid_"+plot_name+".png")
        plt.savefig(filename)
    
    def sub_reverse(self, i, n_var):
        col = i % n_var
        row = i // n_var
        j = (col * n_var) + row
        return j

    def correlation_plot(self, rand1, rand2, name1, name2, ax_sub, color="blue", marginal_dist=True):
        if isinstance(rand1, list) and isinstance(rand2, list):
            min1, max1, min2, max2 = min(rand1), max(rand1), min(rand2), max(rand2)
        else:
            min1, max1, min2, max2 = rand1.min(), rand1.max(), rand2.min(), rand2.max()

        sns.histplot(x=rand1, y=rand2, bins=30, ax=ax_sub, color=color)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
        
        ax_sub.set_xlim(min1, max1)
        ax_sub.set_ylim(min2, max2)
        
        if marginal_dist:
            ax_marg_x = ax_sub.inset_axes([0, 1.05, 1, 0.2], sharex=ax_sub)
            ax_marg_y = ax_sub.inset_axes([1.05, 0, 0.2, 1], sharey=ax_sub)
            sns.histplot(rand1, ax=ax_marg_x, color=color, kde=False)
            sns.histplot(rand2, ax=ax_marg_y, color=color, kde=False)  # NOTA: usa y per il margine verticale
        return ax_sub

    def variance_plot(self, rand1, name1, ax_sub, color="blue"):
        h = sns.histplot(data=rand1, kde = True, ax=ax_sub, color=color)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
        ax_sub.set_ylabel('')
        ax_sub.set_xlabel('')
        return h
    
    def find_nearest_kde(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
     
    def draw_point_overDistribution(self, plotname, folder, n_var, points,  distr, n_sample = 1000):
        if distr is None:
            distr = list()
            for i in range(n_sample):
                mu, sigma = 0, math.sqrt(1) # mean and standard deviation
                s = np.random.normal(mu, sigma, n_var)  
                distr.append({'sample': torch.Tensor(s), 'noise': torch.Tensor(s)})
        

        fig = plt.figure(figsize=(18,18))
        n_col = math.ceil(math.sqrt(n_var))
        n_row = math.ceil(n_var/n_col)  

        gs = gridspec.GridSpec(n_row,n_col)
        distr_dict = dict()
        points_dict = dict()
    
        for i in range(n_var):
            distr_dict[i] = list()
            points_dict[i] = list()

        for sample in distr:
            for i in range(n_var):
                distr_dict[i].append(float(sample['sample'][i].cpu().numpy()))
            
        for sample in points:
            for i in range(n_var):
                points_dict[i].append(float(sample['sample'][i].cpu().numpy()))


        for id_sub in range(n_var):
            ax_sub = fig.add_subplot(gs[id_sub])
            h = sns.histplot(data=np.array(distr_dict[id_sub]), kde = True, element="step", ax=ax_sub, alpha=0.3)
            point_list = list()

            x = ax_sub.lines[0].get_xdata()
            y = ax_sub.lines[0].get_ydata()

            points = list(zip(x, y))
            t_dic = dict(points)

            lls = 1
            for sample in points_dict[id_sub]:
                true_x = sample
                x_point = self.find_nearest_kde(np.array(list(t_dic.keys())), true_x)

                sns.scatterplot(x = [x_point],y = [t_dic[x_point]], s=50)
                ax_sub.text(x_point+.02, t_dic[x_point], str(lls))
                lls += 1

        gs.tight_layout(fig)
        filename = Path(folder, f"{plotname}.png")
        plt.savefig(filename)


class DataComparison_Advanced():
    
    def __init__(self, univar_count, input_folder, suffix_input, time_performance, gen_copula=True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        
        self.TSNE_components = 2
        self.PCA_components = 50
        self.univar_count = univar_count
        self.n_sample_considered = 1000
        self.gen_copula = gen_copula
        self.copula_test = False
        self.time_performance = time_performance
        
        self.loadPrediction_INPUT(input_folder, suffix_input)
        
        
        
    def loadPrediction_INPUT(self, input_folder, suffix_input):        
        self.rand_var_in = dict()        
        self.rand_var_cop = dict()
        for i in range(self.univar_count):
            self.rand_var_in[i] = list()
            self.rand_var_cop[i] = list()
        
        
        #input used to train
        input_instances = Path(input_folder,f"{suffix_input}.csv")
        input_data = pd.read_csv(input_instances)
        
        if self.copula_test:
            j_range = 5
        else:
            j_range = len(input_data['x_input'])
        print("copula in_vect values:\t",j_range)
        for j in range(j_range):
            res = input_data['x_input'][j].strip('][').split(', ')
            for i in range(self.univar_count):
                self.rand_var_in[i].append(float(res[i]))
        print("\tload truth data: done")
        if self.gen_copula:
            self.genCopula()
        
        #output used to compare
    
    def genCopula(self):
        real_data = pd.DataFrame.from_dict(self.rand_var_in)
        print(f"\tfit gaussian copula data: start")
        self.time_performance.start_time("COPULA_TRAINING")
        copula = GaussianMultivariate()        
        copula.fit(real_data)
        self.time_performance.stop_time("COPULA_TRAINING")
        print(f"\tfit gaussian copula data: end")
        cop_train_time = self.time_performance.get_time("COPULA_TRAINING", fun="last")
        self.time_performance.compute_time("COPULA_TRAINING", fun = "first") 
        
        print(f"\tTIME fit gaussian copula data:\t",cop_train_time)
        self.n_sample_considered = self.n_sample_considered
        self.time_performance.start_time("COPULA_GENERATION")
        synthetic_data = copula.sample(self.n_sample_considered)
        
        self.time_performance.stop_time("COPULA_GENERATION")
        cop_gen_time = self.time_performance.get_time("COPULA_GENERATION", fun="last")
        self.time_performance.compute_time("COPULA_GENERATION", fun = "first") 
        
        
        
        print(f"\tTIME gen gaussian copula data:\t",cop_gen_time)
        print(f"\tgenerate gaussian copula data: done \t({self.n_sample_considered} instances)")
        
        for i in range(self.univar_count):
            self.rand_var_cop[i] = synthetic_data[i].tolist()

    def loadPrediction_OUTPUT(self, output_folder, suffix_output):
        self.path_folder = output_folder
        self.suffix = suffix_output
        self.rand_var_out = dict()
        for i in range(self.univar_count):
            self.rand_var_out[i] = list()     
        output_instances = Path(output_folder,f"prediced_instances_{suffix_output}.csv")
        output_data = pd.read_csv(output_instances)
        for j in range(len(output_data['x_output'])):
            res = output_data['x_output'][j].strip('][').split(', ')
            for i in range(self.univar_count):
                self.rand_var_out[i].append(float(res[i]))
        
        
    def comparison_measures(self, measures):
        if 'wasserstein_dist' in measures:
            self.comparison_wasserstein()
        if 'mahalanobis_dist' in measures:
            self.comparison_mahalanobis()
        if 'frechet_inception_dist' in measures:
            self.comparison_frechet_inception()
        if 'tsne_plots' in measures:
            self.comparison_tsne()
          
    def comparison_wasserstein(self): 
        print("\t\twasserstein measure")
        wass_values_ae = dict()
        if self.gen_copula:
            wass_values_cop = dict()
        mean_ae = list()
        mean_cop = list()
        for i in range(self.univar_count):
            dist_real = self.rand_var_in[i]
            dist_fake = self.rand_var_out[i]
            wd_ae = wasserstein_distance(dist_real,dist_fake)
            wass_values_ae[i] = wd_ae
            mean_ae.append(wd_ae)
            
            if self.gen_copula:
                dist_copu = self.rand_var_cop[i]
                wd_cop = wasserstein_distance(dist_real,dist_copu)
                wass_values_cop[i] = wd_cop
                mean_cop.append(wd_cop)

        wass_values_ae['mean'] = np.mean(mean_ae)        
        ws_pd_ae = pd.DataFrame(wass_values_ae.items())
        
        columns=["variable","mean_real","std_real","mean_AE","std_AE"]
        if self.gen_copula:
            wass_values_cop['mean'] = np.mean(mean_cop)
            ws_pd_cop = pd.DataFrame(wass_values_cop.items())
            columns.append("mean_COP")
            columns.append("std_COP")
        pd_stats = pd.DataFrame(columns=columns)
        
        for i in range(self.univar_count):
            in_conf = stats.t.interval( df=len(self.rand_var_in[i])-1, loc=np.mean(self.rand_var_in[i]), scale=stats.sem(self.rand_var_in[i]), confidence=0.90)
            in_mean = np.mean(self.rand_var_in[i])
            in_std = np.std(self.rand_var_in[i])
                
            out_conf = stats.t.interval( df=len(self.rand_var_out[i])-1, loc=np.mean(self.rand_var_out[i]), scale=stats.sem(self.rand_var_out[i]), confidence=0.90)
            out_mean = np.mean(self.rand_var_out[i])
            out_std = np.std(self.rand_var_out[i])
            
            variable_dict = {'variable' : i,'mean_real' : in_mean, 'std_real':in_std, 'mean_AE' : out_mean, 'std_AE':out_std,'diff_real_ea': abs(in_mean-out_mean)}
            
            
            if self.gen_copula:
                cop_conf = stats.t.interval( df=len(self.rand_var_cop[i])-1, loc=np.mean(self.rand_var_cop[i]), scale=stats.sem(self.rand_var_cop[i]), confidence=0.90)
                cop_mean = np.mean(self.rand_var_cop[i])
                cop_std = np.std(self.rand_var_cop[i])
                variable_dict['mean_COP'] = cop_mean
                variable_dict['std_COP'] = cop_std
                variable_dict['diff_real_cop'] = abs(in_mean-cop_mean)                

            pd_stats = pd_stats.append(variable_dict, ignore_index = True)

        filename = Path(self.path_folder, f"wasserstein_compare_{self.suffix}.csv")
        pd_stats.to_csv(filename)
        return pd_stats
            
    
    def invert_matrix(self, matrix):
        try:
            # Check if the matrix is singular by computing its determinant
            if np.linalg.det(matrix) == 0:
                print("The matrix is singular and cannot be inverted.")
                return None
            
            # Compute the inverse if the matrix is not singular
            inv_matrix = np.linalg.inv(matrix)
            return inv_matrix
        
        except np.linalg.LinAlgError as e:
            print(f"Error: {e}")
            return None


    def comparison_mahalanobis(self):
        np_dist_real = pd.DataFrame.from_dict(self.rand_var_in).to_numpy()
        np_dist_gen = pd.DataFrame.from_dict(self.rand_var_out).to_numpy()
        np_dist_cop = pd.DataFrame.from_dict(self.rand_var_cop).to_numpy()
        
        mahala_real_gen = self.mahalanobis(np_dist_real, np_dist_gen)
        print("mahala_real_gen:\t",mahala_real_gen)
        
        mahala_real_cop = self.mahalanobis(np_dist_real, np_dist_cop)
        print("mahala_real_cop:\t",mahala_real_cop)
        
        mahalanobis_dict = {'mahalanobis_real_gen':[mahala_real_gen], 'mahalanobis_real_cop':[mahala_real_cop]}
        pd_stats = pd.DataFrame(mahalanobis_dict)
        
        filename = Path(self.path_folder, f"mahalanobis_compare_{self.suffix}.csv")
        pd_stats.to_csv(filename)
        return pd_stats
        
    def mahalanobis(self, X, Y):
        mu_X = np.mean(X, axis=0)
        mu_Y = np.mean(Y, axis=0)
            
        cov_X = np.cov(X, rowvar=False)
        cov_Y = np.cov(Y, rowvar=False)
        
        cov_combined = (cov_X + cov_Y) / 2
        
        diff = mu_X - mu_Y
        
        if np.linalg.det(cov_combined) == 0:
            print("\t\tThe covariance matrix is singular. Using the pseudo-inverse.")
            inv_cov = np.linalg.pinv(cov_combined)
        else:
            inv_cov = np.linalg.inv(cov_combined)

        dist_mahalanobis = np.sqrt(diff.T @ inv_cov @ diff)
        
        return dist_mahalanobis    
    
    def comparison_frechet_inception(self):
        np_dist_real = pd.DataFrame.from_dict(self.rand_var_in).to_numpy()
        np_dist_gen = pd.DataFrame.from_dict(self.rand_var_out).to_numpy()
        np_dist_cop = pd.DataFrame.from_dict(self.rand_var_cop).to_numpy()
        
        frechet_real_gen = self.frechet_inception_distance(np_dist_real,np_dist_gen)
        print("frechet_inception_real_gen:\t", frechet_real_gen)
        
        frechet_real_cop = self.frechet_inception_distance(np_dist_real, np_dist_cop)
        print("frechet_inception_real_cop:\t",frechet_real_cop)
        
        frechet_dict = {'frechet_inception_real_gen':[frechet_real_gen], 'frechet_inception_real_cop':[frechet_real_cop]}
        pd_stats = pd.DataFrame(frechet_dict)
        
        filename = Path(self.path_folder, f"frechet_inception_compare_{self.suffix}.csv")
        pd_stats.to_csv(filename)
        return pd_stats
    
    def frechet_inception_distance(self, real_samples, generated_samples):
        mu_real = np.mean(real_samples, axis=0)
        mu_generated = np.mean(generated_samples, axis=0)

        sigma_real = np.cov(real_samples, rowvar=False)
        sigma_generated = np.cov(generated_samples, rowvar=False)

        diff = mu_real - mu_generated
        diff_squared = np.sum(diff**2)
        
        covmean, _ = sqrtm(sigma_real @ sigma_generated, disp=False)
    
        # numerical adjust
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff_squared + np.trace(sigma_real + sigma_generated - 2 * covmean)
        return fid
    
    
    def comparison_tsne(self, n_points=None):
        color_list = {"real": (0.122, 0.467, 0.706),"ae":(1.0, 0.498, 0.055)}
        label_list = {"real": "real data","ae":"GAN+AE gen"}
        
            
        df_tsne = pd.DataFrame()
        if n_points == None:
            n_points = self.n_sample_considered
        
        
        len_sample_array = [len(self.rand_var_in[0]), len(self.rand_var_out[0])]
        
        if self.gen_copula:
            color_list["cop"] = (0.173, 0.627, 0.173)
            label_list["cop"] = "copula gen"
            len_sample_array.append(len(self.rand_var_cop[0]))
        
        if n_points > min(len_sample_array):
            n_points = min(len_sample_array)
        
        print("\t\ttsne plot #points:\t",n_points)
        real_indeces =  [i for i in range(len(self.rand_var_in[0]))]
        selected = random.sample(real_indeces, n_points)
        real_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_in[0]))]
        
        neur_indeces =  [i for i in range(len(self.rand_var_out[0]))]
        selected = random.sample(neur_indeces, n_points)
        neur_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_out[0]))]
        
        if self.gen_copula:
            copu_indeces =  [i for i in range(len(self.rand_var_cop[0]))]
            selected = random.sample(copu_indeces, n_points)
            copu_selected = [1 if i in selected else 0 for i in range(len(self.rand_var_cop[0]))]
        
        
        for i in range(self.univar_count):
            n_real = 0
            real_val = list()
            for j in range(len(self.rand_var_in[i])):
                if real_selected[j]==1:
                    real_val.append(self.rand_var_in[i][j])
                    n_real += 1
                    
            
            n_neur = 0
            neur_val = list()        
            for j in range(len(self.rand_var_out[i])):
                if neur_selected[j]==1:
                    neur_val.append(self.rand_var_out[i][j])
                    n_neur += 1
                    
            comp_values = real_val + neur_val
            
            if self.gen_copula:
                copu_val = list()
                n_copu = 0
                for j in range(len(self.rand_var_cop[i])):
                    if copu_selected[j]==1:
                        copu_val.append(self.rand_var_cop[i][j])
                        n_copu += 1
                        
                comp_values += copu_val
            
            df_tsne[f'c_{i}'] = comp_values
                    
        labels =  ["real" for k in range(n_real)] + ["ae" for k in range(n_neur)] 
        if self.gen_copula:
            labels += ["cop" for k in range(n_copu)]
        
        tsne = TSNE(n_components=self.TSNE_components)
        tsne_results = tsne.fit_transform(df_tsne)
        dftest = pd.DataFrame(tsne_results)
        dftest['label'] = labels
        
        
        
        fig = plt.figure(figsize=(16,7))
        params = {1: {'color': 'k', 'label': 'Pass'},
            0: {'color': 'r', 'label': 'Fail'}}


        sns.scatterplot(
            x=0, y=1,
            hue="label",
            palette=color_list,
            data=dftest,
            alpha=0.2,
            legend="full",
        )
        filename = Path(self.path_folder, f"TSNE_plot_{self.suffix}.png")
        plt.savefig(filename)
        
        fig, axs = plt.subplots(figsize=(140,20), ncols=self.univar_count)
        df_tsne['label'] = labels
        
        df_a = df_tsne.loc[df_tsne['label'] == "real"].iloc[0:400]
        df_b = df_tsne.loc[df_tsne['label'] == "ae"].iloc[0:400]
        if self.gen_copula:
            df_c = df_tsne.loc[df_tsne['label'] == "cop"].iloc[0:400]
            df_swarmplot= pd.concat([df_a, df_b, df_c])
        else:
            df_swarmplot= pd.concat([df_a, df_b])
        
        for i in range(self.univar_count):
            sns.violinplot(data=df_tsne[[f'c_{i}',"label"]], y=f'c_{i}', x="label", palette=color_list, hue="label", ax=axs[i])
            sns.swarmplot(data=df_swarmplot[[f'c_{i}',"label"]], y=f'c_{i}', x="label",palette=color_list, size=3, ax=axs[i])
        
        filename = Path(self.path_folder, f"SWARM_plot_{self.suffix}.png")
        fig.savefig(filename)        

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

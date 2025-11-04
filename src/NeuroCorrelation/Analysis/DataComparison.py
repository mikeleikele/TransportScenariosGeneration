import matplotlib.pyplot as plt
from termcolor import cprint
from colorama import init, Style
from matplotlib.ticker import PercentFormatter
import numpy as np
import statistics
from pathlib import Path
import os
import pandas as pd
import math
import random

import torch
import scipy.stats as stats
from matplotlib import cm
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pacmap
import umap
from scipy.stats import wasserstein_distance
from scipy.stats import binned_statistic_dd

import time
from scipy.stats import binned_statistic_2d
from sklearn.metrics import silhouette_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import adjusted_rand_score

from scipy.stats import gaussian_kde
from scipy.integrate import quad
import dask.array as da
import dask.dataframe as dd

from termcolor import cprint
from colorama import Style
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import random
from scipy import stats
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import mahalanobis
from scipy.stats.kde import gaussian_kde
from scipy.linalg import sqrtm
from scipy.stats import binned_statistic_2d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import cprint
from colorama import Style
import umap
import pacmap
from copulas.multivariate import GaussianMultivariate
import os


class DataComparison():

    def __init__(self, univar_count_in, univar_count_out, latent_dim, path_folder, key_value_list, name_key):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.path_folder = path_folder
        self.np_dist = dict()
        self.name_key = name_key
        self.key_value_list = key_value_list  # Lista di canali/chiavi
    
    def get_idName(self, id_univar, max_val):
        num_digit = len(str(max_val))
        name_univ = f'{id_univar:0{num_digit}d}'
        return name_univ

    def data_comparison_plot(self, data, plot_name=None, mode="in", is_npArray=True):
        """
        Genera plot di confronto tra distribuzioni per dati multi-canale.
        
        Args:
            data: dict con struttura {dataset_name: {'data': {channel: {var_id: [samples]}}, 'color': str, 'alpha': float}}
            plot_name: nome base per i plot
            mode: 'in' o 'out'
            is_npArray: se i dati sono numpy array o liste
        """
        if plot_name is not None:
            fold_name = f"{plot_name}_distributions_compare"
        else:
            fold_name = f"univar_distribution_compare"

        # Cartella base
        path_fold_dist_base = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist_base):
            os.makedirs(path_fold_dist_base)
        
        # Determina numero di variabili
        if mode == "in":
            n_var = self.univar_count_in
        elif mode == "out":
            n_var = self.univar_count_out

        # Itera su ogni canale
        for channel_idx, channel_key in enumerate(self.key_value_list):
            # Crea cartella specifica per questo canale
            path_fold_dist = Path(path_fold_dist_base, channel_key)
            if not os.path.exists(path_fold_dist):
                os.makedirs(path_fold_dist)
            
            stats_dict = {"univ_id": []}
            
            # Per ogni variabile/timestep
            for id_univar in range(n_var):
                name_univ = self.get_idName(id_univar, max_val=n_var)
                stats_dict['univ_id'].append(name_univ)
                plt.figure(figsize=(12, 8))
                
                for key in data:
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    
                    # Estrai valori per questo canale e questa variabile
                    list_values = []
                    
                    # Accedi ai dati: data[dataset]['data'][channel][variable]
                    
                    samples = data[key]['data'][channel_key][id_univar]
                    
                    for sample in samples:
                        if is_npArray and isinstance(sample, np.ndarray):
                            # Il sample è già il valore scalare per questo canale
                            list_values.append(float(sample))
                        elif isinstance(sample, (list, tuple)):
                            # Se è lista/tupla, prendi l'elemento channel_idx
                            if len(sample) > channel_idx:
                                list_values.append(sample[channel_idx])
                        else:
                            # Valore scalare
                            list_values.append(float(sample))
                    
                    color_data = data[key]['color']
                    alpha_data = data[key].get('alpha', 0.2)
                    name_data = key
                    mean_val = np.mean(list_values)
                    var_val = np.var(list_values)
                    std_val = np.std(list_values)
                    mean_label = f"{name_data}_mean"
                    std_label = f"{name_data}_var"
                    
                    title_txt = f"{plot_name} - Channel: {channel_key} - var: {id_univar}"
                    plt.title(title_txt)

                    plt.axvline(x=mean_val, linestyle="solid", color=color_data, label=mean_label)
                    plt.axvline(x=(mean_val - std_val), linestyle="dashed", color=color_data, label=std_label)
                    plt.axvline(x=(mean_val + std_val), linestyle="dashed", color=color_data, label=std_label)

                    mean_plt_txt = f"      {name_data} mean: {mean_val:.3f}"
                    plt.text(mean_val, 0, s=mean_plt_txt, rotation=90)

                    bins = np.linspace(0.0, 1.0, 100)

                    # histogram
                    label_hist_txt = f"{name_data}"
                    plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), 
                            bins=bins, histtype='stepfilled', alpha=alpha_data, color=color_data, label=label_hist_txt)
                    # histogram border
                    plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), 
                            bins=bins, histtype=u'step', edgecolor="gray", fc="None", lw=1)

                    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

                    if mean_label not in stats_dict:
                        stats_dict[mean_label] = list()
                    stats_dict[mean_label].append(mean_val)

                    if std_label not in stats_dict:
                        stats_dict[std_label] = list()
                    stats_dict[std_label].append(std_val)
                
                plt.legend(loc='upper right')
                # Salva nella cartella del canale
                filename = Path(path_fold_dist, f"plot_vc_distribution_{plot_name}_var{name_univ}.png")
                plt.savefig(filename)
                plt.close()
                plt.cla()
                plt.clf()

            # Salva statistiche nella cartella del canale
            filename = Path(path_fold_dist, f"{fold_name}_table.csv")
            stats_dict = pd.DataFrame.from_dict(stats_dict)
            stats_dict.to_csv(filename, sep='\t', encoding='utf-8')
            

    def latent_comparison_distribution_plot(self, data_lat, path_fold_dist, plot_name=None, color_data="green"):
        stats_dict = {"univ_id": []}
        for id_comp in range(self.latent_dim):
            list_values = [x.tolist() if isinstance(x, (np.ndarray, torch.Tensor)) else x for x in data_lat[id_comp]]
            name_comp = self.get_idName(id_comp, max_val=self.latent_dim)
            plt.figure(figsize=(12,8))
            list_values_weights = np.ones(len(list_values)) / len(list_values)
            plt.hist(list_values, weights=list_values_weights, histtype='stepfilled', alpha=0.2, color=color_data)
            title_txt = f"{plot_name} - component: {id_comp}"
            plt.title(title_txt)
            filename = Path(path_fold_dist, "plot_latent_distribution_"+plot_name+"_"+name_comp+"_latent.png")
            plt.savefig(filename)
            plt.close()
            plt.cla()
            plt.clf()
    
    def plot_latent_analysis(self, data_lat, plot_name, color_data="green"):    
        _keys = list(data_lat.keys())
        n_keys = len(_keys)
        n_row = n_keys
        
        if plot_name is not None:
            fold_name = f"{plot_name}_distribution"
        else:
            fold_name = f"univar_latent_distribution"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        self.latent_comparison_distribution_plot(data_lat, path_fold_dist, plot_name)

    def plot_latent_corr_analysis(self, data_lat, plot_name, color_data="green"):            
        _keys = list(data_lat.keys())        
        n_keys = len(_keys)
        n_row = n_keys
        
        if plot_name is not None:
            fold_name = f"{plot_name}_distribution"
        else:
            fold_name = f"univar_latent_distribution"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
            
        fig_size_factor = 2 + n_keys
        fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
        
        gs = gridspec.GridSpec(n_row, n_row)
        _key = _keys[0]
        
        marginal_plot = len(_keys) < 7
        
        n_items = len(data_lat[_key])
        
        if n_items > 1000:
            item_selected = random.choices([i for i in range(n_items)], k=1000)
        else:
            item_selected = [i for i in range(n_items)]
        
        for i, key_i in enumerate(data_lat):
            i_val = [x.tolist() if isinstance(x, (np.ndarray, torch.Tensor)) else x for x in data_lat[i]]
            i_key = f"{i}_comp"
                
            for j, key_j in enumerate(data_lat):
                j_val = [x.tolist() if isinstance(x, (np.ndarray, torch.Tensor)) else x for x in data_lat[j]]
                j_key = f"{j}_comp"
                if i <= j:
                    id_sub = (i * n_keys) + j
                    ax_sub = fig.add_subplot(gs[id_sub])
                    if i != j:
                        self.correlation_plot(i_val, j_val, i_key, j_key, ax_sub, color=color_data, marginal_dist=marginal_plot)
                    else:
                        self.variance_plot(i_val, i_key, ax_sub, color=color_data)
                    
        gs.tight_layout(fig)
        filename = Path(path_fold_dist, "plot_lat_correlation_grid_"+plot_name+".png")
        plt.savefig(filename)
        plt.close()
        plt.cla()
        plt.clf()

    def plot_vc_correlationCoeff(self, df_data, plot_name, is_latent=False, corrMatrix=None):
        """
        Genera plot delle matrici di correlazione per dati multi-canale.
        
        Args:
            df_data: DataFrame con colonne che contengono liste/array multi-canale
            plot_name: nome per i file di output
            is_latent: se i dati sono latenti
            corrMatrix: matrice di correlazione pre-calcolata (opzionale)
        """
        if is_latent:
            if plot_name is not None:
                fold_name = f"{plot_name}_distribution"
            else:
                fold_name = f"univar_latent_distribution"
        else:
            fold_name = f"{plot_name}_distribution"
            
        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)

        if corrMatrix is None:
            corrMatrix = self.correlationCoeff(df_data)
        
        rho = corrMatrix
        
        fig_size_factor = 2 + int(len(df_data.keys()))
        
        # Per ogni canale/chiave
        for key in rho:
            # Per ogni tipo di correlazione
            for corr_type in rho[key]:
                csvFile_Path = Path(path_fold_dist, f"data_{key}_{corr_type}_{plot_name}.csv")
                np.savetxt(csvFile_Path, rho[key][corr_type], delimiter=",")
                
                fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
                sns.heatmap(rho[key][corr_type], annot=True, square=True, vmin=-1, vmax=1, cmap='coolwarm')
                plt.title(f"{key} - {corr_type}")
                filename = Path(path_fold_dist, f"plot_{key}_{corr_type}_{plot_name}.png")
                plt.savefig(filename)
                plt.close()
                plt.cla()
                plt.clf()

    def correlationCoeff(self, df_data, select_subset=True, num_to_select=200):
        """
        Calcola coefficienti di correlazione per dati multi-canale.
        
        Args:
            df_data: DataFrame dove ogni cella contiene un array/lista di valori multi-canale
            select_subset: se sottocampionare i dati
            num_to_select: numero di campioni da selezionare
            
        Returns:
            dict: {channel_key: {'pearson': matrix, 'spearman': matrix, ...}}
        """
        rho = dict()
        
        # Per ogni canale
        for idx, key in enumerate(self.key_value_list):
            rho_val_list = []
            
            # Per ogni colonna (variabile) del DataFrame
            for col in df_data.columns:
                channel_values = []
                for row_data in df_data[col]:
                    # Estrai il valore per questo canale
                    if isinstance(row_data, (list, tuple, np.ndarray)):
                        if len(row_data) > idx:
                            channel_values.append(row_data[idx])
                    else:
                        # Valore scalare (compatibilità)
                        channel_values.append(float(row_data))
                
                rho_val_list.append(channel_values)
            
            # Crea DataFrame per questo canale
            data_df = pd.DataFrame(rho_val_list).T
            
            if select_subset:
                num_to_selected = min(num_to_select, data_df.shape[0])
                data_df = data_df.sample(num_to_selected, random_state=0)
            
            rho[key] = dict()
            rho[key]['pearson'] = data_df.corr(method='pearson').values
            rho[key]['spearman'] = data_df.corr(method='spearman').values
            rho[key]['kendall'] = data_df.corr(method='kendall').values
            rho[key]['covar'] = np.cov(data_df.T, bias=False)
        
        return rho
    
    def plot_vc_analysis(self, df_data, plot_name, mode="in", color_data="blue"):
        """
        Genera plot di analisi per dati multi-canale con grid di correlazioni.
        
        Args:
            df_data: DataFrame con dati multi-canale
            plot_name: nome per il file di output
            mode: 'in' o 'out'
            color_data: colore per i plot
        """
        _keys = list(df_data.keys())        
        n_keys = len(_keys)
        
        if mode == "in":
            n_row = self.univar_count_in
        else:
            n_row = self.univar_count_out
        
        fig_size_factor = 2 + n_keys
        fig = plt.figure(figsize=(fig_size_factor, fig_size_factor))
        
        gs = gridspec.GridSpec(n_row, n_row)
        _key = _keys[0]
        
        marginal_plot = len(_keys) < 7
            
        n_items = len(df_data[_key])
        
        if n_items > 1000:
            item_selected = random.choices([i for i in range(n_items)], k=1000)
        else:
            item_selected = [i for i in range(n_items)]
        
        for i, key_i in enumerate(df_data):
            # Estrai valori per il primo canale (indice 0)
            i_val = []
            for l in item_selected:
                val = df_data[key_i][l]
                if isinstance(val, (list, tuple, np.ndarray)):
                    i_val.append(val[0] if len(val) > 0 else val)
                else:
                    i_val.append(float(val))
            
            for j, key_j in enumerate(df_data):
                j_val = []
                for l in item_selected:
                    val = df_data[key_j][l]
                    if isinstance(val, (list, tuple, np.ndarray)):
                        j_val.append(val[0] if len(val) > 0 else val)
                    else:
                        j_val.append(float(val))
                
                if i <= j:
                    id_sub = (i * n_keys) + j
                    ax_sub = fig.add_subplot(gs[id_sub])
                    if i != j:
                        self.correlation_plot(i_val, j_val, f"{key_i}", f"{key_j}", ax_sub, color=color_data, marginal_dist=marginal_plot)
                    else:
                        self.variance_plot(i_val, f"{key_i}", ax_sub, color=color_data)
        
        gs.tight_layout(fig)
        filename = Path(self.path_folder, "plot_vc_correlation_grid_"+plot_name+".png")
        plt.savefig(filename)
        plt.close()
        plt.cla()
        plt.clf()
    
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
            sns.histplot(rand2, ax=ax_marg_y, color=color, kde=False)
        return ax_sub

    def variance_plot(self, rand1, name1, ax_sub, color="blue"):
        h = sns.histplot(data=rand1, kde=True, ax=ax_sub, color=color)
        ax_sub.set_xticks([])
        ax_sub.set_yticks([])
        ax_sub.set_ylabel('')
        ax_sub.set_xlabel('')
        return h
    
    def find_nearest_kde(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
     
    def draw_point_overDistribution(self, plotname, folder, n_var, points, distr, n_sample=1000):
        if distr is None:
            distr = list()
            for i in range(n_sample):
                mu, sigma = 0, math.sqrt(1)
                s = np.random.normal(mu, sigma, n_var)  
                distr.append({'sample': torch.Tensor(s), 'noise': torch.Tensor(s)})

        fig = plt.figure(figsize=(18, 18))
        n_col = math.ceil(math.sqrt(n_var))
        n_row = math.ceil(n_var / n_col)  

        gs = gridspec.GridSpec(n_row, n_col)
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
            h = sns.histplot(data=np.array(distr_dict[id_sub]), kde=True, element="step", ax=ax_sub, alpha=0.3)
            point_list = list()

            x = ax_sub.lines[0].get_xdata()
            y = ax_sub.lines[0].get_ydata()

            points = list(zip(x, y))
            t_dic = dict(points)

            lls = 1
            for sample in points_dict[id_sub]:
                true_x = sample
                x_point = self.find_nearest_kde(np.array(list(t_dic.keys())), true_x)

                sns.scatterplot(x=[x_point], y=[t_dic[x_point]], s=50)
                ax_sub.text(x_point + .02, t_dic[x_point], str(lls))
                lls += 1

        gs.tight_layout(fig)
        filename = Path(folder, f"{plotname}.png")
        plt.savefig(filename)
        plt.close()
        plt.cla()
        plt.clf()

    def data_comparison_plot_nochannels(self, data, plot_name=None, mode="in", is_npArray=True):
        """
        Genera plot di confronto tra distribuzioni per dati multi-canale.
        
        Args:
            data: dict con struttura {dataset_name: {'data': {var_id: [samples]}, 'color': str, 'alpha': float}}
            plot_name: nome base per i plot
            mode: 'in' o 'out'
            is_npArray: se i dati sono numpy array o liste
        """
        if plot_name is not None:
            fold_name = f"{plot_name}_distributions_compare"
        else:
            fold_name = f"univar_distribution_compare"

        path_fold_dist = Path(self.path_folder, fold_name)
        if not os.path.exists(path_fold_dist):
            os.makedirs(path_fold_dist)
        
        # Determina numero di variabili
        if mode == "in":
            n_var = self.univar_count_in
        elif mode == "out":
            n_var = self.univar_count_out

        # Itera su ogni canale
        for channel_idx, channel_key in enumerate(self.key_value_list):
            stats_dict = {"univ_id": []}
            
            # Per ogni variabile/timestep
            for id_univar in range(n_var):
                name_univ = self.get_idName(id_univar, max_val=n_var)
                stats_dict['univ_id'].append(name_univ)
                plt.figure(figsize=(12, 8))
                
                for key in data:
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    
                    # Estrai valori per questo canale e questa variabile
                    list_values = []
                    for sample in data[key]['data'][id_univar]:
                        if is_npArray and isinstance(sample, np.ndarray):
                            # Se sample è un array, prendi l'elemento channel_idx
                            if len(sample.shape) > 0 and len(sample) > channel_idx:
                                list_values.append(sample[channel_idx])
                        elif isinstance(sample, (list, tuple)):
                            # Se è lista/tupla, prendi l'elemento channel_idx
                            if len(sample) > channel_idx:
                                list_values.append(sample[channel_idx])
                        else:
                            # Valore scalare (compatibilità con codice vecchio)
                            list_values.append(float(sample))
                    
                    color_data = data[key]['color']
                    alpha_data = data[key].get('alpha', 0.2)
                    name_data = key
                    mean_val = np.mean(list_values)
                    var_val = np.var(list_values)
                    std_val = np.std(list_values)
                    mean_label = f"{name_data}_mean"
                    std_label = f"{name_data}_var"
                    
                    title_txt = f"{plot_name} - Channel: {channel_key} - var: {id_univar}"
                    plt.title(title_txt)

                    plt.axvline(x=mean_val, linestyle="solid", color=color_data, label=mean_label)
                    plt.axvline(x=(mean_val - std_val), linestyle="dashed", color=color_data, label=std_label)
                    plt.axvline(x=(mean_val + std_val), linestyle="dashed", color=color_data, label=std_label)

                    mean_plt_txt = f"      {name_data} mean: {mean_val:.3f}"
                    plt.text(mean_val, 0, s=mean_plt_txt, rotation=90)

                    bins = np.linspace(0.0, 1.0, 100)

                    # histogram
                    label_hist_txt = f"{name_data}"
                    plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), 
                            bins=bins, histtype='stepfilled', alpha=alpha_data, color=color_data, label=label_hist_txt)
                    # histogram border
                    plt.hist(list_values, weights=np.ones(len(list_values)) / len(list_values), 
                            bins=bins, histtype=u'step', edgecolor="gray", fc="None", lw=1)

                    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

                    if mean_label not in stats_dict:
                        stats_dict[mean_label] = list()
                    stats_dict[mean_label].append(mean_val)

                    if std_label not in stats_dict:
                        stats_dict[std_label] = list()
                    stats_dict[std_label].append(std_val)
                
                plt.legend(loc='upper right')
                filename = Path(path_fold_dist, f"plot_vc_distribution_{plot_name}_{channel_key}_var{name_univ}.png")
                plt.savefig(filename)
                plt.close()
                plt.cla()
                plt.clf()

            # Salva statistiche per ogni canale
            filename = Path(path_fold_dist, f"{fold_name}_{channel_key}_table.csv")
            stats_dict = pd.DataFrame.from_dict(stats_dict)
            stats_dict.to_csv(filename, sep='\t', encoding='utf-8')



class DataComparison_Advanced():
    
    def __init__(self, univar_count, input_folder, suffix_input, time_performance, 
                 data_metadata, name_key, key_value_list, use_copula=True, 
                 load_copula=False, copulaData_filename=None):
        """
        Versione multi-canale compatibile con DataMapsLoader
        Gestisce un dizionario separato per ogni chiave in key_value_list
        
        Args:
            univar_count: numero di variabili univariate per canale
            key_value_list: lista di canali/chiavi (es. ['speed', 'flow', 'occupancy'])
            input_folder: cartella contenente i CSV separati per chiave
            Altri parametri invariati
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
        np.seterr(divide='ignore')
        warnings.simplefilter(action='ignore', category=FutureWarning)
        
        self.name_key = name_key
        self.univar_count = univar_count
        self.key_value_list = key_value_list  # Lista di canali
        self.n_channels = len(key_value_list)
        self.n_sample_considered = 5000
        self.use_copula = use_copula
        self.load_copula = load_copula
        self.copulaData_filename = copulaData_filename
        self.copula_test = False
        self.time_performance = time_performance
        self.alredy_select_data = False
        self.metrics_pd = None
        self.suffix_input = suffix_input
        # Dizionari separati per ogni chiave
        self.np_dist = {key_name: dict() for key_name in self.key_value_list}
        
        # Carica i dati multi-canale
        self.loadPrediction_INPUT(input_folder, suffix_input)
        
        self.color_list = {"real": (0.122, 0.467, 0.706)}
        self.label_list = {"real": "real data"}
        self.comparisons = dict()
        
        if self.use_copula:
            self.color_list["cop"] = (0.173, 0.627, 0.173)
            self.label_list["cop"] = "copula gen"
            self.comparisons["real_cop"] = {"a": "real", "b": "cop"}
        
        for item in data_metadata:
            self.color_list[item['acronym']] = item['color']
            self.label_list[item['acronym']] = item['label']
            self.comparisons[f"real_{item['acronym']}"] = {"a": "real", "b": item['acronym']}
        
        

    def loadPrediction_INPUT(self, input_folder, suffix_input):
        """
        Carica dati multi-canale compatibili con DataMapsLoader
        Ogni key_value_name ha il suo CSV separato
        """
        # Dizionari separati per ogni chiave
        self.rand_var_in = {key_name: dict() for key_name in self.key_value_list}
        self.rand_var_cop = {key_name: dict() for key_name in self.key_value_list}
        cprint(f"{Style.BRIGHT}| loaded IN : {suffix_input}", 'cyan', attrs=['bold'])
        # Carica CSV per ogni chiave
        for key_value_name in self.key_value_list:
            # Inizializza dizionario per questa chiave
            for i in range(self.univar_count):
                self.rand_var_in[key_value_name][i] = list()
                self.rand_var_cop[key_value_name][i] = list()
            
            # Path del CSV per questa chiave
            datasplit_path = Path(input_folder, key_value_name)
            input_instances = Path(datasplit_path, f"{suffix_input}_{key_value_name}.csv")
            
            if not input_instances.exists():
                # Fallback al formato originale se non esiste la struttura datasplit
                input_instances = Path(input_instances)
            #cprint(f"{Style.BRIGHT}| Loading input data {key_value_name}:\t {input_instances}{Style.RESET_ALL}", 'cyan', attrs=['bold'])
            
            if not input_instances.exists():
                print(f"  WARNING: File not found, skipping {key_value_name}")
                continue
                
            input_data = pd.read_csv(input_instances)
            
            # Parsing dei dati
            for j in range(len(input_data['x_input'])):
                x_input_str = input_data['x_input'][j]
                
                # Parse della lista
                if isinstance(x_input_str, str):
                    res = x_input_str.strip('[]').replace(' ', '').split(',')
                else:
                    # Già una lista
                    res = x_input_str
                
                # Ogni riga ha univar_count valori
                for i in range(min(self.univar_count, len(res))):
                    value_str = str(res[i])
                    if value_str.startswith("np.float32"):
                        value_str = value_str.replace("np.float32(", "").replace(")", "")
                    try:
                        value = float(value_str)
                        self.rand_var_in[key_value_name][i].append(value)
                    except ValueError:
                        print(f"  Warning: Could not parse value '{value_str}' at row {j}, var {i}")
                        continue
            cprint(f"{Style.BRIGHT}|\t{key_value_name}: {len(self.rand_var_in[key_value_name][0])} samples\t {self.univar_count} variables{Style.RESET_ALL}", 'cyan', attrs=['bold'])
        
        
        
        # Converti in numpy array per ogni chiave
        for key_value_name in self.key_value_list:
            self.np_dist[key_value_name]['real'] = pd.DataFrame.from_dict(
                self.rand_var_in[key_value_name]
            ).to_numpy()
        
        
        if self.use_copula:
            self.genCopula(input_folder, suffix_input)

            for key_value_name in self.key_value_list:
                copula_dict = self.rand_var_cop.get(key_value_name, {})
                lengths = [len(copula_dict[i]) for i in range(self.univar_count) if i in copula_dict]

                if len(set(lengths)) != 1:
                    print(f"⚠️ Incoerenza in copula '{key_value_name}': lunghezze variabili = {lengths}")
                    continue

                df_copula = pd.DataFrame.from_dict(copula_dict)
                self.np_dist[key_value_name]['cop'] = df_copula.to_numpy()

                print(f"{Style.BRIGHT}\033[38;2;121;212;242m| Copula loaded for '{key_value_name}': shape {df_copula.shape}{Style.RESET_ALL}")


    def genCopula(self, input_folder, suffix_input):
        """Genera dati copula per ogni chiave separatamente"""
        cprint(f"{Style.BRIGHT}| Copula data over {suffix_input}", 'cyan', attrs=['bold'])
        for key_value_name in self.key_value_list:
            cprint(Style.BRIGHT + f"|\tGaussian Copula for {key_value_name}" + Style.RESET_ALL, 'magenta', attrs=["bold"])

            
            if self.load_copula and self.copulaData_filename:
                # Carica copula pre-generata
                copula_file = Path(input_folder, f"{self.copulaData_filename}_{key_value_name}.csv")
                if copula_file.exists():
                    cprint(Style.BRIGHT + f"| \t\tLoad from {copula_file}" + Style.RESET_ALL, 'magenta', attrs=["bold"])
                    synthetic_data = pd.read_csv(copula_file)
                    new_columns = list(range(len(synthetic_data.columns)))
                    synthetic_data.columns = new_columns
                else:
                    print(f"  Copula file not found, will generate new one")
                    self.load_copula = False
            
            if not self.load_copula:
                real_data = pd.DataFrame.from_dict(self.rand_var_in[key_value_name])
                if self.copula_test:
                    j_range = min(5, len(self.rand_var_in[key_value_name][0]))
                else:
                    j_range = len(self.rand_var_in[key_value_name][0])
                
                cprint(Style.BRIGHT + f"| \t\tFitted on {j_range} instances" + Style.RESET_ALL, 'magenta', attrs=["bold"])
                
                real_data = real_data[:j_range]
                cprint(Style.BRIGHT + f"| \t\tFitting START" + Style.RESET_ALL, 'magenta', attrs=["bold"])


                self.time_performance.start_time(f"COPULA_TRAINING_{key_value_name}")
                copula = GaussianMultivariate()        
                copula.fit(real_data)
            
                self.time_performance.stop_time(f"COPULA_TRAINING_{key_value_name}")
                cprint(Style.BRIGHT + f"| \t\tFitting STOP" + Style.RESET_ALL, 'magenta', attrs=["bold"])

                cop_train_time = self.time_performance.get_time(f"COPULA_TRAINING_{key_value_name}", fun="last")
                self.time_performance.compute_time(f"COPULA_TRAINING_{key_value_name}", fun="first") 
            
                print(f"{Style.BRIGHT}\033[38;2;121;212;242m| \t\ttime \tfit gaussian copula data:\t{cop_train_time}{Style.RESET_ALL}")
                self.time_performance.start_time(f"COPULA_GENERATION_{key_value_name}")
                synthetic_data = copula.sample(self.n_sample_considered)
            
                self.time_performance.stop_time(f"COPULA_GENERATION_{key_value_name}")
                cop_gen_time = self.time_performance.get_time(f"COPULA_GENERATION_{key_value_name}", fun="last")
                self.time_performance.compute_time(f"COPULA_GENERATION_{key_value_name}", fun="first") 
            
                copula_instances_folder = Path(input_folder, f"copula_gen_instances_{key_value_name}.csv")
                synthetic_data.to_csv(copula_instances_folder)

                print(f"{Style.BRIGHT}\033[38;2;121;212;242m| \t\ttime \tgen gaussian copula data:\t{cop_gen_time}{Style.RESET_ALL}")
                cprint(Style.BRIGHT + f"| \tfit {key_value_name}: STOP" + Style.RESET_ALL, 'magenta', attrs=["bold"])


                cprint(Style.BRIGHT + f"| \t{self.n_sample_considered} instances generated for {key_value_name}: saved in {copula_instances_folder}" + Style.RESET_ALL, 'magenta', attrs=["bold"])

            # Popola rand_var_cop per questa chiave
            for i in range(self.univar_count):
                self.rand_var_cop[key_value_name][i] = synthetic_data[i].tolist()
        
    
    def loadPrediction_OUTPUT(self, output_folder, suffix_output, key):
        """
        Carica predizioni output per ogni chiave separatamente, compatibile con DataMapsLoader.
        Supporta parsing robusto di "[array([...])]" e fallback automatico.
        """
        self.path_folder = output_folder
        self.suffix = suffix_output

        # Inizializza dizionari
        self.rand_var_out = {key_name: {i: [] for i in range(self.univar_count)} for key_name in self.key_value_list}
        self.rand_var_cop = {key_name: {i: [] for i in range(self.univar_count)} for key_name in self.key_value_list}

        cprint(f"{Style.BRIGHT}| loaded OUT : {self.suffix}", 'cyan', attrs=['bold'])

        for key_value_name in self.key_value_list:
            # Path primario e fallback
            datasplit_path = Path(output_folder, 'datasplit', key_value_name)
            output_instances = Path(datasplit_path, f"datasplit_{suffix_output}_{key}_{key_value_name}.csv")

            if not output_instances.exists():
                output_instances = Path(output_folder, f"prediced_instances_{suffix_output}_{key_value_name}.csv")

            if not output_instances.exists():
                print(f"  WARNING: Output file not found, skipping {key_value_name}")
                continue

            output_data = pd.read_csv(output_instances)
            col_name = 'x_output' if 'x_output' in output_data.columns else 'x_input'

            # Parsing
            for j in range(len(output_data[col_name])):
                raw = output_data[col_name][j]

                if isinstance(raw, str):
                    # Rimuove "array(...)" e parentesi
                    raw = raw.replace("[array(", "").replace(")]", "").replace(")", "").replace("[", "").replace("]", "")
                    items = raw.split(',')
                    values = []
                    for item in items:
                        item = item.strip()
                        try:
                            values.append(float(item))
                        except ValueError:
                            continue
                else:
                    values = list(raw)

                # Inserisci nei canali
                for i in range(min(self.univar_count, len(values))):
                    self.rand_var_out[key_value_name][i].append(values[i])

            cprint(f"{Style.BRIGHT}|\t{key_value_name}: {len(self.rand_var_out[key_value_name][0])} samples\t {self.univar_count} variables{Style.RESET_ALL}", 'cyan', attrs=['bold'])

        # Conversione in numpy
        for key_value_name in self.key_value_list:
            self.np_dist[key_value_name][key] = pd.DataFrame.from_dict(
                self.rand_var_out[key_value_name]
            ).to_numpy()
        print("select_data")   
        self.select_data()

        
    def comparison_measures(self, measures):
        """Esegue misure di confronto per tutti i canali"""
        if not self.alredy_select_data:
            self.select_data()
        
        # Metriche per ogni singolo canale
        if 'metrics' in measures or 'metrics_per_channel' in measures:
            self.comparison_metrics_per_channel()
        
        # Metriche aggregate su tutti i canali
        if 'metrics_aggregate' in measures:
            self.comparison_metrics_aggregate()
        
        if 'tsne_plots' in measures:
            self.comparison_tsne_per_channel(apply_pca=False)
        
        if 'pca_tsne_plots' in measures:
            self.comparison_tsne_per_channel(apply_pca=True)
        
        if 'wasserstein_dist' in measures:
            self.comparison_wasserstein()
        
        if 'swarm_distributions' in measures:
            self.comparison_swarm_distributions()
        
        # Salva metriche
        if self.metrics_pd is not None:
            filename = Path(self.path_folder, f"metrics_compare_{self.suffix}.csv")
            self.metrics_pd.to_csv(filename)
            print(f"\nMetrics saved to: {filename}")
            print(f"{Style.BRIGHT}\033[38;2;122;252;0m| Metrics saved to: {filename}{Style.RESET_ALL}")

    
    def comparison_metrics_per_channel(self):

        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tComputing metrics per channel{Style.RESET_ALL}")
        all_metrics = []
        
        for key_value_name in self.key_value_list:
            print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \t\tchannel: {key_value_name}{Style.RESET_ALL}")
            for comparison in self.comparisons:
                data_A_key = self.comparisons[comparison]['a']
                data_B_key = self.comparisons[comparison]['b']
                
                # Verifica che i dati esistano
                if data_A_key not in self.np_dist[key_value_name] or data_B_key not in self.np_dist[key_value_name]:
                    print(f"  Skipping comparison {comparison}: data not available")
                    continue
                
                data_A = self.np_dist[key_value_name][data_A_key]
                data_B = self.np_dist[key_value_name][data_B_key]

                # Calcola metriche
                try:
                    mah_dist = self.mahalanobis(data_A, data_B)
                    wass_mean, wass_list = self.wasserstein(data_A, data_B)
                    fid = self.frechet_inception_distance(data_A, data_B)
                    corr_metrics = self.corr_matrix(data_A, data_B, 
                                                   label_A=f"{key_value_name}_{data_A_key}", 
                                                   label_B=f"{key_value_name}_{data_B_key}")
                    
                    all_metrics.append({
                        'channel': key_value_name,
                        'comparison': comparison,
                        'mahalanobis': mah_dist,
                        'wasserstein_mean': wass_mean,
                        'wasserstein_std': np.std(wass_list),
                        'frechet': fid,
                        'corr_frobenius': corr_metrics['Frobenius_norm'],
                        'corr_mean_abs': corr_metrics['Mean_absolute'],
                        'corr_total_abs': corr_metrics['Total_absolute']
                    })
                    
                    print(f"    Mahalanobis: {mah_dist:.4f}")
                    print(f"    Wasserstein: {wass_mean:.4f}")
                    print(f"    Frechet: {fid:.4f}")
                    
                except Exception as e:
                    print(f"    Error computing metrics: {e}")
                    continue
        
        # Salva metriche per canale
        if all_metrics:
            self.metrics_pd = pd.DataFrame(all_metrics)
            filename = Path(self.path_folder, f"metrics_per_channel_{self.suffix}.csv")
            self.metrics_pd.to_csv(filename, index=False)
            print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tMetrics saved to:{filename}{Style.RESET_ALL}")
            
        
        return self.metrics_pd
    
    def comparison_metrics_aggregate(self):
        """Calcola metriche aggregate concatenando tutti i canali"""
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tComputing aggregate metrics (all channels){Style.RESET_ALL}")
        # Concatena tutti i canali
        aggregated_data = {}
        for data_key in ['real', 'cop'] + [comp.split('_')[1] for comp in self.comparisons.keys() if comp != 'real_cop']:
            channel_arrays = []
            for key_value_name in self.key_value_list:
                if data_key in self.np_dist[key_value_name]:
                    channel_arrays.append(self.np_dist[key_value_name][data_key])
            
            if channel_arrays:
                aggregated_data[data_key] = np.concatenate(channel_arrays, axis=1)
        
        # Calcola metriche
        aggregate_metrics = []
        for comparison in self.comparisons:
            data_A_key = self.comparisons[comparison]['a']
            data_B_key = self.comparisons[comparison]['b']
            
            if data_A_key not in aggregated_data or data_B_key not in aggregated_data:
                continue
            
            data_A = aggregated_data[data_A_key]
            data_B = aggregated_data[data_B_key]
            
            print(f"\nComparison: {comparison} | Aggregate shape: {data_A.shape}")
            
            mah_dist = self.mahalanobis(data_A, data_B)
            wass_mean, _ = self.wasserstein(data_A, data_B)
            fid = self.frechet_inception_distance(data_A, data_B)
            
            aggregate_metrics.append({
                'comparison': comparison,
                'mahalanobis': mah_dist,
                'wasserstein_mean': wass_mean,
                'frechet': fid,
                'n_features': data_A.shape[1]
            })
            
            print(f"  Mahalanobis: {mah_dist:.4f}")
            print(f"  Wasserstein: {wass_mean:.4f}")
            print(f"  Frechet: {fid:.4f}")
        
        # Salva
        df_aggregate = pd.DataFrame(aggregate_metrics)
        filename = Path(self.path_folder, f"metrics_aggregate_{self.suffix}.csv")
        df_aggregate.to_csv(filename, index=False)
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tAggregate metrics saved to:{filename} {Style.RESET_ALL}")

        return df_aggregate

    def comparison_wasserstein(self):
        """Confronto Wasserstein dettagliato per ogni canale"""
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tWasserstein Distance Analysis{Style.RESET_ALL}")

        
        all_stats = []
        
        for key_value_name in self.key_value_list:
            for i in range(self.univar_count):
                dist_real = self.rand_var_in[key_value_name][i]
                dist_fake = self.rand_var_out[key_value_name][i]
                
                wd = wasserstein_distance(dist_real, dist_fake)
                
                stats_dict = {
                    'channel': key_value_name,
                    'variable': i,
                    'mean_real': np.mean(dist_real),
                    'std_real': np.std(dist_real),
                    'mean_pred': np.mean(dist_fake),
                    'std_pred': np.std(dist_fake),
                    'wasserstein_dist': wd,
                    'mean_diff': abs(np.mean(dist_real) - np.mean(dist_fake))
                }
                
                if self.use_copula:
                    dist_cop = self.rand_var_cop[key_value_name][i]
                    wd_cop = wasserstein_distance(dist_real, dist_cop)
                    stats_dict['mean_cop'] = np.mean(dist_cop)
                    stats_dict['std_cop'] = np.std(dist_cop)
                    stats_dict['wasserstein_dist_cop'] = wd_cop
                
                all_stats.append(stats_dict)
        
        # Salva
        df_stats = pd.DataFrame(all_stats)
        filename = Path(self.path_folder, f"wasserstein_compare_{self.suffix}.csv")
        df_stats.to_csv(filename, index=False)
        
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \t\tstats save in:{self.path_folder} {Style.RESET_ALL}")
        return df_stats
    
    def comparison_tsne_per_channel(self, apply_pca=False):
        """t-SNE separato per ogni canale"""        
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tt-SNE Visualization (PCA={apply_pca}){Style.RESET_ALL}")

        print("self.df_data_selected_per_channel",self.df_data_selected_per_channel)
        for key_value_name in self.key_value_list:
            print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \t\tchannel: {key_value_name}{Style.RESET_ALL}")
            # Seleziona dati per questo canale
            df_channel = self.df_data_selected_per_channel.get(key_value_name)
            if df_channel is None:
                print(f"  No data selected for {key_value_name}")
                continue
            
            data4fit = df_channel.drop(columns=['labels'])
            
            

            
            if apply_pca:
                n_components_pca = min(50, data4fit.shape[1])
                pca = PCA(n_components=n_components_pca)
                data4fit = pca.fit_transform(data4fit)
            
            # t-SNE 2D
            n_samples = data4fit.shape[0]
            adjusted_perplexity = min(30, max(1, n_samples // 2))  # garantisce perplexity < n_samples
            tsne = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42)
            
            
            print("data4fit",data4fit)
            tsne_results = tsne.fit_transform(data4fit)
            df_tsne = pd.DataFrame(tsne_results, columns=[0, 1])
            df_tsne['labels'] = df_channel['labels'].values
            
            # Plot
            fig = plt.figure(figsize=(16, 7))
            sns.scatterplot(x=0, y=1, hue="labels", palette=self.color_list,
                          data=df_tsne, alpha=0.3, legend="full")
            plt.title(f"t-SNE 2D - {key_value_name}")
            
            prefix = "PCA_TSNE" if apply_pca else "TSNE"
            filename = Path(self.path_folder, f"{prefix}_2D_{key_value_name}_{self.suffix}.png")
            plt.savefig(filename, dpi=150)
            plt.close()
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \t\tplots save in:{self.path_folder} {Style.RESET_ALL}")
            

    def comparison_swarm_distributions(self, subset_size=400):
        """Violin/swarm plots per ogni canale"""
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tSwarm Distribution Plots{Style.RESET_ALL}")
        
        for key_value_name in self.key_value_list:
            
            
            df_channel = self.df_data_selected_per_channel.get(key_value_name)
            if df_channel is None:
                continue
            
            # Subset per swarmplot
            labels = df_channel['labels'].unique()
            subsets = [df_channel[df_channel['labels'] == label].iloc[:subset_size] for label in labels]
            df_swarm = pd.concat(subsets, ignore_index=True)
            
            # Plot
            fig, axs = plt.subplots(figsize=(max(20, self.univar_count * 2), 8), 
                                   ncols=self.univar_count)
            
            if self.univar_count == 1:
                axs = [axs]
            
            for i in range(self.univar_count):
                col_name = f'c_{i}'
                sns.violinplot(data=df_channel, y=col_name, x="labels", 
                             palette=self.color_list, hue="labels", ax=axs[i], legend=False)
                sns.swarmplot(data=df_swarm, y=col_name, x="labels", 
                            palette=self.color_list, size=2, ax=axs[i])
                axs[i].set_title(f"Var {i}")
                axs[i].set_xlabel("")
            
            plt.tight_layout()
            filename = Path(self.path_folder, f"SWARM_{key_value_name}_{self.suffix}.png")
            fig.savefig(filename, dpi=150)
            plt.close()
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \t\tplots save in:{self.path_folder} {Style.RESET_ALL}")
        

    
    # ===== UTILITY METHODS =====
    
    def wasserstein(self, X, Y):
        """Wasserstein distance per ogni dimensione"""
        wasserstein_distances = [wasserstein_distance(X[:, i], Y[:, i]) for i in range(X.shape[1])]
        return np.mean(wasserstein_distances), wasserstein_distances
    
    def corr_matrix(self, X, Y, label_A, label_B):
        """Matrice di correlazione con visualizzazione"""
        X_df = pd.DataFrame(X)
        Y_df = pd.DataFrame(Y)
        
        corr_X = X_df.corr()
        corr_Y = Y_df.corr()
        corr_diff = corr_X - corr_Y
    
        # Salva CSV
        filename_X = Path(self.path_folder, f'correlation_{label_A}.csv')
        filename_Y = Path(self.path_folder, f'correlation_{label_B}.csv')
        filename_XY = Path(self.path_folder, f'correlation_diff_{label_A}_{label_B}.csv')
        corr_X.to_csv(filename_X)
        corr_Y.to_csv(filename_Y)
        corr_diff.to_csv(filename_XY)
        
        # Plot solo se dimensione gestibile
        if X.shape[1] <= 50:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_X, cmap="coolwarm", annot=False, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
            plt.title(f"Correlation Matrix - {label_A}")
            filename_Xplot = Path(self.path_folder, f'plot_correlation_{label_A}.png')
            plt.savefig(filename_Xplot, dpi=200, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_Y, cmap="coolwarm", annot=False, vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'})
            plt.title(f"Correlation Matrix - {label_B}")
            filename_Yplot = Path(self.path_folder, f'plot_correlation_{label_B}.png')
            plt.savefig(filename_Yplot, dpi=200, bbox_inches='tight')
            plt.close()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_diff, cmap="coolwarm", annot=False, center=0, 
                       vmin=-1, vmax=1, cbar_kws={'label': 'Difference'})
            plt.title(f"Correlation Difference - {label_A} vs {label_B}")
            filename_XYplot = Path(self.path_folder, f'plot_correlation_diff_{label_A}_{label_B}.png')
            plt.savefig(filename_XYplot, dpi=200, bbox_inches='tight')
            plt.close()
        
        metrics = {
            "Frobenius_norm": np.linalg.norm(corr_diff.values, 'fro'),
            "Mean_absolute": np.mean(np.abs(corr_diff.values)),
            "Total_absolute": np.sum(np.abs(corr_diff.values))
        }
        return metrics
        
    def mahalanobis(self, X, Y):
        """Distanza di Mahalanobis tra due dataset"""
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
      
    def frechet_inception_distance(self, real_samples, generated_samples):
        """FID score"""
        mu_real = np.mean(real_samples, axis=0)
        mu_generated = np.mean(generated_samples, axis=0)
        sigma_real = np.cov(real_samples, rowvar=False)
        sigma_generated = np.cov(generated_samples, rowvar=False)
        diff = mu_real - mu_generated
        diff_squared = np.sum(diff**2)
        covmean, _ = sqrtm(sigma_real @ sigma_generated, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff_squared + np.trace(sigma_real + sigma_generated - 2 * covmean)
        return fid
    
    def bhattacharyya_coefficient(self, X, Y, num_samples=10000):
        """Stima il coefficiente di Bhattacharyya usando KDE multivariato e Monte Carlo"""
        kde_X = gaussian_kde(X.T, bw_method='silverman')
        kde_Y = gaussian_kde(Y.T, bw_method='silverman')
        
        min_vals, max_vals = np.min(X, axis=0), np.max(Y, axis=0)
        samples = np.random.uniform(min_vals, max_vals, size=(num_samples, X.shape[1])).T

        P_samples = kde_X(samples)
        Q_samples = kde_Y(samples)

        BC = np.mean(np.sqrt(P_samples * Q_samples))
        return BC

    def bhattacharyya(self, X, Y):
        """Distanza di Bhattacharyya"""
        BC = self.bhattacharyya_coefficient(X, Y)
        return -np.log(BC) if BC > 0 else np.inf
        
    def select_data(self, n_points=None):
        """Selezione dati per visualizzazione — separata per ogni canale"""
        print("select_data oooop")

        if n_points is None:
            n_points = self.n_sample_considered

        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| visualization:\t{self.suffix}{Style.RESET_ALL}")
        self.df_data_selected_per_channel = {}

        for key_value_name in self.key_value_list:
            # Verifica disponibilità dati
            available_lengths = []
            if key_value_name in self.rand_var_in and 0 in self.rand_var_in[key_value_name]:
                available_lengths.append(len(self.rand_var_in[key_value_name][0]))
            if key_value_name in self.rand_var_out and 0 in self.rand_var_out[key_value_name]:
                available_lengths.append(len(self.rand_var_out[key_value_name][0]))
            if self.use_copula and key_value_name in self.rand_var_cop and 0 in self.rand_var_cop[key_value_name]:
                available_lengths.append(len(self.rand_var_cop[key_value_name][0]))
            print(self.rand_var_cop[key_value_name])
            raise Exception("debug select_data 2")
            print("available_lengths",available_lengths)
            if not available_lengths:
                print(f"{Style.BRIGHT}\033[38;2;255;100;100m| No data available for {key_value_name}, skipping{Style.RESET_ALL}")
                continue

            n_points_channel = min(n_points, min(available_lengths))

            # Selezione randomica
            def select_indices(source_dict):
                if key_value_name not in source_dict or 0 not in source_dict[key_value_name]:
                    return []
                total = len(source_dict[key_value_name][0])
                return random.sample(range(total), n_points_channel)

            selected_real = select_indices(self.rand_var_in)
            selected_out = select_indices(self.rand_var_out)
            selected_cop = select_indices(self.rand_var_cop) if self.use_copula else []
            print("selected_real",selected_real)
            print("selected_out",selected_out)
            print("selected_cop",selected_cop)

            real_vals = [self.rand_var_in["uno"][i][j] for j in selected_real]
            print("real_vals",real_vals)
            # Costruzione DataFrame
            df_channel = pd.DataFrame()
            for i in range(self.univar_count):
                real_vals = [self.rand_var_in[key_value_name][i][j] for j in selected_real]
                out_vals = [self.rand_var_out[key_value_name][i][j] for j in selected_out]
                cop_vals = [self.rand_var_cop[key_value_name][i][j] for j in selected_cop] if self.use_copula else []

                df_channel[f'c_{i}'] = real_vals + out_vals + cop_vals
            raise Exception("debug select_data")
            # Etichette
            labels = (
                ["real"] * len(selected_real) +
                [self.name_key] * len(selected_out) +
                (["cop"] * len(selected_cop) if self.use_copula else [])
            )
            df_channel['labels'] = labels

            print("labels",labels)
            self.df_data_selected_per_channel[key_value_name] = df_channel

        # Logging finale
        print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \tistance labels distribution ({self.suffix}){Style.RESET_ALL}")
        for key_value_name in self.df_data_selected_per_channel:
            print(f"{Style.BRIGHT}\033[38;2;122;252;0m| \t{key_value_name}{Style.RESET_ALL}")
            for label, count in self.df_data_selected_per_channel[key_value_name]['labels'].value_counts().sort_index().items():
                print(f"{Style.BRIGHT}\033[38;2;122;252;0m|\t\tLabel {label:<10} : {count} instances{Style.RESET_ALL}")

        self.alredy_select_data = True

        
    
    def get_channel_data(self, key_value_name, data_type='real'):
        """
        Ottiene i dati numpy per un canale specifico
        
        Args:
            key_value_name: nome del canale
            data_type: 'real', 'cop', o chiave custom
            
        Returns:
            numpy array con shape (n_samples, univar_count)
        """
        if key_value_name not in self.np_dist:
            raise ValueError(f"Channel {key_value_name} not found")
        if data_type not in self.np_dist[key_value_name]:
            raise ValueError(f"Data type {data_type} not found for channel {key_value_name}")
        
        return self.np_dist[key_value_name][data_type]
    
    def get_all_channels_concatenated(self, data_type='real'):
        """
        Concatena tutti i canali in un unico array
        
        Args:
            data_type: 'real', 'cop', o chiave custom
            
        Returns:
            numpy array con shape (n_samples, univar_count * n_channels)
        """
        channel_arrays = []
        for key_value_name in self.key_value_list:
            if data_type in self.np_dist[key_value_name]:
                channel_arrays.append(self.np_dist[key_value_name][data_type])
        
        if not channel_arrays:
            raise ValueError(f"No data found for type {data_type}")
        
        return np.concatenate(channel_arrays, axis=1)
    
    def export_comparison_summary(self):
        """Esporta un riassunto comparativo in formato leggibile"""
        summary_lines = []
        summary_lines.append("=" * 80)
        summary_lines.append("DATA COMPARISON SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"\nDataset: {self.name_key}")
        summary_lines.append(f"Channels: {', '.join(self.key_value_list)}")
        summary_lines.append(f"Variables per channel: {self.univar_count}")
        summary_lines.append(f"Total features: {self.univar_count * self.n_channels}")
        summary_lines.append(f"Use copula: {self.use_copula}")
        
        summary_lines.append("\n" + "-" * 80)
        summary_lines.append("DATA SIZES")
        summary_lines.append("-" * 80)
        
        for key_value_name in self.key_value_list:
            summary_lines.append(f"\n{key_value_name}:")
            for data_type in self.np_dist[key_value_name].keys():
                shape = self.np_dist[key_value_name][data_type].shape
                summary_lines.append(f"  {data_type}: {shape[0]} samples, {shape[1]} features")
        
        if self.metrics_pd is not None:
            summary_lines.append("\n" + "-" * 80)
            summary_lines.append("METRICS SUMMARY")
            summary_lines.append("-" * 80)
            summary_lines.append("\n" + str(self.metrics_pd))
        
        summary_lines.append("\n" + "=" * 80)
        
        # Salva
        summary_text = "\n".join(summary_lines)
        filename = Path(self.path_folder, f"comparison_summary_{self.suffix}.txt")
        with open(filename, 'w') as f:
            f.write(summary_text)
        
        print(summary_text)
        print(f"\n✓ Summary saved to: {filename}")
        
        return summary_text
    
    def plot_metrics_heatmap(self):
        """Crea heatmap delle metriche per confronto visivo"""
        if self.metrics_pd is None or self.metrics_pd.empty:
            print("No metrics available to plot")
            return
        
        # Prepara dati per heatmap
        metrics_to_plot = ['mahalanobis', 'wasserstein_mean', 'frechet']
        available_metrics = [m for m in metrics_to_plot if m in self.metrics_pd.columns]
        
        if not available_metrics:
            print("No suitable metrics found for heatmap")
            return
        
        # Pivot per heatmap
        plot_data = self.metrics_pd[['channel', 'comparison'] + available_metrics].copy()
        
        for metric in available_metrics:
            fig, ax = plt.subplots(figsize=(12, max(6, len(self.key_value_list))))
            
            # Pivot table
            pivot = plot_data.pivot(index='channel', columns='comparison', values=metric)
            
            # Plot
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': metric})
            ax.set_title(f"{metric.replace('_', ' ').title()} by Channel and Comparison")
            ax.set_xlabel("Comparison")
            ax.set_ylabel("Channel")
            
            plt.tight_layout()
            filename = Path(self.path_folder, f"heatmap_{metric}_{self.suffix}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved heatmap: {filename}")
    
    def compare_channel_statistics(self):
        """Confronta statistiche base tra canali"""
        print("\n=== Channel Statistics Comparison ===")
        
        stats_list = []
        
        for key_value_name in self.key_value_list:
            for data_type in ['real', self.name_key]:
                if data_type not in self.np_dist[key_value_name]:
                    continue
                
                data = self.np_dist[key_value_name][data_type]
                
                stats_list.append({
                    'channel': key_value_name,
                    'data_type': data_type,
                    'mean': np.mean(data),
                    'std': np.std(data),
                    'min': np.min(data),
                    'max': np.max(data),
                    'median': np.median(data),
                    'q25': np.percentile(data, 25),
                    'q75': np.percentile(data, 75)
                })
        
        df_stats = pd.DataFrame(stats_list)
        
        # Salva
        filename = Path(self.path_folder, f"channel_statistics_{self.suffix}.csv")
        df_stats.to_csv(filename, index=False)
        print(f"✓ Channel statistics saved to: {filename}")
        
        # Print summary
        print("\n" + df_stats.to_string())
        
        return df_stats



class CorrelationComparison():

    def __init__(self, correlation_matrices, folder):
        self.dict_matrices = correlation_matrices
        self.path_fold = Path(folder, "correlation_comparison")
        if not os.path.exists(self.path_fold):
            os.makedirs(self.path_fold)
    
    def compareMatrices(self, list_comparisons):
        df = pd.DataFrame()
        for (key_a, key_b) in list_comparisons:
            frobenius_val = self.frobenius_norm(key_a, key_b)
            spearmanr_val = self.spearmanr(key_a, key_b)
            cosin_sim_val = self.cosineSimilarity(key_a, key_b)

            new_row = {'matrix_A': self.keyToSting(key_a), 'matrix_B': self.keyToSting(key_b), 'frobenius': frobenius_val, 'spearmanr_statistic': spearmanr_val[0], 'spearmanr_pvalue': spearmanr_val[1], 'cosineSimilarity': cosin_sim_val}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
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
        diff_matrix = matrix1 - matrix2f
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
        cos_sim = dot(matrix1_flat, matrix2_flat) / (norm(matrix1_flat) * norm(matrix2_flat))
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
            assert(type(df) == np.ndarray)
        except:
            if type(df) == pd.DataFrame:
                df = df.values
            else:
                raise TypeError('Must be np.ndarray or pd.DataFrame')
        mask = np.triu_indices(df.shape[0], k=1)
        return df[mask]


class SparseNDHistogram():
    def __init__(self, n_bins, n_comp):
        self.data = {}
        self.n_bins = n_bins
        self.n_comp = n_comp
        self.edges = None
        self.histograms = None
        
    def increment(self, indices):
        if indices in self.data:
            self.data[indices] += 1
        else:
            self.data[indices] = 1

    def get(self, indices):
        return self.data.get(indices, 0)

    def get_indices(self):
        return list(self.data.keys())
    
    def get_indices_values(self):
        val_list = []
        for key in self.data.keys():
            val_list.append(self.data[key])
        return val_list
    
    def get_data(self):
        return self.data

    def compute_edges(self, *datasets):
        edges = []
        for i in range(self.n_comp):
            min_val = np.array(min(data[:, i].astype(np.float64).min() for data in datasets))
            max_val = np.array(max(data[:, i].astype(np.float64).max() for data in datasets))

            edges.append(np.linspace(min_val, max_val, self.n_bins + 1))
        self.edges = edges
        return self.edges

    def set_edges(self, edges):
        self.edges = edges
    
    def print_edges(self):
        return self.edges
            
    def compute_histogram(self, data):
        if self.edges is not None:
            for point in data:
                bin_indices = tuple(max(0, np.digitize(point[i], self.edges[i], right=True) - 1) for i in range(self.n_comp))
                self.increment(bin_indices)
        
    @classmethod
    def from_datasets(cls, n_bins, n_comp, *datasets):        
        histograms = [cls(n_bins, n_comp) for _ in datasets]
        histograms[0].compute_edges(*datasets)
        
        for hist in histograms:
            hist.set_edges(histograms[0].edges)
        for hist, data in zip(histograms, datasets):
            hist.compute_histogram(data)

        return histograms 

    @classmethod
    def compute_error(cls, histograms, error_name=["MAE", "RMSE"], matrix_selected=[1, 2]):
        print("matrix_selected:\t", matrix_selected[0], matrix_selected[1])
        if len(matrix_selected) != 2:
            raise ValueError("Devi selezionare esattamente due istogrammi per calcolare l'errore.")

        hist1, hist2 = histograms[matrix_selected[0]], histograms[matrix_selected[1]]
        indices = set(hist1.get_indices()).union(set(hist2.get_indices()))

        error_sq_sum = 0
        error_abs_sum = 0
        a = 0
        for index in indices:
            val1 = hist1.get(index)
            val2 = hist2.get(index)
            diff = val1 - val2
            error_sq_sum += diff ** 2
            error_abs_sum += abs(diff)
            a += 1
        n = len(indices) if indices else 1
        
        errors_values = dict()
        if "RMSE" in error_name:
            errors_values["RMSE"] = np.sqrt(error_sq_sum / n)
        if "MAE" in error_name:
            errors_values["MAE"] = error_abs_sum / n
        
        return errors_values



    
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
from matplotlib.ticker import PercentFormatter
import datetime
from termcolor import colored, cprint 
from colorama import init, Style

class ModelPrediction():

    def __init__(self, model, device, dataset, key_value_list, univar_count_in, univar_count_out, latent_dim, path_folder, vc_mapping, time_performance, model_type, data_range=None, input_shape="vector", isnoise_in = False, isnoise_out = False):
        self.model = model
        self.dataset = dataset
        self.key_value_list = key_value_list  # lista di chiavi/canali
        self.univar_count_in = univar_count_in
        self.univar_count_out = univar_count_out
        self.latent_dim = latent_dim
        self.path_folder = path_folder
        self.device = device
        self.vc_mapping = vc_mapping
        self.time_performance = time_performance
        # data_range ora è un dict per ogni chiave di key_value_list
        self.data_range = data_range if data_range is not None else {}
        self.inpu_data = list()
        self.late_data = list()
        self.pred_data = list()
        self.inpu_byVar = dict()
        self.pred_byVar = dict()
        self.late_byComp = dict()
        self.input_shape = input_shape
        self.isnoise_in = isnoise_in
        self.isnoise_out = isnoise_out
        self.model_type = model_type
        if self.model_type in ["VAE"]:
            self.latent_keys = ["mu","logvar", "z"]
        else:
            self.latent_keys = ["latent"]

    def predict(self, input_sample, time_key, pred2numpy=True, latent=True, save=True, experiment_name=None, remapping=False):
        """
        Esegue la predizione mantenendo la struttura corretta (B, R, C).
        """
        for key_idx, key in enumerate(self.key_value_list):
            self.inpu_data = list()
            self.late_data = dict()
            for late_key in self.latent_keys:
                self.late_data[late_key] = list()
            self.pred_data = list()
            
            time_key_key = f"{time_key}_{key}_pred"
            
            for item in input_sample:
                with torch.no_grad():
                    sample_list = [item['sample'].type(torch.float32)]
                    x_in = torch.stack(sample_list, dim=0).to(device=self.device)

                    self.time_performance.start_time(time_key_key)
                    out = self.model(x_in)
                    self.time_performance.stop_time(time_key_key)

                    # Salva x_input: mantieni dimensione batch se presente
                    # out["x_input"]['data'] ha shape (B, R, C)
                    x_input_batch = out["x_input"]['data']
                    self.inpu_data.append(x_input_batch)
                    
                    # Salva latent space
                    if "x_latent" in out:
                        for late_key in self.latent_keys:
                            # Prendi il primo elemento del batch per il latent
                            self.late_data[late_key].append(out["x_latent"][late_key][0])
                    
                    # Salva x_output: mantieni dimensione batch
                    # out["x_output"]['data'] ha shape (B, R, C)
                    x_output_batch = out["x_output"]['data']
                    self.pred_data.append(x_output_batch)

            self.time_performance.compute_time(time_key_key, fun="mean")
            print_time_key = self.time_performance.get_time(time_key_key, fun="mean")
            print(f"{Style.BRIGHT}\033[38;2;121;212;242m| \ttime \t {time_key_key} prediction:\t{print_time_key}{Style.RESET_ALL}")

            self.predict_sortByUnivar(pred2numpy=pred2numpy)
            
            if latent:
                self.latent_sortByComponent(pred2numpy=pred2numpy)
            if save:
                self.saveData(experiment_name, latent, remapping, key)


    def saveData(self, experiment_name, latent, remapping=True, key=None):
        """
        Salva i dati per un singolo canale (key) in formato compatibile.
        Ogni file contiene solo i valori di quel canale, per tutte le variabili.
        """
        if key is None:
            raise ValueError("La chiave del canale (key) deve essere specificata")

        # Trova l'indice del canale corrente
        if key not in self.key_value_list:
            raise ValueError(f"Chiave '{key}' non trovata in key_value_list")
        ch_idx = self.key_value_list.index(key)

        # Range per denormalizzazione
        data_range = self.data_range.get(key, None)
        min_val = 0
        diff_minmax = None
        if data_range is not None:
            min_val = data_range['min_val']
            diff_minmax = data_range['max_val'] - min_val

        # Colonne del DataFrame
        columns = ['instance_id', 'x_input']
        if latent:
            for late_key in self.latent_keys:
                columns.append(f'x_latent_{late_key}')
        columns.append('x_output')

        df_export = pd.DataFrame(columns=columns)

        # Itera sulle istanze
        for idx in range(len(self.pred_data)):
            x_tensor = self.inpu_data[idx][0]  # (R, C)
            y_tensor = self.pred_data[idx][0]  # (R, C)

            # Estrai solo il canale corrente → vettori (R,)
            x_vec = x_tensor[:, ch_idx].detach().cpu().numpy()
            y_vec = y_tensor[:, ch_idx].detach().cpu().numpy()

            # Denormalizza se necessario
            if diff_minmax is not None:
                x_vec = (x_vec * diff_minmax) + min_val
                y_vec = (y_vec * diff_minmax) + min_val

            # Costruisci la riga
            new_row = {'instance_id': idx, 'x_input': [x_vec], 'x_output': [y_vec]}

            if latent:
                for late_key in self.latent_keys:
                    z_tensor = self.late_data[late_key][idx]
                    z_np = z_tensor.detach().cpu().numpy()
                    new_row[f'x_latent_{late_key}'] = z_np.tolist()

            df_export.loc[len(df_export)] = new_row

        # Salvataggio
        filename = f"prediced_instances_{experiment_name}_{key}.csv"
        path_file = Path(self.path_folder, filename)
        df_export.to_csv(path_file, index=False)

        print(f"{Style.BRIGHT}\033[38;2;121;212;242m| \tSaved single-channel prediction to:\t{path_file}{Style.RESET_ALL}")
        print(f"{Style.BRIGHT}\033[38;2;121;212;242m| \tChannel: {key} \tShape: {x_vec.shape}{Style.RESET_ALL}")



    def predict_sortByUnivar(self, pred2numpy=True, mode=["input", "output"]):
        """
        Raggruppa in formato channel-first usando self.key_value_list come nomi canali.
        Assume che i tensori abbiano layout (batch, n_vars, n_channels) o (n_vars, n_channels) per sample.
        Risultato:
        self.pred_byVar[channel_name][var_id] = list di valori (uno per sample)
        self.inpu_byVar[channel_name][var_id]  = analogo per input
        """
        import numpy as _np
        import torch as _torch

        def to_numpy(x):
            return x.detach().cpu().numpy() if isinstance(x, _torch.Tensor) else _np.asarray(x)

        def ensure_bvn(t):
            # return numpy array shaped (batch, n_vars, n_channels)
            arr = to_numpy(t)
            if arr.ndim == 3:
                return arr
            if arr.ndim == 2:
                return arr[_np.newaxis, ...]
            raise ValueError(f"Unsupported tensor ndim={arr.ndim}; expected 2 or 3")

        # infer channels / names
        if not hasattr(self, "key_value_list") or not self.key_value_list:
            sample_series = getattr(self, "pred_data", None) or getattr(self, "inpu_data", None)
            if not sample_series:
                raise ValueError("Impossibile inferire key_value_list: imposta self.key_value_list o fornisci pred_data/inpu_data")
            first = sample_series[0][0] if isinstance(sample_series[0], (list, tuple)) else sample_series[0]
            arr_first = to_numpy(first)
            n_ch = int(arr_first.shape[-1]) if arr_first.ndim in (2,3) else None
            if n_ch is None:
                raise ValueError("Impossibile inferire numero di canali dal sample")
            self.key_value_list = [f"ch{c}" for c in range(n_ch)]
        self.n_channels = len(self.key_value_list)

        def collect(series, n_vars):
            # init channel-first structure: channel_name -> var_id -> list(samples)
            result = {ch_name: {vid: [] for vid in range(n_vars)} for ch_name in self.key_value_list}
            for item in series:
                t = item[0] if isinstance(item, (list, tuple)) else item
                arr_bvn = ensure_bvn(t)  # (batch, n_vars, n_channels)
                # verify n_vars dimension
                for b in range(arr_bvn.shape[0]):
                    batch_item = arr_bvn[b]   # (n_vars, n_channels)
                    if batch_item.shape[0] < n_vars:
                        # try transpose if channels and vars swapped
                        if batch_item.shape[1] == n_vars:
                            batch_item = batch_item.T
                        else:
                            raise ValueError(f"Found {batch_item.shape[0]} variables < expected {n_vars}")
                    for vid in range(n_vars):
                        for ch in range(batch_item.shape[-1]):
                            ch_name = self.key_value_list[ch] if ch < self.n_channels else f"ch{ch}"
                            val = batch_item[vid, ch]
                            if pred2numpy:
                                result[ch_name][vid].append(_np.asarray(val))
                            else:
                                # re-index original torch tensor when possible to preserve type
                                if isinstance(t, _torch.Tensor):
                                    if t.ndim == 3:
                                        result[ch_name][vid].append(t[b, vid, ch].detach())
                                    elif t.ndim == 2:
                                        result[ch_name][vid].append(t[vid, ch].detach())
                                    else:
                                        result[ch_name][vid].append(_torch.from_numpy(_np.asarray(val)))
                                else:
                                    result[ch_name][vid].append(_torch.from_numpy(_np.asarray(val)))
            return result

        if "input" in mode:
            self.inpu_byVar = collect(getattr(self, "inpu_data", []), self.univar_count_in)

        if "output" in mode:
            self.pred_byVar = collect(getattr(self, "pred_data", []), self.univar_count_out)


                                
        
    def latent_sortByComponent(self, pred2numpy=True):
        self.late_byComp = dict()
        
        for late_key in self.latent_keys:
            self.late_byComp[late_key] = dict()
            for id_comp in range(self.latent_dim):
                self.late_byComp[late_key][id_comp] = list()
        
        
        for late_key in self.latent_keys:


            for lat in self.late_data[late_key]:
                if self.input_shape == "vector":
                    for late_key in self.latent_keys:
                        for id_comp in range(self.latent_dim):                
                            if pred2numpy:
                                self.late_byComp[late_key][id_comp].append(lat[id_comp].detach().cpu().numpy())
                            else:
                                self.late_byComp[late_key][id_comp].append(lat[id_comp])
                elif self.input_shape == "matrix":
                
                    for id_comp in range(self.latent_dim):                
                        for lat_varValue_instance in lat[0][id_comp]:
                            if pred2numpy:
                                self.late_byComp[late_key][id_comp].append(lat_varValue_instance.detach().cpu().numpy())
                            else:
                                self.late_byComp[late_key][id_comp].append(lat_varValue_instance)

    def getPred(self):
        by_univar_dict = {"input":self.inpu_data, "latent": self.late_data, "output":self.pred_data}
        return by_univar_dict


    def getPred_byUnivar(self):
        by_univar_dict = {"input":self.inpu_byVar, "output":self.pred_byVar}
        return by_univar_dict

    def getLat_byComponent(self):
        by_comp_dict = {"latent":self.late_byComp}
        return by_comp_dict

    def getLat(self):
        comp_dict = {"latent":self.late_data}
        return comp_dict
    
    def getLatent2data(self,latent_key_toplot):
        data_latent = list()
        
        for istance in self.late_data[latent_key_toplot]:
            istance_dict = {'sample': istance}
            data_latent.append(istance_dict)
        comp_dict = {"latent":data_latent}
        return comp_dict
    
    
    def input_byvar(self):
        self.predict_sortByUnivar(mode="input",pred2numpy=True)
        resultDict = dict()
        resultDict["prediction_data_byvar"] = {"input":self.inpu_byVar}
        return resultDict
    
    
    def compute_prediction(self, experiment_name, time_key, remapping_data=False, force_noLatent=False, latent_key_toplot=None):
        """
        Compute predictions with multi-channel support.
        
        IMPORTANT: 'input' here refers to the model's internal representation (e.g., latent space),
        NOT the original input data. It may have different dimensions and structure than 'output'.
        
        Structure of prediction_data_byvar after predict_sortByUnivar:
            prediction_data_byvar['input'][channel_name][var_id] = [sample_values...]
            prediction_data_byvar['output'][channel_name][var_id] = [sample_values...]
        """
        resultDict = dict()
        self.predict(
            self.dataset, 
            time_key=time_key, 
            latent=(self.latent_dim is not None and not force_noLatent), 
            experiment_name=experiment_name, 
            remapping=remapping_data
        )
        
        prediction_data = self.getPred()
        prediction_data_byvar = self.getPred_byUnivar()
        
        resultDict["prediction_data_byvar"] = prediction_data_byvar
        
        # Input might be latent representation with different structure
        inp_data_vc = self._build_dataframe_flexible(
            prediction_data_byvar['input'],
            self.univar_count_in,
            self.isnoise_in,
            data_type="input"
        )
        resultDict["inp_data_vc"] = inp_data_vc
        
        # Output should have multi-channel structure
        out_data_vc = self._build_multiindex_dataframe(
            prediction_data_byvar['output'],
            self.univar_count_out,
            self.isnoise_out
        )
        resultDict["out_data_vc"] = out_data_vc
        
        if self.latent_dim is not None and not force_noLatent:
            lat_data = self.getLat()
            resultDict["latent_data"] = lat_data
            
            lat_data_bycomp = self.getLat_byComponent()
            resultDict["latent_data_bycomp"] = lat_data_bycomp['latent']
            
            lat2dataInput = self.getLatent2data(latent_key_toplot=latent_key_toplot)
            resultDict["latent_data_input"] = lat2dataInput
        return resultDict


    def _build_dataframe_flexible(self, data_structure, n_vars, is_noise, data_type="input"):
        """
        Build a DataFrame handling both multi-channel and single-channel structures.
        
        Args:
            data_structure: Can be either:
                - dict {channel_name: {var_id: [values]}} for multi-channel
                - dict {var_id: [values]} for single-channel
            n_vars: number of variables
            is_noise: if True, use numeric names; else use vc_mapping
            data_type: "input" or "output" for logging
            
        Returns:
            pd.DataFrame with appropriate column structure
        """
        # Detect structure: check if first key is a channel name or a var_id
        first_key = list(data_structure.keys())[0]
        
        # If first key is in key_value_list, it's multi-channel
        is_multichannel = first_key in self.key_value_list if hasattr(self, 'key_value_list') else False
        
        if is_multichannel:
            return self._build_multiindex_dataframe(data_structure, n_vars, is_noise)
        else:
            return self._build_simple_dataframe(data_structure, n_vars, is_noise)


    def _build_simple_dataframe(self, data_by_var, n_vars, is_noise):
        """
        Build a simple DataFrame for single-channel data (e.g., latent representations).
        
        Args:
            data_by_var: dict {var_id: [sample_values...]}
            n_vars: number of variables
            is_noise: if True, use numeric names; else use vc_mapping
            
        Returns:
            pd.DataFrame with simple columns
        """
        # Determine number of samples
        n_samples = None
        for var_id in range(n_vars):
            if var_id in data_by_var:
                n_samples = len(data_by_var[var_id])
                break
        
        if n_samples is None:
            raise ValueError("Could not determine number of samples from data")
        
        # Collect columns
        data_dict = {}
        
        for var_id in range(n_vars):
            if var_id not in data_by_var:
                continue
            
            # Get variable name
            if is_noise:
                var_name = f"{var_id}"
            else:
                # vc_mapping can be either dict or list
                if isinstance(self.vc_mapping, dict):
                    var_name = self.vc_mapping.get(var_id, f"var_{var_id}")
                elif isinstance(self.vc_mapping, (list, tuple)):
                    var_name = self.vc_mapping[var_id] if var_id < len(self.vc_mapping) else f"var_{var_id}"
                else:
                    var_name = f"var_{var_id}"
            
            # Get values
            values = data_by_var[var_id]
            
            # Convert to list of scalars
            values_list = []
            for v in values:
                if hasattr(v, 'item'):
                    values_list.append(v.item())
                elif isinstance(v, (list, tuple)):
                    if len(v) > 0:
                        values_list.append(float(v[0]) if hasattr(v[0], '__float__') else v[0])
                    else:
                        values_list.append(0.0)
                else:
                    values_list.append(float(v))
            
            data_dict[var_name] = values_list
        
        if not data_dict:
            raise ValueError("No data collected for DataFrame construction")
        
        # Check all columns have the same length
        lengths = {k: len(v) for k, v in data_dict.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            min_len = min(unique_lengths)
            data_dict = {k: v[:min_len] for k, v in data_dict.items()}
        
        # Create simple DataFrame
        df = pd.DataFrame(data_dict)
        
        return df


    def _build_multiindex_dataframe(self, data_by_channel, n_vars, is_noise):
        """
        Build a DataFrame with MultiIndex columns (channel, variable).
        
        Args:
            data_by_channel: dict {channel_name: {var_id: [sample_values...]}}
            n_vars: number of variables
            is_noise: if True, use numeric names; else use vc_mapping
            
        Returns:
            pd.DataFrame with MultiIndex columns
        """
        # First, determine the number of samples (should be the same for all)
        n_samples = None
        for ch_name in self.key_value_list:
            if ch_name in data_by_channel:
                for var_id in range(n_vars):
                    if var_id in data_by_channel[ch_name]:
                        n_samples = len(data_by_channel[ch_name][var_id])
                        break
                if n_samples is not None:
                    break
        
        if n_samples is None:
            raise ValueError("Could not determine number of samples from data")
        # Collect all columns as dict: {(channel, var_name): [values]}
        data_dict = {}
        
        for ch_name in self.key_value_list:
            if ch_name not in data_by_channel:
                continue
                
            for var_id in range(n_vars):
                if var_id not in data_by_channel[ch_name]:
                    continue
                
                # Get variable name
                if is_noise:
                    var_name = f"{var_id}"
                else:
                    # vc_mapping can be either dict or list
                    if isinstance(self.vc_mapping, dict):
                        var_name = self.vc_mapping.get(var_id, f"var_{var_id}")
                    elif isinstance(self.vc_mapping, (list, tuple)):
                        var_name = self.vc_mapping[var_id] if var_id < len(self.vc_mapping) else f"var_{var_id}"
                    else:
                        var_name = f"var_{var_id}"
                
                # Get values for this (channel, variable) combination
                values = data_by_channel[ch_name][var_id]
                
                # Convert to list of scalars
                values_list = []
                for v in values:
                    if hasattr(v, 'item'):
                        # numpy array or tensor with single value
                        values_list.append(v.item())
                    elif isinstance(v, (list, tuple)):
                        # If it's a list/tuple, take first element
                        if len(v) > 0:
                            values_list.append(float(v[0]) if hasattr(v[0], '__float__') else v[0])
                        else:
                            values_list.append(0.0)
                    else:
                        values_list.append(float(v))
                
                data_dict[(ch_name, var_name)] = values_list
        
        if not data_dict:
            raise ValueError("No data collected for DataFrame construction")
        
        # Check all columns have the same length
        lengths = {k: len(v) for k, v in data_dict.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            # Truncate all to minimum length
            min_len = min(unique_lengths)
            data_dict = {k: v[:min_len] for k, v in data_dict.items()}
        
        # Create MultiIndex
        multi_index = pd.MultiIndex.from_tuples(
            list(data_dict.keys()),
            names=["channel", "variable"]
        )
        
        # Create DataFrame directly from dict
        df = pd.DataFrame(data_dict)
        
        return df
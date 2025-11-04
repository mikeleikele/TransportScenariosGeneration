from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
import torch.nn as nn
import torch
from torch.functional import F
from torchmetrics.functional.regression import kendall_rank_corrcoef
from torchmetrics import SpearmanCorrCoef
from src.NeuroCorrelation.ModelTraining.LossCofficentsFunction import LossCofficentsFunction
from sklearn.decomposition import PCA
from torch.autograd import Function
from scipy.stats import spearmanr               
from scipy.special import kl_div
import numpy as np

class LossFunction(nn.Module):
    def __init__(self, loss_case, univar_count, latent_dim, device, key_value_list, batch_shape="vector"):
        self.loss_case = loss_case
        self.univar_count = univar_count  # numero di roads (variabili)
        self.latent_dim = latent_dim
        self.batch_shape = batch_shape
        self.device = device
        self.n_dimensions = len(key_value_list)  # numero di dimensioni per variabile
        self.key_value_list = key_value_list if key_value_list is not None else [f'var{i+1}' for i in range(n_dimensions)]
        self.statsData = None
        self.vc_mapping = None
        self.check_coeff = dict()
        
    def get_lossTerms(self):
        return self.loss_case
     
    def set_coefficent(self, epochs_tot, path_folder):
        self.loss_coeff = LossCofficentsFunction(self.loss_case, epochs_tot=epochs_tot, path_folder=path_folder)   
        
    def loss_change_coefficent(self, loss_name, loss_coeff):
        if loss_name in self.loss_case:
            self.loss_case[loss_name] = {'type': 'fixed', 'value': loss_coeff}
        self.loss_coeff.setCoefficents()
    
    def get_Loss_params(self):
        if len(self.loss_case) == 0:
            return {"loss_case":"-", "latent_dim":self.latent_dim, "univar_count":self.univar_count,
                    "batch_shape":self.batch_shape, "n_dimensions":self.n_dimensions, "key_value_list":self.key_value_list}
        else:
            return {"loss_case":self.loss_case, "latent_dim":self.latent_dim, "univar_count":self.univar_count,
                    "batch_shape":self.batch_shape, "n_dimensions":self.n_dimensions, "key_value_list":self.key_value_list}
    
    def set_stats_data(self, stats_data, vc_mapping):
        self.statsData = stats_data
        self.vc_mapping = vc_mapping
        
        
    def computate_loss(self, values_in, epoch, verbose=False):

        if self.statsData is None or self.vc_mapping is None:
            raise Exception("statsData NOT SET")
        values = values_in
        loss_total = torch.zeros(1).to(device=self.device)
        loss_dict = dict()
        coeff = self.loss_coeff.getCoefficents(epoch=epoch)
        self.check_coeff[epoch] = coeff
        
        if "MSE_LOSS" in self.loss_case:
            mse_similarities_dict = self.MSE_similarities(values)
            loss_dict["MSE_LOSS"] = {}
            for dim_key, mse_val in mse_similarities_dict.items():
                loss_coeff = mse_val.mul(coeff["MSE_LOSS"])
                loss_total += loss_coeff
                loss_dict["MSE_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"MSE_LOSS {dim_key} - {loss_coeff}")
        
        if "RMSE_LOSS" in self.loss_case:
            rmse_similarities_dict = self.RMSE_similarities(values)
            loss_dict["RMSE_LOSS"] = {}
            for dim_key, rmse_val in rmse_similarities_dict.items():
                loss_coeff = rmse_val.mul(coeff["RMSE_LOSS"])
                loss_total += loss_coeff
                loss_dict["RMSE_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"RMSE_LOSS {dim_key} - {loss_coeff}")
                
        if "MEDIAN_LOSS_batch" in self.loss_case:
            median_similarities_dict = self.median_similarities(values, compute_on="B")
            loss_dict["MEDIAN_LOSS_batch"] = {}
            for dim_key, median_val in median_similarities_dict.items():
                loss_coeff = median_val.mul(coeff["MEDIAN_LOSS_batch"])
                loss_total += loss_coeff
                loss_dict["MEDIAN_LOSS_batch"][dim_key] = loss_coeff
                if verbose:
                    print(f"MEDIAN_LOSS_batch {dim_key} - {loss_coeff}")
        
        if "MEDIAN_LOSS_dataset" in self.loss_case:
            median_similarities_dict = self.median_similarities(values, compute_on="D")
            loss_dict["MEDIAN_LOSS_dataset"] = {}
            for dim_key, median_val in median_similarities_dict.items():
                loss_coeff = median_val.mul(coeff["MEDIAN_LOSS_dataset"])
                loss_total += loss_coeff
                loss_dict["MEDIAN_LOSS_dataset"][dim_key] = loss_coeff
                if verbose:
                    print(f"MEDIAN_LOSS_dataset {dim_key} - {loss_coeff}")
        
        if "VARIANCE_LOSS" in self.loss_case:            
            variance_similarities_dict = self.variance_similarities(values)
            loss_dict["VARIANCE_LOSS"] = {}
            for dim_key, var_val in variance_similarities_dict.items():
                loss_coeff = var_val.mul(coeff["VARIANCE_LOSS"])
                loss_total += loss_coeff
                loss_dict["VARIANCE_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"VARIANCE_LOSS {dim_key} - {loss_coeff}")
        
        if "COVARIANCE_LOSS" in self.loss_case:            
            covariance_similarities_loss = self.covariance_similarities(values)
            loss_dict["COVARIANCE_LOSS"] = {}
            for dim_key, var_val in covariance_similarities_loss.items():
                loss_coeff = var_val.mul(coeff["COVARIANCE_LOSS"])
                loss_total += loss_coeff
                loss_dict["COVARIANCE_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"COVARIANCE_LOSS {dim_key} - {loss_coeff}")
        
        if "JENSEN_SHANNON_DIVERGENCE_LOSS" in self.loss_case:            
            jensen_shannon_divergence_loss = self.jensen_shannon_divergence(values)
            loss_dict["JENSEN_SHANNON_DIVERGENCE_LOSS"] = {}
            for dim_key, var_val in jensen_shannon_divergence_loss.items():
                loss_coeff = var_val.mul(coeff["JENSEN_SHANNON_DIVERGENCE_LOSS"])
                loss_total += loss_coeff
                loss_dict["JENSEN_SHANNON_DIVERGENCE_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"JENSEN_SHANNON_DIVERGENCE_LOSS {dim_key} - {loss_coeff}")

        if "SPEARMAN_CORRELATION_LOSS" in self.loss_case:            
            spearman_correlation_loss = self.spearman_correlation(values)
            loss_dict["SPEARMAN_CORRELATION_LOSS"] = {}
            for dim_key, var_val in spearman_correlation_loss.items():
                loss_coeff = var_val.mul(coeff["SPEARMAN_CORRELATION_LOSS"])
                loss_total += loss_coeff
                loss_dict["SPEARMAN_CORRELATION_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"SPEARMAN_CORRELATION_LOSS {dim_key} - {loss_coeff}")

        if "PEARSON_CORRELATION_LOSS" in self.loss_case:            
            pearson_correlation_loss = self.pearson_correlation(values)
            loss_dict["PEARSON_CORRELATION_LOSS"] = {}
            for dim_key, var_val in pearson_correlation_loss.items():
                loss_coeff = var_val.mul(coeff["PEARSON_CORRELATION_LOSS"])
                loss_total += loss_coeff
                loss_dict["PEARSON_CORRELATION_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"PEARSON_CORRELATION_LOSS {dim_key} - {loss_coeff}")
        
        if "KL_DIVERGENCE_LOSS" in self.loss_case:            
            kl_divergence_latent_loss = self.kl_divergence_latent(values)
            loss_dict["KL_DIVERGENCE_LOSS"] = {}
            for dim_key, var_val in kl_divergence_latent_loss.items():
                loss_coeff = var_val.mul(coeff["KL_DIVERGENCE_LOSS"])
                loss_total += loss_coeff
                loss_dict["KL_DIVERGENCE_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"KL_DIVERGENCE_LOSS {dim_key} - {loss_coeff}")
        
        if "CORRELATION_MATRICES_LOSS" in self.loss_case:            
            correlation_matrix_loss_loss = self.correlation_matrix_loss(values)
            loss_dict["CORRELATION_MATRICES_LOSS"] = {}
            for dim_key, var_val in correlation_matrix_loss_loss.items():
                loss_coeff = var_val.mul(coeff["CORRELATION_MATRICES_LOSS"])
                loss_total += loss_coeff
                loss_dict["CORRELATION_MATRICES_LOSS"][dim_key] = loss_coeff
                if verbose:
                    print(f"KL_DIVERGENCORRELATION_MATRICES_LOSSCE_LOSS {dim_key} - {loss_coeff}")

        if verbose:
            print("loss_total - ", loss_total)
        loss_dict["loss_total"] = loss_total
        return loss_dict

    def MSE_similarities(self, values):
        """
        Calcola MSE tra input e output per ogni dimensione.
        
        Ritorna un dizionario con chiavi da self.key_value_list:
        {
            key_value_list[0]: MSE calcolato su tutte le roads per la prima dimensione,
            key_value_list[1]: MSE calcolato su tutte le roads per la seconda dimensione,
            ...
        }
        """
        loss_dict = {}
        
        batch_input = []
        batch_output = []
        for val in values:
            batch_input.append(val['x_input']['data'])
            batch_output.append(val['x_output']['data'])
        
        batch_input = torch.stack(batch_input, dim=0)
        batch_output = torch.stack(batch_output, dim=0)
        
        for dim_idx in range(self.n_dimensions):
            loss_per_dim = torch.zeros(1).to(device=self.device)
            for road_idx in range(self.univar_count):
                input_vals = batch_input[:, road_idx, dim_idx]  # shape: [batch_size]
                output_vals = batch_output[:, road_idx, dim_idx]  # shape: [batch_size]
                
                mse = torch.mean((input_vals - output_vals) ** 2)
                loss_per_dim += mse
            
            loss_per_dim /= self.univar_count
            loss_dict[self.key_value_list[dim_idx]] = loss_per_dim
        return loss_dict
    

    def RMSE_similarities(self, values):
        """
        Calcola RMSE tra input e output per ogni dimensione.
        
        Ritorna un dizionario con chiavi da self.key_value_list.
        """
        loss_dict = {}
        
        # Raccogli tutti i dati del batch
        batch_input = []
        batch_output = []
        for val in values:
            batch_input.append(val['x_input']['data'])
            batch_output.append(val['x_output']['data'])
        
        # Stack: [batch_size, n_roads, n_dimensions]
        batch_input = torch.stack(batch_input, dim=0)
        batch_output = torch.stack(batch_output, dim=0)
        
        # Calcola RMSE per ogni dimensione
        for dim_idx in range(self.n_dimensions):
            mse_sum = torch.zeros(1).to(device=self.device)
            
            for road_idx in range(self.univar_count):
                input_vals = batch_input[:, road_idx, dim_idx]
                output_vals = batch_output[:, road_idx, dim_idx]
                
                mse = torch.mean((input_vals - output_vals) ** 2)
                mse_sum += mse
            
            # RMSE = sqrt(media degli MSE)
            rmse = torch.sqrt(mse_sum / self.univar_count)
            loss_dict[self.key_value_list[dim_idx]] = rmse
        
        return loss_dict

 
    def median_similarities(self, values, compute_on="batch"):
        """
        Calcola la differenza tra le mediane di input e output per ogni dimensione.
        
        Ritorna un dizionario con chiavi da self.key_value_list.
        """
        loss_dict = {}
        
        # Raccogli tutti i dati del batch
        batch_input = []
        batch_output = []
        for val in values:
            batch_input.append(val['x_input']['data'])
            batch_output.append(val['x_output']['data'])
        
        # Stack: [batch_size, n_roads, n_dimensions]
        batch_input = torch.stack(batch_input, dim=0)
        batch_output = torch.stack(batch_output, dim=0)
        
        # Calcola mediana per ogni dimensione
        for dim_idx in range(self.n_dimensions):
            loss_per_dim = torch.zeros(1).to(device=self.device)
            dim_key = self.key_value_list[dim_idx]
            
            for road_idx in range(self.univar_count):
                input_vals = batch_input[:, road_idx, dim_idx]  # shape: [batch_size]
                output_vals = batch_output[:, road_idx, dim_idx]  # shape: [batch_size]
                
                if compute_on == "batch" or compute_on == "B":
                    median_in = torch.median(input_vals)
                elif compute_on == "dataset" or compute_on == "D":
                    # Statistiche pre-calcolate dal dataset
                    if isinstance(self.statsData['median_val'], dict):
                        key = f"road{road_idx}_{dim_key}"
                        median_in = torch.tensor(self.statsData['median_val'].get(key, 0.0), device=self.device)
                    elif isinstance(self.statsData['median_val'], (list, np.ndarray, torch.Tensor)):
                        median_in = torch.tensor(self.statsData['median_val'][road_idx][dim_idx], device=self.device)
                
                median_out = torch.median(output_vals)
                
                # Differenza quadratica
                loss_per_dim += (median_in - median_out) ** 2
            
            loss_dict[dim_key] = loss_per_dim
        
        return loss_dict


    def variance_similarities(self, values, compute_on="batch"):
        """
        Calcola la differenza tra le varianze (std) di input e output per ogni dimensione.
        
        Ritorna un dizionario con chiavi da self.key_value_list.
        """
        loss_dict = {}
        
        # Raccogli tutti i dati del batch
        batch_input = []
        batch_output = []
        for val in values:
            batch_input.append(val['x_input']['data'])
            batch_output.append(val['x_output']['data'])
        
        # Stack: [batch_size, n_roads, n_dimensions]
        batch_input = torch.stack(batch_input, dim=0)
        batch_output = torch.stack(batch_output, dim=0)
        
        # Calcola std per ogni dimensione
        for dim_idx in range(self.n_dimensions):
            loss_per_dim = torch.zeros(1).to(device=self.device)
            dim_key = self.key_value_list[dim_idx]
            
            for road_idx in range(self.univar_count):
                input_vals = batch_input[:, road_idx, dim_idx]  # shape: [batch_size]
                output_vals = batch_output[:, road_idx, dim_idx]  # shape: [batch_size]
                
                if compute_on == "batch" or compute_on == "B":
                    std_in = torch.std(input_vals)
                elif compute_on == "dataset" or compute_on == "D":
                    # Statistiche pre-calcolate dal dataset
                    if isinstance(self.statsData['variance_val'], dict):
                        key = f"road{road_idx}_{dim_key}"
                        std_in = torch.tensor(self.statsData['variance_val'].get(key, 0.0), device=self.device)
                    elif isinstance(self.statsData['variance_val'], (list, np.ndarray, torch.Tensor)):
                        std_in = torch.tensor(self.statsData['variance_val'][road_idx][dim_idx], device=self.device)
                
                std_out = torch.std(output_vals)
                
                # Differenza quadratica
                loss_per_dim += (std_in - std_out) ** 2
            
            loss_dict[dim_key] = loss_per_dim
        
        return loss_dict


    def covariance_similarities(self, values):
        """
        Calcola, per ogni dimensione canale, la distanza tra la matrice di covarianza
        di input e output (sulle R variabili) usando le B osservazioni del batch.

        Input format: same as MSE_similarities / RMSE_similarities
        values: list[ item ] where item['x_input']['data'] and item['x_output']['data']
                have shape (n_roads, n_dimensions)

        Ritorna:
        dict { key_value_list[dim_idx]: loss_tensor } con loss_tensor scalare su device
        """
        loss_ret = {}

        # stack: (B, R, D)
        input_list = [v['x_input']['data'] for v in values]
        output_list = [v['x_output']['data'] for v in values]
        batch_input = torch.stack(input_list, dim=0).to(self.device)   # (B, R, D)
        batch_output = torch.stack(output_list, dim=0).to(self.device) # (B, R, D)

        B, R, D = batch_input.shape

        # per-dimension covariance difference (Frobenius norm squared, normalized)
        for dim_idx in range(self.n_dimensions):
            # seleziona (B, R) e trasponi a (R, B) per torch.cov
            mat_in = batch_input[:, :, dim_idx].transpose(0, 1).contiguous()   # (R, B)
            mat_out = batch_output[:, :, dim_idx].transpose(0, 1).contiguous() # (R, B)

            if mat_in.size(1) < 2:
                # se non ci sono abbastanza osservazioni, usare matrice zero (no info)
                cov_in = torch.zeros((R, R), device=self.device)
                cov_out = torch.zeros((R, R), device=self.device)
            else:
                cov_in = torch.cov(mat_in, correction=1)   # (R, R)
                cov_out = torch.cov(mat_out, correction=1) # (R, R)

            diff = cov_in - cov_out
            # Frobenius norm squared (più stabile e comparable tra dimensioni)
            fro_sq = torch.norm(diff, p='fro') ** 2

            # normalizzazione opzionale: divido per R^2 so that scale is comparable across different R
            norm = float(R * R) if R > 0 else 1.0
            loss_val = fro_sq / norm

            # normalize shape to [1] and keep grad_fn
            if not isinstance(loss_val, torch.Tensor):
                loss_val = torch.tensor([loss_val], device=self.device, dtype=torch.float32)
            else:
                if loss_val.dim() == 0:
                    loss_val = loss_val.unsqueeze(0)
                else:
                    loss_val = loss_val.view(-1)[:1]

            loss_ret[self.key_value_list[dim_idx]] = loss_val

        return loss_ret


    def spearman_correlation(self, values, method='differentiable', aggr='fro'):
        """
        Per ogni canale D calcola la matrice di correlazione di Spearman su B osservazioni
        delle R variabili e misura la distanza input vs output.

        Args:
        values: list di elementi con item['x_input']['data'] e item['x_output']['data']
                ogni data tensor per sample: (R, D) o (R,)
        method: 'differentiable' (soft-rank, differenziabile) oppure 'scipy' (spearmanr non-diff)
        aggr: 'fro' -> norma di Frobenius; 'fro_sq' -> norma di Frobenius al quadrato

        Returns:
        dict mapping self.key_value_list[d] -> scalar Tensor (sul device)
        """
        input_list = [v['x_input']['data'] for v in values]
        output_list = [v['x_output']['data'] for v in values]
        batch_input = torch.stack(input_list, dim=0).to(self.device)
        batch_output = torch.stack(output_list, dim=0).to(self.device)

        if batch_input.dim() == 2:
            batch_input = batch_input.unsqueeze(-1)
            batch_output = batch_output.unsqueeze(-1)

        B, R, D = batch_input.shape
        loss_dict = {}

        for d in range(D):
            X_in = batch_input[:, :, d]   # (B, R)
            X_out = batch_output[:, :, d] # (B, R)

            if method == 'differentiable':
                S_in = self.spearman_correlation_differentiable(X_in)
                S_out = self.spearman_correlation_differentiable(X_out)
            elif method == 'scipy':                
                S_in_np = spearmanr(X_in.detach().cpu().numpy(), axis=0).correlation
                S_out_np = spearmanr(X_out.detach().cpu().numpy(), axis=0).correlation
                S_in = torch.tensor(S_in_np, device=self.device, dtype=X_in.dtype)
                S_out = torch.tensor(S_out_np, device=self.device, dtype=X_out.dtype)
                S_in = torch.nan_to_num(S_in, nan=0.0, posinf=0.0, neginf=0.0)
                S_out = torch.nan_to_num(S_out, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                raise ValueError("method must be 'differentiable' or 'scipy'")

            diff = S_in - S_out
            if aggr == 'fro':
                val = torch.norm(diff, p='fro')
            elif aggr == 'fro_sq':
                val = torch.norm(diff, p='fro') ** 2
            else:
                raise ValueError("aggr must be 'fro' or 'fro_sq'")

            # ensure val is Tensor with shape [1] and preserve grad_fn when present
            if not isinstance(val, torch.Tensor):
                val = torch.tensor([val], device=self.device, dtype=torch.float32)
            else:
                if val.dim() == 0:
                    val = val.unsqueeze(0)
                else:
                    # collapse to single-element 1-D tensor if accidentally larger
                    val = val.view(-1)[:1]

            loss_dict[self.key_value_list[d]] = val

        return loss_dict
    
    def spearman_correlation_differentiable(self, data, softmax_temp=1.0):
        """
        Differentiable pseudo-Spearman correlation.

        Args:
        data: Tensor shape (n_samples, n_features)  (es. B x R)
        softmax_temp: temperatura per la softmax (più piccolo -> ranghi più netti)

        Returns:
        correlation_matrix: Tensor shape (n_features, n_features) su self.device
        """
        eps = 1e-8
        n, p = data.shape
        mx = torch.max(data, dim=0, keepdim=True).values
        ex = torch.exp((data - mx) / (softmax_temp + eps))
        soft_p = ex / (torch.sum(ex, dim=0, keepdim=True) + eps)   # (n, p)
        cdf = torch.cumsum(soft_p, dim=0)                           # (n, p)
        expected_rank = 1.0 + (n - 1.0) * cdf                       # (n, p)
        ranks_centered = expected_rank - torch.mean(expected_rank, dim=0, keepdim=True)
        cov = (ranks_centered.t() @ ranks_centered) / max(n - 1, 1)
        diag = torch.diagonal(cov)
        stds = torch.sqrt(torch.clamp(diag, min=eps))
        corr = cov / (torch.outer(stds, stds) + eps)
        corr = torch.clamp(corr, -1.0, 1.0)
        return corr


    def pearson_correlation(self, values, aggr='fro', channel_reduce=None):
        """
        Calcola, per ogni canale D separatamente, la differenza tra matrici di
        correlazione di Pearson (input vs output) e restituisce dict scalare per canale.

        Args:
        values: list of items with item['x_input']['data'] and item['x_output']['data']
                each data tensor per sample: (R, D) or (R,)
        aggr: 'fro' -> Frobenius norm; 'fro_sq' -> Frobenius norm squared
        channel_reduce: None | 'mean'|'sum'|'first' -> se ogni sample ha shape (R, D) ma vuoi 
                        ridurre D->1 prima (non tipico quando D rappresenta canali distinti)
        Returns:
        dict mapping self.key_value_list[d] -> scalar tensor on self.device
        """
        input_list = [v['x_input']['data'] for v in values]
        output_list = [v['x_output']['data'] for v in values]
        batch_input = torch.stack(input_list, dim=0).to(self.device)
        batch_output = torch.stack(output_list, dim=0).to(self.device)

        # Normalizza shape a (B, R, D)
        if batch_input.dim() == 2:
            batch_input = batch_input.unsqueeze(-1)
            batch_output = batch_output.unsqueeze(-1)

        B, R, D = batch_input.shape
        loss_dict = {}

        # Optional channel reduction (if user really wants to reduce channels before per-channel Pearson)
        def maybe_reduce_channel(t):
            # t : (R, D)
            if channel_reduce is None:
                return t
            if t.dim() == 1:
                return t
            if channel_reduce == 'mean':
                return t.mean(dim=1)        # -> (R,)
            if channel_reduce == 'sum':
                return t.sum(dim=1)
            if channel_reduce == 'first':
                return t[:, 0]
            raise ValueError("channel_reduce must be None|'mean'|'sum'|'first'")

        for d in range(D):
            X_in = batch_input[:, :, d]   # (B, R)
            X_out = batch_output[:, :, d] # (B, R)

            # se richiesto, riduci canali (ma tipicamente channel_reduce è None)
            X_in = maybe_reduce_channel(X_in) if channel_reduce else X_in
            X_out = maybe_reduce_channel(X_out) if channel_reduce else X_out

            # pearson_corr si aspetta (n_samples, n_features) -> qui (B, R)
            S_in = self.pearson_corr(X_in)
            S_out = self.pearson_corr(X_out)

            diff = S_in - S_out
            if aggr == 'fro':
                val = torch.norm(diff, p='fro')
            elif aggr == 'fro_sq':
                val = torch.norm(diff, p='fro') ** 2
            else:
                raise ValueError("aggr must be 'fro' or 'fro_sq'")

            # assicuriamoci scalar tensor sul device
            # ensure val is Tensor with shape [1] and preserve grad_fn when present
            if not isinstance(val, torch.Tensor):
                val = torch.tensor([val], device=self.device, dtype=torch.float32)
            else:
                if val.dim() == 0:
                    val = val.unsqueeze(0)
                else:
                    # collapse to single-element 1-D tensor if accidentally larger
                    val = val.view(-1)[:1]

            loss_dict[self.key_value_list[d]] = val

        return loss_dict

    def pearson_corr(self, data):
        """
        Calcola la matrice di correlazione di Pearson in modo numericamente stabile.

        Args:
        data: Tensor shape (n_samples, n_features)  (es. B x R)
        Returns:
        correlation_matrix: Tensor shape (n_features, n_features) su self.device
        """
        eps = 1e-12
        data = data.to(self.device)

        # mean per feature (colonna)
        means = torch.mean(data, dim=0, keepdim=True)

        # centered data
        centered = data - means  # (n, p)

        # cov matrix (p x p)
        n = max(data.size(0), 1)
        cov_matrix = (centered.t() @ centered) / max(n - 1, 1)

        # std devs
        diag = torch.diagonal(cov_matrix)
        std_devs = torch.sqrt(torch.clamp(diag, min=eps))

        # outer product denom, evitamo divisioni per zero
        denom = torch.outer(std_devs, std_devs)
        correlation_matrix = cov_matrix / (denom + eps)

        # clamp numericamente a [-1,1]
        correlation_matrix = torch.clamp(correlation_matrix, -1.0, 1.0)

        return correlation_matrix


    def jensen_shannon_divergence(self, values, reduction='mean'):
        """
        Calcola la divergenza di Jensen-Shannon tra input e output per ogni dimensione.
        
        La JSD misura la similarità tra due distribuzioni di probabilità.
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), dove M = 0.5 * (P + Q)
        
        Args:
            values: list di elementi con item['x_input']['data'] e item['x_output']['data']
                    ogni data tensor per sample: (R, D) dove R=n_roads, D=n_dimensions
            reduction: 'mean' -> media su roads; 'sum' -> somma su roads
        
        Returns:
            dict mapping self.key_value_list[d] -> scalar Tensor (sul device)
        """
        loss_dict = {}
        kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        eps = 1e-8
        
        # Raccogli batch: (B, R, D)
        batch_input = []
        batch_output = []
        for val in values:
            batch_input.append(val['x_input']['data'])
            batch_output.append(val['x_output']['data'])
        
        batch_input = torch.stack(batch_input, dim=0).to(self.device)
        batch_output = torch.stack(batch_output, dim=0).to(self.device)
        
        # Se shape è (B, R) -> (B, R, 1)
        if batch_input.dim() == 2:
            batch_input = batch_input.unsqueeze(-1)
            batch_output = batch_output.unsqueeze(-1)
        
        B, R, D = batch_input.shape
        
        # Calcola JSD per ogni dimensione separatamente
        for dim_idx in range(D):
            jsd_sum = torch.zeros(1, device=self.device)
            
            for road_idx in range(R):
                # Estrai dati per questa road e dimensione: shape (B,)
                x_in = batch_input[:, road_idx, dim_idx]
                x_out = batch_output[:, road_idx, dim_idx]
                
                # Reshape a (B, 1) per softmax
                x_in = x_in.unsqueeze(-1)
                x_out = x_out.unsqueeze(-1)
                
                # Converti in distribuzioni di probabilità
                p = F.softmax(x_in, dim=0)  # normalizza su batch
                q = F.softmax(x_out, dim=0)
                
                # Calcola distribuzione media M
                m = 0.5 * (p + q)
                m = m.clamp(min=eps)
                
                # Calcola JSD = 0.5 * [KL(P||M) + KL(Q||M)]
                kl_pm = kl(p.log(), m.log())
                kl_qm = kl(q.log(), m.log())
                jsd_value = 0.5 * (kl_pm + kl_qm)
                
                jsd_sum += jsd_value
            
            # Applica riduzione
            if reduction == 'mean':
                loss_val = jsd_sum / R
            elif reduction == 'sum':
                loss_val = jsd_sum
            else:
                raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction}")
            
            # Assicura che sia un tensor [1] con grad_fn preservato
            if not isinstance(loss_val, torch.Tensor):
                loss_val = torch.tensor([loss_val], device=self.device, dtype=torch.float32)
            else:
                if loss_val.dim() == 0:
                    loss_val = loss_val.unsqueeze(0)
                else:
                    loss_val = loss_val.view(-1)[:1]
            
            loss_dict[self.key_value_list[dim_idx]] = loss_val
        
        return loss_dict


    def correlation_matrix_loss(self, values, aggregation="MAE", correlation_matrix_mode="all", 
                            correlation_matrix_sparsify=False, sliding_window_size=100, 
                            sparsify_threshold=0.05, compute_on="batch"):
        """
        Calcola la differenza tra matrici di correlazione di input e output per ogni dimensione.
        
        La matrice di correlazione cattura le relazioni tra le R roads/variabili.
        Questa loss penalizza differenze nella struttura di correlazione tra input e output.
        
        Args:
            values: list di elementi con item['x_input']['data'] e item['x_output']['data']
                    ogni data tensor per sample: (R, D)
            aggregation: metodo per aggregare le differenze - "MAE", "MSE", o "RMSE"
            correlation_matrix_mode: "all" (usa tutte le osservazioni), "feature_selection", 
                                    o "sliding_window"
            correlation_matrix_sparsify: bool, se True applica sparsification alla matrice
            sliding_window_size: dimensione finestra per mode="sliding_window"
            sparsify_threshold: soglia per sparsification (valori sotto vengono azzerati)
            compute_on: "batch" o "dataset" (non usato in questa versione)
        
        Returns:
            dict mapping self.key_value_list[d] -> scalar Tensor (sul device)
        """
        loss_dict = {}
        
        # Converti string "False"/"True" in bool se necessario
        if isinstance(correlation_matrix_sparsify, str):
            correlation_matrix_sparsify = correlation_matrix_sparsify.lower() in ['true', '1', 'yes']
        
        # Raccogli batch: (B, R, D)
        batch_input = []
        batch_output = []
        for val in values:
            batch_input.append(val['x_input']['data'])
            batch_output.append(val['x_output']['data'])
        
        batch_input = torch.stack(batch_input, dim=0).to(self.device)
        batch_output = torch.stack(batch_output, dim=0).to(self.device)
        
        # Se shape è (B, R) -> (B, R, 1)
        if batch_input.dim() == 2:
            batch_input = batch_input.unsqueeze(-1)
            batch_output = batch_output.unsqueeze(-1)
        
        B, R, D = batch_input.shape
        
        # Calcola correlation matrix loss per ogni dimensione
        for dim_idx in range(D):
            # Estrai dati per questa dimensione: (B, R)
            x_in = batch_input[:, :, dim_idx]   # shape: [B, R]
            x_out = batch_output[:, :, dim_idx]  # shape: [B, R]
            
            # Calcola matrici di correlazione (R, R)
            corr_in = self.get_correlation_matrix(
                x_in, 
                mode=correlation_matrix_mode,
                sparsify=correlation_matrix_sparsify,
                window_size=sliding_window_size,
                sparsify_threshold=sparsify_threshold
            )
            
            corr_out = self.get_correlation_matrix(
                x_out,
                mode=correlation_matrix_mode,
                sparsify=correlation_matrix_sparsify,
                window_size=sliding_window_size,
                sparsify_threshold=sparsify_threshold
            )
            
            # Calcola differenza secondo aggregation
            if aggregation == "MSE":
                loss_val = torch.mean((corr_in - corr_out) ** 2)
            elif aggregation == "RMSE":
                loss_val = torch.sqrt(torch.mean((corr_in - corr_out) ** 2))
            elif aggregation == "MAE":
                loss_val = torch.mean(torch.abs(corr_in - corr_out))
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
            
            # Assicura che sia un tensor [1] con grad_fn preservato
            if not isinstance(loss_val, torch.Tensor):
                loss_val = torch.tensor([loss_val], device=self.device, dtype=torch.float32)
            else:
                if loss_val.dim() == 0:
                    loss_val = loss_val.unsqueeze(0)
                else:
                    loss_val = loss_val.view(-1)[:1]
            
            loss_dict[self.key_value_list[dim_idx]] = loss_val
        
        return loss_dict

    def get_correlation_matrix(self, val_matr, mode, sparsify=False, window_size=100, 
                            sparsify_threshold=0.05):
        """
        Calcola la matrice di correlazione secondo il mode specificato.
        
        Args:
            val_matr: Tensor shape (B, R) dove B=batch_size, R=n_roads
            mode: "all", "feature_selection", o "sliding_window"
            sparsify: bool, se True applica sparsification
            window_size: per mode="sliding_window"
            sparsify_threshold: soglia per sparsification
        
        Returns:
            correlation_matrix: Tensor shape (R, R)
        """
        if mode == "all":
            correlation_mat = self.compute_correlation_matrix(val_matr)
        elif mode == "feature_selection":
            correlation_mat = self.correlation_matrix_by_feature_selection(val_matr)
        elif mode == "sliding_window":
            correlation_mat = self.correlation_matrix_by_sliding_window(val_matr, window_size=window_size)
        else:
            raise ValueError(f"Unknown correlation_matrix_mode: {mode}")
        
        if sparsify:
            correlation_mat = self.sparsify_correlation_matrix(correlation_mat, threshold=sparsify_threshold)
        
        return correlation_mat

    def compute_correlation_matrix(self, val_matr):
        """
        Calcola la matrice di correlazione di Pearson.
        
        Args:
            val_matr: Tensor shape (B, R) dove B=n_samples, R=n_features/roads
        
        Returns:
            corr_matrix: Tensor shape (R, R)
        """
        eps = 1e-8
        
        # Centra i dati (sottrai media per feature)
        val_matr_centered = val_matr - val_matr.mean(dim=0, keepdim=True)
        
        # Matrice di covarianza
        n_samples = val_matr.shape[0]
        cov_matrix = torch.matmul(val_matr_centered.T, val_matr_centered) / max(n_samples - 1, 1)
        
        # Deviazioni standard
        std = torch.sqrt(torch.diag(cov_matrix).clamp(min=eps))
        
        # Matrice di correlazione
        std_matrix = std.unsqueeze(1) @ std.unsqueeze(0)
        corr_matrix = cov_matrix / (std_matrix + eps)
        
        # Clamp a [-1, 1] per stabilità numerica
        corr_matrix = torch.clamp(corr_matrix, -1.0, 1.0)
        
        return corr_matrix

    def correlation_matrix_by_feature_selection(self, val_matr):
        """
        Calcola matrice di correlazione con feature selection.
        
        Seleziona le features più rilevanti basandosi sulla varianza
        e calcola la correlazione solo su quelle.
        
        Args:
            val_matr: Tensor shape (B, R)
        
        Returns:
            corr_matrix: Tensor shape (R, R)
        """
        # Calcola varianza per feature
        variance = torch.var(val_matr, dim=0)
        
        # Seleziona top 50% features per varianza
        n_features = val_matr.shape[1]
        n_selected = max(2, n_features // 2)
        
        _, top_indices = torch.topk(variance, k=n_selected)
        selected_data = val_matr[:, top_indices]
        
        # Calcola correlazione sulle features selezionate
        corr_selected = self.compute_correlation_matrix(selected_data)
        
        # Ricostruisci matrice completa (R, R) inserendo zeri per features non selezionate
        corr_matrix = torch.zeros(n_features, n_features, device=self.device)
        
        for i, idx_i in enumerate(top_indices):
            for j, idx_j in enumerate(top_indices):
                corr_matrix[idx_i, idx_j] = corr_selected[i, j]
        
        return corr_matrix

    def correlation_matrix_by_sliding_window(self, val_matr, window_size=100):
        """
        Calcola matrice di correlazione usando sliding window.
        
        Usa solo le ultime window_size osservazioni per calcolare la correlazione.
        
        Args:
            val_matr: Tensor shape (B, R)
            window_size: numero di osservazioni da considerare
        
        Returns:
            corr_matrix: Tensor shape (R, R)
        """
        n_samples = val_matr.shape[0]
        
        # Usa solo le ultime window_size osservazioni
        if n_samples > window_size:
            windowed_data = val_matr[-window_size:, :]
        else:
            windowed_data = val_matr
        
        return self.compute_correlation_matrix(windowed_data)

    def sparsify_correlation_matrix(self, correlation_mat, threshold=0.05):
        """
        Applica sparsification alla matrice di correlazione.
        
        Azzera valori sotto la soglia per enfatizzare correlazioni forti.
        
        Args:
            correlation_mat: Tensor shape (R, R)
            threshold: soglia assoluta sotto cui azzerare
        
        Returns:
            sparse_corr_matrix: Tensor shape (R, R)
        """
        # Crea maschera per valori sopra soglia (in valore assoluto)
        mask = torch.abs(correlation_mat) >= threshold
        
        # Applica maschera
        sparse_matrix = correlation_mat * mask.float()
        
        # Mantieni diagonale (autocorrelazione = 1)
        n = correlation_mat.shape[0]
        eye_mask = torch.eye(n, device=self.device, dtype=torch.bool)
        sparse_matrix[eye_mask] = correlation_mat[eye_mask]
        
        return sparse_matrix


    def kl_divergence_latent(self, values, reduction='mean', use_dimensions=False, use_channels=True):
        """
        Calcola la divergenza KL dello spazio latente rispetto a una distribuzione Gaussiana standard.
        
        Questa è la regolarizzazione KL tipica dei Variational Autoencoders (VAE):
        KL(q(z|x) || p(z)) dove p(z) = N(0, I)
        
        Formula: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        Args:
            values: list di elementi con item['x_latent']['mu'] e item['x_latent']['logvar']
                    ogni mu/logvar tensor: shape (latent_dim,) o (R, latent_dim) o altro
            reduction: 'mean' -> media sul batch; 'sum' -> somma sul batch; 'none' -> per sample
            use_dimensions: bool, se True ritorna dict per dimensione, altrimenti valore singolo
            use_channels: bool, se True ritorna dict per canale, altrimenti valore singolo
        Returns:
            Se use_dimensions=False: 
                dict con chiave 'latent_kl' -> scalar Tensor
            Se use_dimensions=True:
                dict mapping self.key_value_list[d] -> scalar Tensor per ogni dimensione
                (se applicabile alla struttura del latent space)
        """
        # Estrai mu e logvar dal batch
        latent_mu_list = []
        latent_logvar_list = []
        
        for value in values:
            if 'x_latent' in value and 'mu' in value['x_latent'] and 'logvar' in value['x_latent']:
                latent_mu_list.append(value['x_latent']['mu'])
                latent_logvar_list.append(value['x_latent']['logvar'])
            else:
                raise KeyError("Missing 'x_latent' with 'mu' and 'logvar' in values")
        
        # Stack: shape dipende dalla struttura (tipicamente [B, latent_dim])
        latent_mu = torch.stack(latent_mu_list, dim=0).to(self.device)
        latent_logvar = torch.stack(latent_logvar_list, dim=0).to(self.device)
        
        # Formula VAE standard: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # Calcoliamo elemento per elemento
        kl_per_element = -0.5 * (1 + latent_logvar - latent_mu.pow(2) - latent_logvar.exp())
        
        if not use_dimensions:
            # Comportamento standard: riduzione su tutte le dimensioni
            # kl_per_element shape: [B, latent_dim] o [B, R, latent_dim] etc.
            
            if reduction == 'mean':
                # Media su batch e latent dimensions
                loss_val = torch.mean(kl_per_element)
            elif reduction == 'sum':
                # Somma su latent dims, media su batch
                kl_per_sample = torch.sum(kl_per_element, dim=-1)  # somma su latent_dim
                loss_val = torch.mean(kl_per_sample)  # media su batch
            elif reduction == 'none':
                # Per sample (media su latent dims)
                loss_val = torch.mean(kl_per_element, dim=-1)  # [B]
            else:
                raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
            
            # Assicura shape [1] per compatibilità
            if loss_val.dim() == 0:
                loss_val = loss_val.unsqueeze(0)
            elif loss_val.dim() > 1:
                loss_val = torch.mean(loss_val).unsqueeze(0)
            if use_channels:
                ret_loss = dict()
                for i, key in enumerate(self.key_value_list):
                    ret_loss[key] = loss_val
                return ret_loss
            else:
                return {'latent_kl': loss_val}
        
        else:
            # Modalità per dimensione: calcola KL separatamente per ogni dimensione latente
            # Utile se vogliamo monitorare contributo di diverse parti del latent space
            loss_dict = {}
            
            # Assumiamo kl_per_element shape: [B, latent_dim]
            if kl_per_element.dim() > 2:
                # Se shape più complessa, flatten intermedio
                B = kl_per_element.shape[0]
                kl_per_element = kl_per_element.view(B, -1)
            
            n_latent_dims = kl_per_element.shape[-1]
            
            # Se abbiamo key_value_list, usiamolo; altrimenti crea keys generiche
            if n_latent_dims <= len(self.key_value_list):
                keys = self.key_value_list[:n_latent_dims]
            else:
                keys = [f'latent_dim_{i}' for i in range(n_latent_dims)]
            
            for dim_idx in range(n_latent_dims):
                kl_this_dim = kl_per_element[:, dim_idx]  # [B]
                
                if reduction == 'mean':
                    loss_val = torch.mean(kl_this_dim)
                elif reduction == 'sum':
                    loss_val = torch.sum(kl_this_dim)
                elif reduction == 'none':
                    loss_val = kl_this_dim  # mantieni [B]
                else:
                    raise ValueError(f"reduction must be 'mean', 'sum', or 'none'")
                
                # Assicura shape [1] (tranne per reduction='none')
                if reduction != 'none':
                    if not isinstance(loss_val, torch.Tensor):
                        loss_val = torch.tensor([loss_val], device=self.device, dtype=torch.float32)
                    else:
                        if loss_val.dim() == 0:
                            loss_val = loss_val.unsqueeze(0)
                        else:
                            loss_val = loss_val.view(-1)[:1]
                
                loss_dict[keys[dim_idx]] = loss_val
            
            return loss_dict

#--
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


class LossFunction(nn.Module):
    def __init__(self, loss_case, univar_count, latent_dim, device, batch_shape="vector"):
        self.loss_case = loss_case
        self.univar_count = univar_count
        self.latent_dim = latent_dim
        self.batch_shape = batch_shape
        self.device = device
        self.first = 0
    
    def loss_change_coefficent(self, loss_name, loss_coeff):
        if loss_name in self.loss_case:
            self.loss_case[loss_name] = loss_coeff
    
    def get_Loss_params(self):
        return {"loss_case":self.loss_case, "latent_dim":self.latent_dim, "univar_count":self.univar_count,"batch_shape":self.batch_shape}
    
    def computate_loss(self, values_in, verbose=False):
        self.first += 1
        if self.batch_shape =="vecdtor":
            
            if self.first==3:
                print("======= VALUES LOSS : begin ================")
                #print(values)
                print("======= VALUES LOSS    end  ================")
    
        values = values_in
        loss_total = torch.zeros(1).to(device=self.device)
        loss_dict = dict()
        if "MSE_LOSS" in self.loss_case:
            reconstructed_similarities_loss = self.MSE_similarities(values)
            coeff = self.loss_case["MSE_LOSS"]
            loss_coeff = reconstructed_similarities_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["MSE_LOSS"] = loss_coeff
            if verbose:
                print("MSE_LOSS - ", loss_coeff)
        
        if "RMSE_LOSS" in self.loss_case:
            reconstructed_similarities_loss = self.RMSE_similarities(values)
            coeff = self.loss_case["RMSE_LOSS"]
            loss_coeff = reconstructed_similarities_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["RMSE_LOSS"] = loss_coeff
            if verbose:
                print("RMSE_LOSS - ", loss_coeff)
                
        if "MEDIAN_LOSS" in self.loss_case:
            median_similarities_loss = self.median_similarities(values)
            coeff = self.loss_case["MEDIAN_LOSS"]
            loss_coeff = median_similarities_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["MEDIAN_LOSS"] = loss_coeff
            if verbose:
                print("MEDIAN_LOSS - ", loss_coeff)
        
        if "VARIANCE_LOSS" in self.loss_case:            
            variance_similarities_loss = self.variance_similarities(values)
            coeff = self.loss_case["VARIANCE_LOSS"]
            loss_coeff = variance_similarities_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["VARIANCE_LOSS"] = loss_coeff
            if verbose:
                print("VARIANCE_LOSS - ", loss_coeff)
        
        if "COVARIANCE_LOSS" in self.loss_case:            
            covariance_similarities_loss = self.covariance_similarities(values)
            coeff = self.loss_case["COVARIANCE_LOSS"]
            loss_coeff = covariance_similarities_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["COVARIANCE_LOSS"] = loss_coeff
            if verbose:
                print("COVARIANCE_LOSS - ", loss_coeff)
        
        if "COVARIANCE_LOSS" in self.loss_case:
            covariance_similarities_loss = self.covariance_similarities(values)
            coeff = self.loss_case["COVARIANCE_LOSS"]
            loss_coeff = covariance_similarities_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["COVARIANCE_LOSS"] = loss_coeff
            if verbose:
                print("COVARIANCE_LOSS - ", loss_coeff)

        if "DECORRELATION_LATENT_LOSS" in self.loss_case:
            decorrelation_latent_loss = self.decorrelation_latent(values)
            coeff = self.loss_case["DECORRELATION_LATENT_LOSS"]
            loss_coeff = decorrelation_latent_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["DECORRELATION_LATENT_LOSS"] = loss_coeff
            if verbose:
                print("DECORRELATION_LATENT_LOSS - ", loss_coeff)

        if "JENSEN_SHANNON_DIVERGENCE_LOSS" in self.loss_case:
            jensen_shannon_divergence_loss = self.jensen_shannon_divergence(values)
            coeff = self.loss_case["JENSEN_SHANNON_DIVERGENCE_LOSS"]
            
            loss_coeff = jensen_shannon_divergence_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["JENSEN_SHANNON_DIVERGENCE_LOSS"] = loss_coeff
            if verbose:
                print("JENSEN_SHANNON_DIVERGENCE_LOSS - ", loss_coeff)
        
        if "KENDALL_CORRELATION_LOSS" in self.loss_case:
            kendall_correlation_loss = self.kendall_correlation(values)
            coeff = self.loss_case["KENDALL_CORRELATION_LOSS"]
            
            loss_coeff = kendall_correlation_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["KENDALL_CORRELATION_LOSS"] = loss_coeff
            if verbose:
                print("KENDALL_CORRELATION_LOSS - ", loss_coeff)
        
        if "SPEARMAN_CORRELATION_LOSS" in self.loss_case:
            spearman_correlation_loss = self.spearman_correlation(values)
            coeff = self.loss_case["SPEARMAN_CORRELATION_LOSS"]
            
            loss_coeff = spearman_correlation_loss.mul(coeff)
            loss_total += loss_coeff
            loss_dict["SPEARMAN_CORRELATION_LOSS"] = loss_coeff
            if verbose:
                print("SPEARMAN_CORRELATION_LOSS - ", loss_coeff)

        if verbose:
            print("loss_total - ", loss_total)
        loss_dict["loss_total"] = loss_total
        return loss_dict

    def median_similarities(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        variance_list_in = []
        variance_list_out = []
        #variance_list= [list() for i in range(self.univar_count)]
        
        for id_item, val in enumerate(values):
            variance_list_in.append(val['x_input'])
            variance_list_out.append(val['x_output'])

        variance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        variance_matr_in = torch.reshape(torch.cat(variance_list_in), (len(values),self.univar_count))
        variance_in = torch.median(variance_matr_in, axis=0)[0]
        
        variance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        variance_matr_out = torch.reshape(torch.cat(variance_list_out), (len(values),self.univar_count))
        
        variance_out = torch.median(variance_matr_out, axis=0)[0]
        
        for inp,oup in zip(variance_in, variance_out):
            loss_ret += torch.square(torch.norm(torch.sub(inp,oup,alpha=1), p=2))
        return loss_ret#torch.mul(loss_ret,1)

    def variance_similarities(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        variance_list_in = []
        variance_list_out = []
        #variance_list= [list() for i in range(self.univar_count)]
        
        for id_item, val in enumerate(values):
            variance_list_in.append(val['x_input'])
            variance_list_out.append(val['x_output'])

        variance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        variance_matr_in = torch.reshape(torch.cat(variance_list_in), (len(values),self.univar_count))
        variance_in = torch.var(variance_matr_in, axis=0)

        variance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        variance_matr_out = torch.reshape(torch.cat(variance_list_out), (len(values),self.univar_count))
        
        variance_out = torch.var(variance_matr_out, axis=0)
        #loss_mse = nn.MSELoss()
        for inp,oup in zip(variance_in, variance_out):
            loss_ret += torch.square(torch.norm(torch.sub(inp,oup,alpha=1), p=2))
        return loss_ret#torch.mul(loss_ret,1)

    def covariance_similarities(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        covariance_list_in = []
        covariance_list_out = []
        
        for id_item, val in enumerate(values):
            covariance_list_in.append(val['x_input'])
            covariance_list_out.append(val['x_output'])

        covariance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_in = torch.reshape(torch.cat(covariance_list_in), (len(values),self.univar_count))
        covariance_matr_in = torch.transpose(covariance_matr_in, 0, 1)
        covariance_in = torch.cov(covariance_matr_in, correction=1)
        

        covariance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_out = torch.reshape(torch.cat(covariance_list_out), (len(values),self.univar_count))
        covariance_matr_out = torch.transpose(covariance_matr_out, 0, 1)
        covariance_out = torch.cov(covariance_matr_out, correction=1)

        for inp_row,oup_row in zip(covariance_in, covariance_out):
            for inp_item,oup_item in zip(inp_row,oup_row):
                loss_ret += torch.square(torch.norm(torch.sub(inp_item,oup_item, alpha=1), p=2))
        return loss_ret
    
    def kendall_correlation(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        covariance_list_in = []
        covariance_list_out = []
        
        for id_item, val in enumerate(values):
            covariance_list_in.append(val['x_input'])
            covariance_list_out.append(val['x_output'])
        covariance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_in = torch.reshape(torch.cat(covariance_list_in), (len(values),self.univar_count))
        covariance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_out = torch.reshape(torch.cat(covariance_list_out), (len(values),self.univar_count))
        
        kendall_values = kendall_rank_corrcoef(covariance_matr_in, covariance_matr_out)
        
        for val in kendall_values:
            loss_ret += Tensor([1]).to(device=self.device)-val
        loss_ret /= len(kendall_values)
        return loss_ret


    def spearman_correlation(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        covariance_list_in = []
        covariance_list_out = []
        
        for id_item, val in enumerate(values):
            covariance_list_in.append(val['x_input'])
            covariance_list_out.append(val['x_output'])
        covariance_matr_in = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_in = torch.reshape(torch.cat(covariance_list_in), (len(values),self.univar_count))
        covariance_matr_out = torch.Tensor(len(values), self.univar_count).to(device=self.device)
        covariance_matr_out = torch.reshape(torch.cat(covariance_list_out), (len(values),self.univar_count))
        spearman_obj = SpearmanCorrCoef(self.univar_count)
        
        spearman_values = spearman_obj(covariance_matr_in, covariance_matr_out)
        
        for val in spearman_values:
            loss_ret += val
        loss_ret /= len(spearman_values)
        return loss_ret

    def MSE_similarities(self, values):
        """
        ys_true : vector of items where each item is a groundtruth matrix 
        ys_pred : vector of items where each item is a prediction matrix 
        return the sum of 2nd proximity of 2 matrix
        """
        loss_ret = torch.zeros(1).to(device=self.device)     
        loss_mse = nn.MSELoss()

        for i, val in enumerate(values):
            
            loss_mse_val = loss_mse(val['x_output'], val['x_input'])
            loss_ret += loss_mse_val
        loss_ret /= len(values)
        return loss_ret#torch.mul(loss_ret,1)
    

    def RMSE_similarities(self, values):
        """
        ys_true : vector of items where each item is a groundtruth matrix 
        ys_pred : vector of items where each item is a prediction matrix 
        return the sum of 2nd proximity of 2 matrix
        """
        loss_ret = torch.zeros(1).to(device=self.device)      
        loss_mse = nn.MSELoss()
        for i, val in enumerate(values):            
            loss_mse_val = loss_mse(val['x_output'], val['x_input'])
            loss_ret += loss_mse_val
        loss_ret_sqrt = torch.sqrt(loss_ret)
        return loss_ret_sqrt


    def decorrelation_latent(self, values):
        M = len(values)
        
        loss_ret = torch.zeros(1).to(device=self.device)
        for k in range(self.latent_dim):
            for i in range(k):
                loss_a = torch.zeros(1).to(device=self.device)
                loss_b = torch.zeros(1).to(device=self.device)
                
                for j in range(M):
                    z_j = values[j]['x_latent']
                    z_j_i = z_j[k]
                    z_j_k = z_j[i]
                    loss_a += z_j_i * z_j_k

                loss_a = torch.div(loss_a, M)
                loss_b_0 = torch.zeros(1).to(device=self.device)
                loss_b_1 = torch.zeros(1).to(device=self.device)
                for j in range(M):
                    z_j = values[j]['x_latent']
                    loss_b_0 += z_j[k]
                    loss_b_1 += z_j[i]
                loss_b = loss_b_0 * loss_b_1
                loss_b = torch.div(loss_b, M**2)

                loss_dif = torch.abs(loss_a - loss_b)
                loss_ret += loss_dif
        return loss_ret
    
    def jensen_shannon_divergence(self, values):
        loss_ret = torch.zeros(1).to(device=self.device)
        kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        
        
        JSD_list_in = dict()
        JSD_list_out = dict()

        for vc in range(self.univar_count):
            JSD_list_in[vc] = list()
            JSD_list_out[vc] = list()
        
        for id_item, val in enumerate(values):
            for vc in range(self.univar_count):
                JSD_list_in[vc].append(val['x_input'][vc])
                JSD_list_out[vc].append(val['x_output'][vc])

        for vc in range(self.univar_count):
            
            x_in_i = torch.Tensor( 1,len(values)).to(device=self.device)
            x_in_i = torch.stack(JSD_list_in[vc], dim=0)
            
            x_out_i = torch.Tensor( 1,len(values)).to(device=self.device)
            x_out_i = torch.stack(JSD_list_out[vc], dim=0)
            
            x_in_v = x_in_i.view(-1, x_in_i.size(-1))
            x_out_v = x_out_i.view(-1, x_out_i.size(-1))

            x_in =  F.softmax(x_in_v, dim=1)
            x_out = F.softmax(x_out_v, dim=1)

            m = (0.5 * (x_in + x_out)).log()
            a = kl(x_in.log(), m)
            b = kl(x_out.log(), m)
            jsd_value = 0.5 * (a + b)
            loss_ret += jsd_value
        return loss_ret


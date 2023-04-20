import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from src.NeuroCorrelation.DataBatchGenerator import DataBatchGenerator

class GenTraining():

    def __init__(self, model, epoch, dataset):
        self.model = model()
        self.epoch = epoch
        self.dataset = dataset
        
        self.criterion = nn.MSELoss()
        model_params = self.model.parameters()
        self.optimizer = SGD(params=model_params, lr=0.01, momentum=0.9)

    def training(self, batch_size=128, shuffle_data=True):
        for epoch in range(self.epoch):
            if epoch%25 ==0:
                print("epoch: ", epoch)
            dataLoaded= DataBatchGenerator(dataset=self.dataset, batch_size=batch_size, shuffle=shuffle_data)
            dataBatches = dataLoaded.generate()
            for batch_num, dataBatch in enumerate(dataBatches):
                loss = torch.zeros([1])
                self.optimizer.zero_grad()
                for i, (samplef, noisef) in enumerate(dataBatch):
                    sample = samplef.float()
                    noise = noisef.float()
                    # compute the model output
                    
                    yhat = self.model.forward(x=noise)
                    # calculate loss
                    crit = self.criterion(yhat, sample)
                    loss += crit
                # credit assignment
                loss.backward()
                # update model weights
                self.optimizer.step()

    def dataset_predicted(self):
        predictions, actuals = list(), list()
        for i, (samplef, noisef) in enumerate(dataset_couple):
            sample = samplef.float()
            noise = noisef.float()
            # evaluate the model on the test set
            yhat = gen_model(noise)
            predictions.append(yhat)
                

from torchsummary import summary
from pathlib import Path

class NetworkDetails():
    
    def __init__(self, model, loss, path):
        self.model = model
        self.loss = loss
        self.path = path
        
    def saveModelParams(self):
        ann_str = self.saveModel_structure()
        loss_str = self.saveModel_loss()
        file_str = f"{ann_str}\n\n{loss_str}"
        filename = Path(self.path, "summary_network.txt")
        with open(filename, 'w') as file:
            file.write(file_str)
        print("SETTING PHASE: Summary model file - DONE")
    
    def saveModel_structure(self):
        net_summary = self.model().summary()
        ann_str = f"{net_summary}"
        return ann_str
    
    def saveModel_loss(self):
        loss_str = ""
        for key in self.loss:
            loss_summary = self.loss[key].get_Loss_params()            
            loss_summary_str = f"{key}:: loss: {loss_summary}"
            loss_str = '\n'.join([loss_str, loss_summary_str])
        return loss_str
    
    
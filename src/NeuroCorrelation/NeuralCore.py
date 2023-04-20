from src.NeuroCorrelation.GenTraining import GenTraining
from src.NeuroCorrelation.DataSynteticGeneration import DataSynteticGeneration
from src.NeuroCorrelation.NeuralModels import NeuralModels, GEN_fl


class NeuralCore():

    def __init__(self, device, model_case="fullyRectangle"):
        self.device = device
        #nm = GEN_fl()
        self.model = GEN_fl #mn.get_model()
        self.epoch = 1000
        dataGenerator = DataSynteticGeneration(torch_device=device)
        self.data = dataGenerator.casualGraph(num_of_samples = 10000, size_random=78)
        self.gen_train = GenTraining(self.model, self.epoch, self.data)


    def training_model(self):
        self.gen_train.training()
    
    def overfittingPlots():
        predictions, actuals = list(), list()
        for i, (samplef, noisef) in enumerate(self.data):
            sample = samplef.float()
            noise = noisef.float()
            # evaluate the model on the test set
            yhat = gen_model(noise)
            predictions.append(yhat)

        sampled_list = list()
        for i in range(len(dataset_couple)):
            sampled_list.append(dataset_couple[i][0][j].numpy())
        sampled_list_np = np.array(sampled_list,dtype = float)    

        print("mean:\t",statistics.mean(sampled_list_np),"stdev:\t",statistics.stdev(sampled_list_np))
        plt.hist(sampled_list)
        plt.show()

        generated_list = list()
        for i in range(len(dataset_couple)):
            generated_list.append(predictions[i][j].detach().numpy())
        generated_list_np = np.array(generated_list,dtype = float)    

        print("mean:\t",statistics.mean(generated_list_np),"stdev:\t",statistics.stdev(generated_list_np))
        plt.hist(generated_list)
        plt.show()

        predictions = list()
        generated_list = list()

        for i in range(10000):
            noise = getRandom()
            yhat = gen_model(noise)
            predictions.append(yhat)

        generated_list = list()
        for i in range(10000):
            generated_list.append(predictions[i][j].detach().numpy())
        generated_list_np = np.array(generated_list,dtype = float)    

        print("mean:\t",statistics.mean(generated_list_np),"stdev:\t",statistics.stdev(generated_list_np))
        plt.hist(generated_list)
        plt.show()
        
    
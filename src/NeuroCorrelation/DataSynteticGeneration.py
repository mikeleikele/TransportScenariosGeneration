import numpy as np
import torch
# Visualization libraries
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_dense_adj, to_networkx

class DataSynteticGeneration():

    def __init__(self, torch_device):
        self.torch_device = torch_device

    def casualGraph(self, num_of_samples = 10000, size_random=78):
        self.graph = KarateClub()
        data = self.graph[0]
        g_adj = to_dense_adj(data.edge_index)[0].numpy()#.astype(int)
        g_adj = torch.Tensor([[g_adj.tolist()]]).numpy()#.astype(int)
        self.G = to_networkx(data, to_undirected=True)
        self.pos_nodes = nx.spring_layout(self.G, seed=0)
        
        
        adj_distance = [ [ [ list() ] for i in range(34)] for j in range(34)]
        for (node_a, node_b) in self.G.edges:
            node_a_x = self.pos_nodes[node_a][0]
            node_a_y = self.pos_nodes[node_a][1]

            node_b_x = self.pos_nodes[node_b][0]
            node_b_y = self.pos_nodes[node_b][1]

            edge_len_x = abs(node_a_x - node_b_x)
            edge_len_y = abs(node_a_y - node_b_y)
            
            edge_len = ((edge_len_x**2) + (edge_len_y**2))**(1/2)

            edge_len_dev = edge_len * 0.1
            edge_len_random = np.random.normal(loc=edge_len, scale=edge_len_dev, size = num_of_samples)

            adj_distance[node_a][node_b] = [edge_len_random, edge_len, edge_len_dev]
        self.adj_distance_list = []
        for rows in adj_distance:
            for dist_rand in rows:
                if len(dist_rand[0]) !=0:
                    self.adj_distance_list.append(dist_rand)
        dataset_couple = []
        for i in range(num_of_samples):
            dataset_couple.append((self.getSample(i), self.getRandom(size_random)))
        return dataset_couple


    def getSample(self, key_sample):
        sample = []
        for ed in self.adj_distance_list:    
            sample.append(ed[0][key_sample])  
        return torch.from_numpy(np.array(sample)).float().to(self.torch_device)
    
    def getRandom(self, size):
        randomNoise = torch.randn(size).uniform_(0,1).to(self.torch_device)
        return randomNoise.float()
import numpy as np
import networkx as nx
import os
from torch.utils.data import Dataset, DataLoader
from graph import Graph
from matplotlib import pyplot as plt

class GraphDataset(Dataset):
    def __init__(self, foldername):
        file_names = os.listdir(foldername)
        counter = 0 
        while "{}.graph".format(counter) in file_names:
            counter += 1
        self.len = counter
        self.foldername = foldername

    def __getitem__(self, i):
        return Graph.load('{}/{}.graph'.format(self.foldername, i))

    def __len__(self):
        return self.len

class GraphDatasetEager(Dataset):
    def __init__(self, foldername):
        file_names = os.listdir(foldername)
        counter = 0 
        while "{}.graph".format(counter) in file_names:
            counter += 1
        self.len = counter
        self.foldername = foldername

        self.graphs = [Graph.load('{}/{}.graph'.format(foldername, i)) for i in range(self.len)]

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return self.len

from heuristics import *
from test import *

if __name__=='__main__':
    c, nc = colorize_using_heuristic(graph2, dynamic_ordered_heuristic)
    print(c, nc)

    # ds = GraphDatasetEager('../data/erdos_renyi/train')
    # print(ordered_heuristic(ds[0].adj_list))

    # train_dl = DataLoader(GraphDatasetEager('../data/erdos_renyi/train'))
    # valid_dl = DataLoader(GraphDatasetEager('../data/erdos_renyi/valid'))
    # test_dl = DataLoader(GraphDatasetEager('../data/erdos_renyi/test'))


    # G = nx.erdos_renyi_graph(10, 0.5)
    # G = Graph.from_networkx_graph(G)
    # G.save('test.graph')
    # print(Graph.load('test.graph'))

    # generate_erdos_renyi('erdos_renyi', 1000, 200, 200)
    # generate_watts_strogatz('watts_strogatz', 1000, 200, 200)
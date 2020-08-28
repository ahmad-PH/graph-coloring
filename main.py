import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from graph import Graph
import os
from graph_colorizer import GraphColorizer

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

    def __iter__(self):
        return iter(self.__getitem__(i) for i in range(self.len))

class GraphDatasetEager(GraphDataset):
    def __init__(self, foldername):
        super().__init__(foldername)
        self.graphs = [Graph.load('{}/{}.graph'.format(foldername, i)) for i in range(self.len)]

    def __getitem__(self, i):
        return self.graphs[i]


# ============================== Testing Region ==============================
from matplotlib import pyplot as plt
from heuristics import *
from test import *
from dataset_generators import *
from utility import *
from networkx.algorithms import approximation
from networkx.algorithms.community.kclique import k_clique_communities
from networkx.algorithms.coloring.greedy_coloring import greedy_color

def n_edges(adj_list):
    sum_of_degrees =  sum([len(neighborhood) for neighborhood in adj_list])
    if sum_of_degrees % 2 != 0:
        raise ValueError('sum of degrees in adjacency list is not even.')
    return int(sum_of_degrees / 2)

if __name__=='__main__':
    gc = GraphColorizer()
    g = Graph.from_networkx_graph(nx.erdos_renyi_graph(5, 0.5))
    gc.forward(g)

    import sys
    sys.exit(0)


if __name__=='__main__':
    # avg = run_heuristic_on_dataset(ordered_heuristic, GraphDataset('../data/erdos_renyi_1K/train'))
    # avg = run_heuristic_on_dataset(ordered_heuristic, GraphDataset('../data/watts_strogatz_1K/train'))

    # Generate a Graph:

    # nx_g = nx.watts_strogatz_graph(1000, 4, 0.5)

    nx_g = nx.erdos_renyi_graph(1000, 0.5)
    # g = Graph.from_networkx_graph(nx_g)
    # n_edge = n_edges(g.adj_list)

    # for i, row in enumerate(g.adj_list):
        # print(i, ':', row, len(row))

    # G = Graph.from_networkx_graph(nx_g)
    # a = approximation.max_clique(nx_g)
    color = greedy_color(nx_g)
    # a = list(k_clique_communities(nx_g, 3))
    print(len(set(color.values()))) 
    # _, nc = colorize_using_heuristic(G.adj_list, unordered_heuristic)
    # print(nc)



    # ds = GraphDataset('../data/watts_strogatz_1K/train')
    # nx.draw(ds[0].get_nx_graph())
    # plt.show()

    # train_dl = DataLoader(GraphDatasetEager('../data/erdos_renyi_10/train'))
    # valid_dl = DataLoader(GraphDatasetEager('../data/erdos_renyi_10/valid'))
    # test_dl = DataLoader(GraphDatasetEager('../data/erdos_renyi_10/test'))

    # generate_erdos_renyi('erdos_renyi_', 400, 20, 20, 1000) 
    # generate_watts_strogatz('watts_strogatz', 400, 20, 20, 1000)
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader

a = [
    [1, 3],
    [0, 2, 3, 4],
    [1, 4],
    [0, 1, 4],
    [1, 2, 3],
]

def adj_list_to_matrix(adj_list):
    n = len(adj_list)
    adj_matrix = np.zeros((n,n))
    for i in range(n):
        adj_matrix[i, adj_list[i]] = 1
    return adj_matrix

def adj_matrix_to_list(adj_matrix):
    adj_list = []
    for i in range(adj_matrix.shape[0]):
        adj_list.append([])
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1: 
                adj_list[-1].append(j)
    return adj_list

    
def vertex_order_heuristic(adj_list):
    n_vertices = len(adj_list)

    degrees = [len(row) for row in adj_list]
    first_vertex = np.argmax(degrees)
    color_degrees = np.zeros(n_vertices)
    color_degrees[first_vertex] = float('+inf')

    result = []
    while len(result) < n_vertices:
        next_vertex = np.argmax(color_degrees)
        result.append(next_vertex)
        color_degrees[adj_list[next_vertex]] += 1
        color_degrees[next_vertex] = float('-inf')

    return result

def generate_erdos_renyi(foldername, n_train, n_valid, n_test):
    os.mkdir(foldername)
    for subfoldername, graph_count in [('train', n_train), ('valid', n_valid), ('test', n_test)]:
        os.mkdir(os.path.join(foldername, subfoldername))
        for i in range(graph_count):
            G = nx.erdos_renyi_graph(10, 0.5)
            nx.write_adjlist(G, "{}/{}/{}.adjlist".format(foldername, subfoldername, i))

def generate_watts_strogatz(foldername, n_train, n_valid, n_test):
    os.mkdir(foldername)
    for subfoldername, graph_count in [('train', n_train), ('valid', n_valid), ('test', n_test)]:
        os.mkdir(os.path.join(foldername, subfoldername))
        for i in range(graph_count):
            G = nx.watts_strogatz_graph(10, 4, 0.5)
            nx.write_adjlist(G, "{}/{}/{}.adjlist".format(foldername, subfoldername, i))

class GraphDataset(Dataset):
    def __init__(self, foldername):
        file_names = os.listdir(foldername)
        counter = 0 
        while "{}.adjlist".format(counter) in file_names:
            counter += 1
        self.len = counter
        self.foldername = foldername

    def __getitem__(self, i):
        return nx.read_adjlist('{}/{}.adjlist'.format(self.foldername, i), nodetype=int)

    def __len__(self):
        return self.len

class GraphDatasetEager(Dataset):
    def __init__(self, foldername):
        file_names = os.listdir(foldername)
        counter = 0 
        while "{}.adjlist".format(counter) in file_names:
            counter += 1
        self.len = counter
        self.foldername = foldername

        self.graphs = [nx.read_adjlist('{}/{}.adjlist'.format(foldername, i), nodetype=int) for i in range(self.len)]
        print(self.len)

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return self.len


if __name__=='__main__':
    generate_erdos_renyi('erdos_renyi', 1000, 200, 200)
    generate_watts_strogatz('watts_strogatz', 1000, 200, 200)


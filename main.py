# imports from standard python libraries
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import *
from matplotlib import pyplot as plt

# imports from local files
from graph import Graph
from heuristics import *
from test import *
from dataset_generators import *
from utility import *
from networkx.algorithms import approximation
from networkx.algorithms.community.kclique import k_clique_communities
from networkx.algorithms.coloring.greedy_coloring import greedy_color
from graph_dataset import GraphDataset, GraphDatasetEager
from test import graph1, graph2
from dataset_generators import generate_erdos_renyi, generate_watts_strogatz
from graph_colorizer import GraphColorizer
from utility import kneser_graph


# for k in range(3,10):
#     for n in range(2*k+1, 2*k+7):
#         graph = kneser_graph(n, k)
#         print('\n')
#         print('n: {}, k: {}'.format(n, k))
#         # print(graph)
#         color = greedy_color(graph.get_nx_graph())
#         n_colors = len(set(color.values()))
#         print('chromatic number: {}, greedy approx: {}'.format(n-2*k+2, n_colors))
# import sys; sys.exit(0)


# generate_erdos_renyi('../data/erdos_renyi_500', 20, 5, 5, 500, 0.5)



if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    colorizer = GraphColorizer(loss_type="reinforce", device=device)
    optimizer = torch.optim.Adam(colorizer.parameters(), lr=0.01)
    # graph = Graph(graph4)

    import random
    random.seed(0)
    np.random.seed(0)
    nx_g = nx.erdos_renyi_graph(200, 0.3)
    # nx_g = nx.erdos_renyi_graph(1000, 0.5)
    graph = Graph.from_networkx_graph(nx_g)

    color = greedy_color(graph.get_nx_graph())
    print('approx:', len(set(color.values()))) 
    # import sys; sys.exit(0)

    baseline = 0.
    decay = 0.95
    losses = []
    for i in range(100):
        print("epoch: {}".format(i))
        optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):
        coloring, loss = colorizer.forward(graph, baseline)
        n_used_colors = len(set(coloring))
        baseline = decay * baseline + (1-decay) * n_used_colors
        loss.backward()
        optimizer.step()
        print('loss:', loss.item())
        print('n_used: {}, new_baseline: {}'.format(n_used_colors, baseline))
        losses.append(loss.item())
        print(coloring)


    plt.plot(losses)
    plt.show()

    import sys 
    sys.exit()


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
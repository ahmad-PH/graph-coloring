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
from dataset_generators import generate_erdos_renyi, generate_watts_strogatz
from graph_utility import generate_kneser_graph, generate_queens_graph

from colorizer_8_pointer_netw import GraphColorizer

mode = "single_run"
# mode = "dataset_run"

def test_on_single_graph():
    # graph = Graph(bipartite_10_vertices)

    # graph = generate_queens_graph(5,5)
    # graph = generate_queens_graph(6,6)
    # graph = generate_queens_graph(7,7)
    graph = generate_queens_graph(8,8)
    # graph = generate_queens_graph(8,12)
    # graph = generate_queens_graph(11,11)
    # graph = generate_queens_graph(13,13)

    # graph = GraphDatasetEager('../data/erdos_renyi_100/train')[0]

    # graph = Graph.from_networkx_graph(nx.erdos_renyi_graph(100, 0.5))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # colorizer = GraphColorizer(loss_type="reinforce", device=device)
    colorizer = GraphColorizer(loss_type="reinforce", graph=graph, device=device)
    optimizer = torch.optim.Adam(colorizer.parameters(), lr=0.0001)

    color = greedy_color(graph.get_nx_graph(), strategy="DSATUR")

    baseline = EWMAWithCorrection(0.95)
    losses = []
    baselines = []
    best_answer = float('+inf')
    best_answer_epoch = -1
    n_used_list = []

    for i in range(200):
        print("\n\nepoch: {}".format(i))
        optimizer.zero_grad()
        coloring, loss = colorizer.forward(graph, baseline.get_value())
        n_used_colors = len(set(coloring))
        n_used_list.append(n_used_colors)
        baseline.update(n_used_colors)
        loss.backward()

        # from multi_head_attention import MultiHeadAttention

        # def print_module_info(name, module):
        #     if isinstance(module, nn.ModuleList):
        #         for child in module:
        #             print_module_info(name, child)
        #         return 

        #     if isinstance(module, MultiHeadAttention) and module.pointer_mode == False:
        #         # plt.title('{} {}: data'.format(name, module))
        #         # plt.hist(module.out.flatten().detach().cpu().numpy())
        #         # plt.figure()

        #         # plt.title('{} {}: grad'.format(name, module))
        #         # plt.hist(module.out.grad.flatten().detach().cpu().numpy())
        #         # plt.figure()
        #         print('{}: {}'.format(name, module))
        #         # print('data:', module.out.shape)
        #         print('grad: ', module.out.grad.mean())

        #     if name == "attn_combiner":
        #         print('{}: {}'.format(name, module))
        #         print(module[1])

        # for name, module in colorizer.named_children():
        #     print_module_info(name, module)
        # plt.show()
 

        # print('embedding grads: ', torch.mean(colorizer.embeddings.grad), torch.std(colorizer.embeddings.grad))
        # for j in range(colorizer.n_attn_layers):
        #     print('attn grad:', torch.mean(colorizer.neighb_attns[j].W.grad), torch.std(colorizer.neighb_attns[j].W.grad))
        # print('classifier grad:', torch.mean(colorizer.color_classifier.fc1.weight.grad), torch.std(colorizer.color_classifier.fc1.weight.grad))

        optimizer.step()

        print('loss:', loss.item())
        print('n_used: {}, new_corrected_baseline: {}'.format(n_used_colors, baseline.get_value()))
        losses.append(loss.item())
        baselines.append(baseline.get_value())
        if n_used_colors < best_answer:
            best_answer = n_used_colors
            best_answer_epoch = i

    print('best answer is {} and first happened at epoch {}'.format(best_answer, best_answer_epoch))
    print('approx answer:', len(set(color.values()))) 
    
    from globals import data
    print('appending:', best_answer)
    data.results.append(best_answer)
    # print(best_answer, file=data.file_writer)
    # file_writer.flush()

    # plt.title('losses')
    # plt.plot(losses)
    # plt.figure()
    # plt.title('n_used')
    # plt.plot(n_used_list)
    # plt.plot(baselines)
    # plt.show()


def test_on_dataset():
    # torch.manual_seed(1) # produces nan

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ds = GraphDatasetEager('../data/erdos_renyi_100/train')
    colorizer = GraphColorizer(loss_type="reinforce", device=device)
    optimizer = torch.optim.Adam(colorizer.parameters(), lr=0.0001) #0.0005)

    approx_answers = []
    for i in range(len(ds)):
        color = greedy_color(ds[i].get_nx_graph(), strategy="DSATUR")
        approx_answers.append(len(set(color.values())))

    baselines = [EWMAWithCorrection(0.95) for _ in range(len(ds))]
    losses : List[List[float]] = [[] for _ in range(len(ds))]
    n_used_list : List[List[int]] = [[] for _ in range(len(ds))]
    final_answers = []
    n_epochs = 20

    for epoch in range(n_epochs):
        print("\nepoch: {}".format(epoch))
        for i in range(len(ds)):
            print('\ngraph number: {}'.format(i))
            optimizer.zero_grad()
            coloring, loss = colorizer.forward(ds[i], baselines[i].get_value())
            n_used_colors = len(set(coloring))
            n_used_list[i].append(n_used_colors)
            baselines[i].update(n_used_colors)
            loss.backward()
            optimizer.step()
            print('loss:', loss.item())
            print('n_used: {}, new_baseline: {}'.format(n_used_colors, baselines[i].get_value()))
            print('approx answer: {}'.format(approx_answers[i]))
            losses[i].append(loss.item())
            if epoch == n_epochs - 1:
                final_answers.append(n_used_colors)

    # plt.title('w norms:')
    # plt.plot(colorizer.attn_w_norms)
    # plt.figure()

    # plt.title('a norms:')
    # plt.plot(colorizer.attn_a_norms)
    # plt.figure()

    # plt.title('classifier norm 1:')
    # plt.plot(colorizer.classif_norms[0])
    # plt.figure()

    # plt.title('classifier norm 2:')
    # plt.plot(colorizer.classif_norms[1])
    # plt.figure()

    # plt.title('classifier norm 3:')
    # plt.plot(colorizer.classif_norms[2])

    # plt.show()

    approx_answers = np.array(approx_answers)
    final_answers = np.array(final_answers)
    difference = final_answers - approx_answers
    print('final, then greedy answers:')
    print(final_answers)
    print(approx_answers)
    score = np.average(difference)
    n_better = np.sum(difference < 0.)
    n_worse = np.sum(difference > 0.)
    n_equal = np.sum(difference == 0.)
    print('score: {}, better: {}, worse: {}, equal: {}'.format(score, n_better, n_worse, n_equal))


    # shows a plot of each graphs loss and n_used
    for i in range(len(ds)):
        plt.title('{}th graph losses'.format(i))
        plt.plot(losses[i])
        plt.figure()
        plt.title('{}th graph n_used'.format(i))
        plt.plot(n_used_list[i])
        plt.show()



if __name__=="__main__":
    from globals import initialize_globals, free_globals, data
    initialize_globals()
    data.results = []
    try:
        if mode == "dataset_run":
            test_on_dataset()
            print('results:', data.results)
            print('mean, std:', np.mean(data.results), np.std(data.results)) 
        elif mode == "single_run":
            test_on_single_graph()
            print('results:', data.results)
    finally:
        free_globals()
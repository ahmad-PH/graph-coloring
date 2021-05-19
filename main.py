import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import *
from matplotlib import pyplot as plt
import sys

from graph import Graph
from heuristics import *
from test import *
from dataset_generators import *
from utility import *
from networkx.algorithms import approximation
from networkx.algorithms.community.kclique import k_clique_communities
from networkx.algorithms.coloring.greedy_coloring import greedy_color
from graph_dataset import GraphDataset, GraphDatasetEager, generate_embeddings_for_dataset, generate_optimal_solutions_for_dataset
from dataset_generators import generate_erdos_renyi, generate_watts_strogatz
from graph_utility import generate_kneser_graph, generate_queens_graph
from globals import data
from manual_emb_utility import plot_points
from manual_emb_main import learn_embeddings

from colorizer_4_extra_layers import GraphColorizer

mode = "single_run"
# mode = "dataset_run"

def test_on_single_graph():

    # graph = generate_queens_graph(5,5)
    graph = generate_queens_graph(6,6)
    # graph = generate_queens_graph(7,7)
    # graph = generate_queens_graph(8,8)
    # graph = generate_queens_graph(8,12)
    # graph = generate_queens_graph(11,11)
    # graph = generate_queens_graph(13,13)
    # graph = GraphDatasetEager('../data/erdos_renyi_100/train')[0]

    # ================== Generate Optimal Embeddings: ==================
    # from exact_coloring import find_k_coloring
    # from test_data_generator import generate_clustered_embeddings
    # optimal_coloring = find_k_coloring(graph, 7)
    # save_to_file(optimal_coloring, 'optimalcoloring.txt')

    # optimal_clustering = generate_clustered_embeddings(10, graph.n_vertices, 7, optimal_coloring, 2.)
    # torch.save(optimal_clustering, 'optimalclustering_q6_6.pt')

    # print('solution:', [(i, v) for i, v in enumerate(optimal_coloring)])
    # plot_points(optimal_clustering, annotate=True)
    # plt.title('embeddings')
    # plt.show()
    # sys.exit(0)

    # ================== Learn Normal Embeddings: ==================
    # embeddings, _ = learn_embeddings(graph, 10, 7)
    # torch.save(embeddings, 'normalclustering_q6_6.pt')

    # ================== Load embeddings and labels for q6_6: ==================
    embeddings = torch.load('optimal_embeddings_q6_6.pt', map_location=fastest_available_device()) # optimal embeddings
    # embeddings = torch.load('normal_embeddings_q6_6.pt', map_location=fastest_available_device()) # normal embeddings
    data.optimal_coloring = load_from_file('optimalcoloring.txt')

    # ================== Load  embeddings and labels for a graph from er_50_challenging: ==================
    # graph_index = 2
    # ds_path = '../data/er_50_challenging'
    # ds = GraphDataset(ds_path)
    # graph = ds[graph_index]
    # embeddings = torch.load('{}/embeddings/{}.pt'.format(ds_path, graph_index), map_location=fastest_available_device())
    # data.optimal_coloring = load_from_file('{}/solutions/{}.txt'.format(ds_path, graph_index))
    # print('chromatic_number:', len(set(data.optimal_coloring)))

    colorizer = GraphColorizer(embeddings.shape[1], device=fastest_available_device())
    optimizer = torch.optim.Adam(colorizer.parameters(), lr=0.001)

    n_epochs = 400
    baseline = EWMAWithCorrection(0.9)
    losses = []
    baselines = []
    n_color_performance = []
    # corrected_performance = []
    costs = []
    violation_ratios = []
    data.n_used_lambda_scheduler = LinearScheduler(0.001, 0.1, n_epochs)

    for i in range(n_epochs):
        print("\n\nepoch: {}".format(i))

        optimizer.zero_grad()
        coloring, loss, extra = colorizer.forward(graph, embeddings, baseline.get_value())
        n_used_colors = len(set(coloring))
        baseline.update(extra.cost)
        print('props', coloring_properties(coloring, graph))
        print('n_used:', n_used_colors)
        print('cost: {}, new_corrected_baseline: {}'.format(extra.cost, baseline.get_value()))

        # Recordings for plotting purposes
        n_color_performance.append(n_used_colors)
        costs.append(extra.cost)
        violation_ratios.append(extra.violation_ratio)
        # corrected_performance.append(len(set(correct_coloring(coloring, graph))))

        loss.backward()
        optimizer.step()

        # print('loss:', loss.item())
        losses.append(loss.item())
        baselines.append(baseline.get_value())

    # greedy_coloring = greedy_color(graph.get_nx_graph(), strategy="DSATUR")
    # print('greedy answer:', len(set(greedy_coloring.values()))) 
    
    best_answer = np.min(n_color_performance)
    data.results.append(best_answer)

    plt.title('losses')
    plt.plot(losses)
    plt.figure()

    # plt.title('corrected_performance')
    # plt.plot(corrected_performance)
    # plt.figure()

    plt.title('cost + baseline')
    plt.plot(costs, label='cost')
    plt.plot(baselines, label='baseline')
    plt.legend()
    plt.figure()

    plt.title('n_used')
    plt.plot(n_color_performance)
    plt.figure()

    plt.title('violation_ratios')
    plt.plot(violation_ratios)

    plt.show()


def test_on_dataset():
    # torch.manual_seed(1) # produces nan

    ds = GraphDatasetEager('../data/erdos_renyi_100/train')
    colorizer = GraphColorizer(loss_type="reinforce", device=fastest_available_device())
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
from graph import Graph
from utility import *
from test import *
from graph_utility import is_proper_coloring
from globals import data
from snap_utility import load_snap_graph
from networkx.algorithms.coloring.greedy_coloring import greedy_color
from networkx.algorithms.clique import find_cliques, graph_clique_number
from exact_coloring import find_k_coloring
from manual_emb_utility import *

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import sys

from manual_emb_main import learn_embeddings

# mode = "single_run"
mode = "dataset_run"

# G = snap.LoadEdgeList(snap.TNGraph, "../data/singular/new/p2p-Gnutella04.txt")
# G = snap.LoadEdgeList(snap.TNGraph, "../data/singular/new/ca-HepTh.txt")
# GC = convert(G)

# g = load_snap_graph("../data/singular/new/p2p-Gnutella04.txt")
# g = load_snap_graph("../data/singular/new/ca-HepTh.txt")
# g = load_snap_graph("../data/singular/new/email-Eu-core.txt")
# g = load_snap_graph('../data/singular/new/CollegeMsg.txt')

# g.save('../data/singular/new/CollegeMsg.graph')

# import sys; sys.exit(0)

if mode == "single_run":

    embedding_dim = 10

    # graph = Graph.load('../data/singular/new/ca-HepTh.graph')
    # graph = Graph.from_mtx_file('../data/singular/new/rgg_n_2_17_s0.mtx')
    # graph = Graph.from_mtx_file('../data/singular/new/kron_g500-logn16.mtx')
    # graph, n_clusters = generate_queens_graph(5,5), 5
    # graph, n_clusters = generate_queens_graph(7,7), 7
    # graph, n_clusters = generate_queens_graph(13,13), 13
    graph, n_clusters = Graph.load('../data/singular/new/email-Eu-core.graph'), 21
    # graph, n_clusters = Graph.load('../data/singular/new/CollegeMsg.graph'), 10

    # graph = Graph.load('../data/singular/ws_1000')
    # graph = Graph.load('../data/singular/k_1000')

    # coloring = find_k_coloring(graph, 7)
    # if coloring is not None:
    #     for i, c in enumerate(coloring):
    #         print('{}: {}'.format(i, c))
    #     print(len(set(coloring)))
    #     print(is_proper_coloring(coloring, graph))

    # import sys; sys.exit(0)

    # clique_num = graph_clique_number(graph.get_nx_graph())
    # print('clique: ', clique_num)
    # print('greedy coloring num: ', greedy_coloring_number(graph, 'DSATUR'))
    # import sys; sys.exit(0)

    # cliques = find_cliques(graph.get_nx_graph())

    # for clique in cliques:
    #     if len(clique) == 6:
    #         print('maximum clique:')
    #         print(clique)
    #         print('\n')

    # sys.exit(0)

    # graph = generate_queens_graph(20, 20)

    # # knesser graph:
    # n, k = 10, 3
    # graph = generate_kneser_graph(n, k)
    # print(graph.n_vertices, graph.n_edges, kneser_graph_chromatic_number(n, k))

    seed = np.random.randint(0, 1000000)
    # seed = 547519 # case study on q7_7
    # seed = 348266 # peterson 1:
    # seed = 266412
    # seed = 72511 # error on peterson with dim=2
    # seed = 66009 # error on peterson with dim=10
    # seed = 682580 # error on peterson with dim = 10
    # seed = 62843 # error on q7_7 with dim=10
    # seed = 11698
    # seed = 232231
    # seed = 874581 # n_color = 16 for q13_13 with jaccard multiplier of 5
    # seed = 619964
    # seed = 901365 # gives correct answer for q6_6

    torch.random.manual_seed(seed)
    np.random.seed(seed)
    print('seed is: ', seed)

    embeddings, results = learn_embeddings(graph, n_clusters, embedding_dim, verbose=True)

    print('results:')
    print(results)
    # print('adj list:')
    # for i in range(len(graph.adj_list)):
    #     print('{}: {}'.format(i, graph.adj_list[i]))

    # print('embeddings')
    # for i in range(embeddings.shape[0]):
    #     print('{}: {}'.format(i, embeddings[i].data))
    # print(embeddings.shape)


    # torch.save(embeddings, 'q6_6.pt')

    # plot the losses:
    plt.plot(data.losses['scaled_neighborhood_loss'], label='scaled_neighborhood')
    plt.plot(data.losses['scaled_compactness_loss'], label='scaled_compactness')
    plt.plot(data.losses['overall'], label='overall')
    plt.legend()
    plt.title('losses')
    plt.savefig('/home/ahmad/Desktop/losses.png')
    plt.figure()

    plt.plot(data.losses['neighborhood_loss'], label='neighborhood')
    plt.plot(data.losses['compactness_loss'], label='compactness')
    plt.legend()
    plt.title('raw losses')
    plt.savefig('/home/ahmad/Desktop/raw_losses.png')
    plt.figure()

    plt.plot([i*10 for i in range(len(data.n_color_performance))], data.n_color_performance)
    plt.title('n_color_performance')
    plt.savefig('/home/ahmad/Desktop/color_performance.png')
    plt.show()

    # plt.plot(data.neighborhood_losses_p2, label='neighborhood_p2')
    # plt.plot(data.compactness_losses_p2, label='compactness_p2')
    # plt.plot(data.losses_p2, label='overall_p2')
    # plt.legend()
    # plt.show()
    # plt.figure()

elif mode == "dataset_run":
    # if os.path.exists('results.txt'):
    #     print('results.txt exists.')
    #     sys.exit(0)

    embedding_dim = 10

    # dataset = [
    #     (graph1, 3),
    #     (graph2, 2),
    #     (graph3, 3),
    #     (bipartite_10_vertices, 2),
    #     (slf_hard, 3),
    #     (petersen_graph, 3),
    #     (generate_queens_graph(5,5), 5),
    #     (generate_queens_graph(6,6), 7),
    #     (generate_queens_graph(7,7), 7),
    #     (generate_queens_graph(8,8), 9),
    #     (generate_queens_graph(8,12), 12),
    #     (generate_queens_graph(13, 13), 13),
    #     (Graph.load('../data/singular/ws_10').set_name('ws_10'), 4), # chi
    #     (Graph.load('../data/singular/ws_100').set_name('ws_100'), 4), # DSATUR
    #     (Graph.load('../data/singular/ws_1000').set_name('ws_1000'), 4), # DSATUR
    #     # (Graph.load('../data/singular/ws_10000').set_name('ws_10000'), -1),
    #     (Graph.load('../data/singular/k_10').set_name('k_10'), 3), # chi
    #     (Graph.load('../data/singular/k_100').set_name('k_100'), 6), # chi
    #     (Graph.load('../data/singular/k_1000').set_name('k_1000'), 5), # chi
    #     # (Graph.load('../data/singular/k_10000').set_name('k_10000'), 4),
    #     (Graph.load('../data/singular/er_10').set_name('er_10'), 5), # DSATUR
    #     (Graph.load('../data/singular/er_100').set_name('er_100'), 18), # DSATUR
    #     # (Graph.load('../data/singular/er_1000').set_name('er_1000'), 114), # DSATUR 
    #     # (Graph.load('../data/singular/er_10000').set_name('er_10000'), -1),
    # ]

    # dataset = [
    #     # (generate_queens_graph(7, 7), 7)
    #     (generate_queens_graph(8,12), 12),
    # ]

    dataset = [
        (Graph.load('../data/singular/new/email-Eu-core.graph').set_name('email-Eu-core'), 21), # DSATUR
        (Graph.load('../data/singular/new/CollegeMsg.graph').set_name('CollegeMsg'), 10), # DSATUR
        # (Graph.load('../data/singular/new/ca-HepTh.graph').set_name('ca-HepTh'), 32), # DSATUR
    ]

    n_runs_per_graph = 10

    with open('results.txt', 'w') as out:
        summary = []
        for graph, n_clusters in dataset:
            seed = np.random.randint(0, 1000000)
            torch.random.manual_seed(seed)
            np.random.seed(seed)

            print('results for graph: ', graph.name, file=out)
            print('(seed is: {})'.format(seed), file=out)
            results_list = []
            for i in range(n_runs_per_graph):
                _ , results = learn_embeddings(graph, n_clusters, embedding_dim, verbose=False)
                results_list.append(results)
            
            violation_ratio_list = np.array([result.violation_ratio for result in results_list])
            n_used_colors_list = np.array([result.n_used_colors for result in results_list])

            print('violation_ratio: {}'.format(np.mean(violation_ratio_list)), file=out)
            min_used_colors = np.min(n_used_colors_list)
            ratio_of_good_runs = np.mean(n_used_colors_list == min_used_colors)
            print('min used colors: {}, ratio of good runs: {}'.format(min_used_colors, ratio_of_good_runs), file=out)
            mean_used_colors = np.mean(n_used_colors_list)
            stddev_used_colors = np.std(n_used_colors_list)
            print('n_used stats: {}, {}'.format( mean_used_colors, stddev_used_colors), file=out)
            print('n_used: {}'.format(n_used_colors_list), file=out)
            summary.append([min_used_colors, ratio_of_good_runs, mean_used_colors, stddev_used_colors, np.mean(violation_ratio_list)])
            print('\n\n', file=out)
            out.flush()

            print('success')

        print('summary:', file=out)
        for item in summary:
            print('{:.1f}, {:.1f},  {:.1f}, {:.2f}, {:.1f}%'.format(item[0], item[1], item[2], item[3], 100 * item[4]), file=out)
        
        

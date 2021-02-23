from graph import Graph
from utility import *
from graph_utility import *
from typing import List
from sklearn.cluster import KMeans
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
import os
import sys

neighborhood_losses_p1 : List[float] = []
compactness_losses_p1 : List[float] = []
losses_p1 : List[float] = []

neighborhood_losses_p2 : List[float] = []
compactness_losses_p2 : List[float] = []
losses_p2 : List[float] = []


def learn_embeddings(graph, n_clusters, embedding_dim, verbose):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = DataDump()

    embeddings = initialize_embeddings(graph.n_vertices, embedding_dim, mode="normal", device=device)

    with torch.no_grad():
        adj_matrix = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32).to(device) 
        # adj_matrix = torch.tensor(graph.get_adj_matrix()).to(device) # bool by default 

        # print('type:', adj_matrix.dtype)

        inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(device)
        # inverted_adj_matrix = adj_matrix.logical_not().logical_and(torch.eye(graph.n_vertices, dtype=torch.bool).logical_not())

        # print('mem3:', torch.cuda.memory_allocated())
        # print('type:', inverted_adj_matrix.dtype)

        overlap_matrix = torch.zeros(graph.n_vertices, graph.n_vertices).to(device)
        for i in range(graph.n_vertices):
            for j in range(graph.n_vertices):
                if i == j: continue
                n_i = set(graph.adj_list[i])
                n_j = set(graph.adj_list[j])
                overlap_matrix[i][j] = len(n_i.intersection(n_j)) / (len(n_i.union(n_j)) + 1e-8)

            overlap_matrix[i][graph.adj_list[i]] = 0. # suppress entries of neighbors

        global_overlap_matrix = torch.zeros_like(overlap_matrix)
        n_terms = 1
        beta = 0.9
        for i in range(1, n_terms + 1):
            global_overlap_matrix += (beta ** (i-1)) * torch.matrix_power(overlap_matrix, i)
        
        for i in range(graph.n_vertices):
            global_overlap_matrix[i][i] = 0
            global_overlap_matrix[i][graph.adj_list[i]] = 0 # suppress entries of neighbors

        lambda_3 = 5.
        similarity_matrix = inverted_adj_matrix + lambda_3 * global_overlap_matrix
        
    optimizer = torch.optim.Adam([embeddings], lr=0.1)

    n_phase1_iterations = 200
    lambda_1_scheduler = LinearScheduler(0.99, 0.5, n_phase1_iterations) # used to be 0.01 -> 0.02 
    
    # phase 1
    for i in range(n_phase1_iterations):
        if i % 10 == 0 and verbose:
            plt.title('{}'.format(i))
            colors = ['b'] * graph.n_vertices
            colors[5] = 'g'
            colors[40] = 'r'
            plot_points(embeddings, annotate=True, c=colors)
            # plot_points(embeddings, annotate=True, c=classes_to_colors(proper_coloring))
            plt.savefig('images/{}'.format(i))
            plt.clf()

        # if i % 20 == 0 and i != 0: 
        #     reinitialize_embeddings(embeddings, loss_function =
        #         lambda emb: compute_neighborhood_losses(emb, adj_matrix), ratio=0.1)

        optimizer.zero_grad()
        
        distances = compute_pairwise_distances(embeddings, embeddings)

        # with torch.no_grad():
        #     # similarity_matrix = (2 * -adj_matrix + 1) - torch.eye(graph.n_vertices).to(self.device)
        #     similarity_matrix = -adj_matrix
        #     similarity_matrix = similarity_matrix.float()

        # neighborhood_loss = - torch.sum(all_distances * adj_matrix.float() / torch.sum(adj_matrix, dim=1, keepdim=True))
        neighborhood_loss = compute_neighborhood_losses(embeddings, adj_matrix, precomputed_distances=distances).sum()

        # original:
        # center = torch.mean(embeddings, dim=0)
        # center_repeated = center.unsqueeze(0).expand(N, -1)
        # compactness_loss = torch.sum(torch.norm(embeddings - center_repeated, dim=1) ** 2)

        # compactness_loss = torch.sum(distances * inverted_adj_matrix.float() / torch.sum(inverted_adj_matrix, dim=1, keepdim=True))
        compactness_loss = torch.sum(distances * similarity_matrix.float() / (torch.sum(similarity_matrix, dim=1, keepdim=True) + 1e-10))

        lambda_1 = lambda_1_scheduler.get_next_value()
        loss = lambda_1 * neighborhood_loss + (1 - lambda_1) * compactness_loss # lambda_1 used to be for compactness
        neighborhood_losses_p1.append((1 - lambda_1) * neighborhood_loss)
        compactness_losses_p1.append(lambda_1 * compactness_loss)
        losses_p1.append(loss)

        loss.backward()
        optimizer.step()

        # if i % 10 == 0:
            # plt.figure()
            # plot_points(embeddings, title='epoch {}'.format(i), annotate=True)

    # phase 2
    kmeans = KMeans(n_clusters)
    kmeans.fit(embeddings.detach().cpu().numpy())
    cluster_centers = torch.tensor(kmeans.cluster_centers_).to(device)

    if verbose:
        print('end of phase 1:')
        plt.figure()
        plot_points(embeddings, annotate=True)
        plot_points(cluster_centers, c='orange')
        plt.title('end of phase 1')

    # if verbose:
    #     clique_number = graph_clique_number(graph.get_nx_graph())
    #     for clique in find_cliques(graph.get_nx_graph()):
    #         if len(clique) == clique_number:
    #             print('maximum clique:')
    #             print(clique)
    #             print('\n')
    #             c = ['b'] * graph.n_vertices
    #             s = [10] * graph.n_vertices
    #             for i in clique:
    #                 c[i] = 'r'
    #                 s[i] = 40
    #             plt.figure()
    #             plot_points(embeddings, annotate=True, c=c, s=s)
    #             plot_points(cluster_centers, annotate=True, c='orange')
    #             plt.title('clique')


    # for i in range(100):
    #     optimizer.zero_grad()

    #     # if i % 20 == 0 and i != 0: 
    #     #     embeddings = reinitialize_embeddings(embeddings,
    #     #         loss_function=lambda emb: compute_neighborhood_losses(emb, adj_matrix))

    #     neighborhood_loss = compute_neighborhood_losses(embeddings, adj_matrix).sum()
        
    #     distances_from_centers = compute_pairwise_distances(embeddings, cluster_centers)
    #     compactness_loss = torch.sum(torch.min(distances_from_centers, dim=1)[0] ** 2)

    #     # _lambda = 0.1
    #     lambda_2 = 0.1
    #     loss = (1 - lambda_2) * neighborhood_loss + lambda_2 * compactness_loss
    #     neighborhood_losses_p2.append((1 - lambda_2) * neighborhood_loss)
    #     compactness_losses_p2.append(lambda_2 * compactness_loss)
    #     losses_p2.append(loss)

    #     loss.backward()
    #     optimizer.step()

    # if verbose:
    #     print('end of phase 2:')
 
    colors = torch.argmin(compute_pairwise_distances(embeddings, cluster_centers), dim=1)
    colors = colors.detach().cpu().numpy()
    properties = coloring_properties(colors, graph)
    results.violation_ratio = properties[2]

    if verbose:
        print('colors:')
        print(colors)
        print('properties:')
        print(properties)

        # plt.figure()
        # plot_points(embeddings, annotate=True)
        # plot_points(cluster_centers, c='orange', annotate=True)
        # plt.title('end of phase 2')

    if verbose:
        violators = set([])
        for v1, row in enumerate(graph.adj_list):
            for v2 in row:
                if colors[v1] == colors[v2]:
                    violators.add(v1)
                    print('violation: ({}, {})'.format(v1, v2))

        # for v in sorted(list(violators)):
        #     plt.figure()
        #     c = highlight_neighborhood(v, graph)
        #     plot_points(embeddings, annotate=True, c=c)
        #     plot_points(cluster_centers, c='orange', annotate=True)

    correct_coloring(colors, graph)

    if verbose:
        print('corrected_colors:')
        print(colors)
        print(is_proper_coloring(colors, graph))
        print('n_used_colors:', len(set(colors)))

        plt.show()

    results.n_used_colors = len(set(colors))
    return embeddings, results


mode = "single_run"
# mode = "dataset_run"

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
    n_clusters = 7

    # graph = Graph.load('../data/singular/new/ca-HepTh.graph')
    # graph = Graph.from_mtx_file('../data/singular/new/rgg_n_2_17_s0.mtx')
    # graph = Graph.from_mtx_file('../data/singular/new/kron_g500-logn16.mtx')
    graph = generate_queens_graph(7,7)
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

    # seed = np.random.randint(0, 1000000)
    seed = 547519 # case study on q7_7
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
    plt.plot(neighborhood_losses_p1, label='neighborhood')
    plt.plot(compactness_losses_p1, label='compactness')
    plt.plot(losses_p1, label='overall')
    plt.legend()
    plt.show()
    plt.figure()

    plt.plot(neighborhood_losses_p2, label='neighborhood_p2')
    plt.plot(compactness_losses_p2, label='compactness_p2')
    plt.plot(losses_p2, label='overall_p2')
    plt.legend()
    plt.show()
    plt.figure()

elif mode == "dataset_run":
    # if os.path.exists('results.txt'):
    #     print('results.txt exists.')
    #     sys.exit(0)

    embedding_dim = 10

    dataset = [
        (graph1, 3),
        (graph2, 2),
        (graph3, 3),
        (bipartite_10_vertices, 2),
        (slf_hard, 3),
        (petersen_graph, 3),
        (generate_queens_graph(5,5), 5),
        (generate_queens_graph(6,6), 7),
        (generate_queens_graph(7,7), 7),
        (generate_queens_graph(8,8), 9),
        (generate_queens_graph(8,12), 12),
        (generate_queens_graph(13, 13), 13),
        (Graph.load('../data/singular/ws_10').set_name('ws_10'), 4), # chi
        (Graph.load('../data/singular/ws_100').set_name('ws_100'), 4), # DSATUR
        (Graph.load('../data/singular/ws_1000').set_name('ws_1000'), 4), # DSATUR
        # (Graph.load('../data/singular/ws_10000').set_name('ws_10000'), -1),
        (Graph.load('../data/singular/k_10').set_name('k_10'), 3), # chi
        (Graph.load('../data/singular/k_100').set_name('k_100'), 6), # chi
        (Graph.load('../data/singular/k_1000').set_name('k_1000'), 5), # chi
        # (Graph.load('../data/singular/k_10000').set_name('k_10000'), 4),
        (Graph.load('../data/singular/er_10').set_name('er_10'), 5), # DSATUR
        (Graph.load('../data/singular/er_100').set_name('er_100'), 18), # DSATUR
        # (Graph.load('../data/singular/er_1000').set_name('er_1000'), 114), # DSATUR 
        # (Graph.load('../data/singular/er_10000').set_name('er_10000'), -1),
    ]

    # dataset = [
    #     # (generate_queens_graph(7, 7), 7)
    #     (generate_queens_graph(8,12), 12),
    # ]

    # dataset = [
    #     (Graph.load('../data/singular/new/email-Eu-core.graph').set_name('email-Eu-core'), 21), # DSATUR
    #     (Graph.load('../data/singular/new/CollegeMsg.graph').set_name('CollegeMsg'), 10), # DSATUR
    #     # (Graph.load('../data/singular/new/ca-HepTh.graph').set_name('ca-HepTh'), 32), # DSATUR
    # ]

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
            print('n_used stats: {}, {}'.format(np.mean(n_used_colors_list), np.std(n_used_colors_list)), file=out)
            print('n_used: {}'.format(n_used_colors_list), file=out)
            summary.append([np.mean(n_used_colors_list), np.std(n_used_colors_list), np.mean(violation_ratio_list)])
            print('\n\n', file=out)
            out.flush()

        print('summary:', file=out)
        for item in summary:
            print('{:.1f}, {:.2f}, {:.1f}%'.format(item[0], item[1], 100 * item[2]), file=out)
            # print(', '.join(str(i) for i in item), file=out)
        
        

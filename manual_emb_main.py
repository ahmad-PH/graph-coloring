from graph import Graph
from utility import *
from graph_utility import *
from typing import List
from sklearn.cluster import KMeans
from globals import data
from networkx.algorithms.coloring.greedy_coloring import greedy_color
from networkx.algorithms.clique import find_cliques, graph_clique_number
from exact_coloring import find_k_coloring
from manual_emb_utility import *
from sim_matrix_registry import SimMatrixRegistry

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import os
import sys
import time

def learn_embeddings(graph, embedding_dim, n_clusters = None, use_cached_sim_matrices = False, verbose = False):

    data.losses : Mapping[str, List(float)] = {}
    loss_names = ['neighborhood_loss', 'compactness_loss', 
        'scaled_neighborhood_loss', 'scaled_compactness_loss', 'overall'
    ]
    for loss_name in loss_names:
        data.losses[loss_name] = []

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

        sim_matrix_t1 = time.time()
        registry = SimMatrixRegistry.get_instance()

        if use_cached_sim_matrices:
            similarity_matrix = registry.get_similarity_matrix(graph.name)
            if graph.name == None or similarity_matrix == None:
                if verbose:
                    print('No existing similarity matrix detected for graph {}, creating one.'.format(graph.name))
                similarity_matrix = calculate_similarity_matrix(graph, inverted_adj_matrix, device)
                if graph.name != None:
                    registry.register_similarity_matrix(similarity_matrix, graph.name)
        else:
            similarity_matrix = calculate_similarity_matrix(graph, inverted_adj_matrix, device)
        sim_matrix_t2 = time.time()
        
    optimizer = torch.optim.Adam([embeddings], lr=0.1)

    n_phase1_iterations = 200
    lambda_1_scheduler = LinearScheduler(0.99, 0.5, n_phase1_iterations) 
    if verbose:
        data.n_color_performance = []

    # phase 1
    phase_1_t1 = time.time()
    for i in range(n_phase1_iterations):

        # ======= A bunch of plots and logs: =======

        # if i % 10 == 0 and verbose:
        #     plt.title('{}'.format(i))
        #     # plot_points(embeddings, annotate=True, c=classes_to_colors(proper_coloring))
        #     plot_points(embeddings, annotate=True)
        #     plt.savefig('images/{}'.format(i))
        #     plt.clf()

        if i % 10 == 0 and verbose and n_clusters != None:
            # colors = colorize_embedding_guided_slf_max(embeddings.detach(), graph)
            kmeans = KMeans(n_clusters)
            kmeans.fit(embeddings.detach().cpu().numpy())
            colors = kmeans.labels_
            n_used_colors = len(set(colors))
            data.n_color_performance.append(n_used_colors)

        # ======= The actual optimization: =======

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

        # print('distances:')
        # print(distances)

        lambda_1 = lambda_1_scheduler.get_next_value()
        loss = lambda_1 * neighborhood_loss + (1 - lambda_1) * compactness_loss # lambda_1 used to be for compactness

        data.losses['neighborhood_loss'].append(neighborhood_loss)
        data.losses['compactness_loss'].append(compactness_loss)
        data.losses['scaled_neighborhood_loss'].append(lambda_1 * neighborhood_loss)
        data.losses['scaled_compactness_loss'].append((1 - lambda_1) * compactness_loss)
        data.losses['overall'].append(loss)

        loss.backward()
        optimizer.step()

        # if i % 10 == 0:
            # plt.figure()
            # plot_points(embeddings, title='epoch {}'.format(i), annotate=True)
    phase_1_t2 = time.time()

    # phase 2

    # # slf code:
    # coloring = colorize_embedding_guided_slf_max(embeddings.detach(), graph)
    # assert(is_proper_coloring(coloring, graph))
    # results.n_used_colors = len(set(coloring))

    if n_clusters == None:
        n_clusters = guess_best_n_clusters(embeddings, graph)

    clustering_t1 = time.time()
    clustering_results = []
    for i in range(11):
        if i == 0:
            kmeans = KMeans(n_clusters)
        else:
            kmeans = KMeans(n_clusters, init='random')
        kmeans.fit(embeddings.detach().cpu().numpy())
        cluster_centers = torch.tensor(kmeans.cluster_centers_).to(device)
        colors = kmeans.labels_
        properties = coloring_properties(colors, graph)
        violation_ratio = properties[2]

        # if verbose:
        #     print('colors:')
        #     print(colors)
        #     print('properties:')
        #     print(properties)

            # plt.figure()
            # plot_points(embeddings, annotate=True)
            # plot_points(cluster_centers, c='orange', annotate=True)
            # plt.title('end of phase 2')

        # if verbose:
        #     violators = set([])
        #     for v1, row in enumerate(graph.adj_list):
        #         for v2 in row:
        #             if colors[v1] == colors[v2]:
        #                 violators.add(v1)
        #                 print('violation: ({}, {})'.format(v1, v2))

        correction_t1 = time.time()
        colors = correct_coloring(colors, graph)
        correction_t2 = time.time()

        # if verbose:
        #     print('corrected_colors:')
        #     print(colors)
        #     if is_proper_coloring(colors, graph) == False:
        #         raise Exception('corrected coloring is not correct!')
        #     plt.show()

        n_used_colors = len(set(colors))
        clustering_results.append([n_used_colors, violation_ratio, colors, cluster_centers])

    best_clustering_index = np.argmin([result[0] for result in clustering_results])
    results.n_used_colors, results.violation_ratio, _, best_cluster_centers = clustering_results[best_clustering_index]
    clustering_t2 = time.time()


    sim_matrix_time = sim_matrix_t2 - sim_matrix_t1
    phase1_time = phase_1_t2 - phase_1_t1
    # clustering_time = clustering_t2 - clustering_t1
    # correction_time = correction_t2 - correction_t1

    if verbose:
        print('sim_matrix time: ', sim_matrix_time)
        print('phase1 time: ', phase1_time)
        # print('clustering time: ', clustering_time)
        # print('correction time: ', correction_time)

    return embeddings, results
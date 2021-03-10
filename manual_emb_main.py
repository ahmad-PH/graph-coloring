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

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import os
import sys

def learn_embeddings(graph, n_clusters, embedding_dim, verbose):

    data.neighborhood_losses_p1 : List[float] = []
    data.compactness_losses_p1 : List[float] = []
    data.losses_p1 : List[float] = []

    data.neighborhood_losses_p2 : List[float] = []
    data.compactness_losses_p2 : List[float] = []
    data.losses_p2 : List[float] = []

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

        # common_neighbor_count_matrix = torch.matmul(adj_matrix, adj_matrix).to(device)
        # overlap_matrix_2 = torch.zeros_like(common_neighbor_count_matrix).to(device)

        # for i in range(graph.n_vertices):
        #     for j in range(graph.n_vertices):
        #         if i == j: 
        #             overlap_matrix_2[i][j] = 0.
        #         elif j < i:
        #             overlap_matrix_2[i][j] = overlap_matrix_2[j][i]
        #         else:
        #             intersection = common_neighbor_count_matrix[i][j]
        #             union = (common_neighbor_count_matrix[i][i] + common_neighbor_count_matrix[j][j] - common_neighbor_count_matrix[i][j])
        #             overlap_matrix_2[i][j] = intersection / union 

        #     overlap_matrix_2[i][graph.adj_list[i]] = 0. # suppress entries of neighbors

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
        data.neighborhood_losses_p1.append((1 - lambda_1) * neighborhood_loss)
        data.compactness_losses_p1.append(lambda_1 * compactness_loss)
        data.losses_p1.append(loss)

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
    #     data.neighborhood_losses_p2.append((1 - lambda_2) * neighborhood_loss)
    #     data.compactness_losses_p2.append(lambda_2 * compactness_loss)
    #     data.losses_p2.append(loss)

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

    colors = correct_coloring(colors, graph)

    if verbose:
        print('corrected_colors:')
        print(colors)
        if is_proper_coloring(colors, graph) == False:
            raise Exception('corrected coloring is not correct!')
        print('n_used_colors:', len(set(colors)))

        plt.show()

    results.n_used_colors = len(set(colors))
    return embeddings, results
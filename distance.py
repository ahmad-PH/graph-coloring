from graph import Graph
from test import graph1
import torch
import torch.nn as nn
from utility import *
from matplotlib import pyplot as plt
from test import *
import numpy as np
from typing import List
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from graph_utility import is_proper_coloring
from graph_dataset import GraphDataset, GraphDatasetEager
from globals import data
import networkx as nx
from networkx.algorithms.coloring.greedy_coloring import greedy_color
import os
import sys

neighborhood_losses_p1 : List[float] = []
compactness_losses_p1 : List[float] = []
losses_p1 : List[float] = []

neighborhood_losses_p2 : List[float] = []
compactness_losses_p2 : List[float] = []
losses_p2 : List[float] = []

def correct_coloring(coloring, graph):
    colors_used = set(coloring)
    n_colors_used = len(colors_used)

    while(True):
        n_violations = [0] * graph.n_vertices
        found_violations = False
        for v1, row in enumerate(graph.adj_list):
            for v2 in row:
                if coloring[v1] == coloring[v2]:
                    # print('violation: ({}, {})'.format(v1, v2))
                    n_violations[v1] += 1
                    found_violations = True

        if not found_violations:
            break
        
        fixed_a_vertex = False
        for v in range(graph.n_vertices):
            if n_violations[v] > 0:
                forbidden_colors = set(coloring[graph.adj_list[v]])
                if len(forbidden_colors) < n_colors_used:
                    possible_colors = colors_used - forbidden_colors
                    prev_color = coloring[v] # TEST
                    coloring[v] = next(iter(possible_colors)) # maybe choose something other than 0 ? (the one with least loss maybe)
                    fixed_a_vertex = True
                    # print('vertex fixed without new color: {}, from {} to {}, with possible colors: {}'.format(
                        # v, prev_color, coloring[v], possible_colors
                    # ))
                    break
        
        if fixed_a_vertex:
            continue
        
        max_violator = np.argmax(n_violations)
        new_color = n_colors_used
        # print('choosing new color: {} for the max violator: {}'.format(new_color, max_violator))
        coloring[max_violator] = new_color
        colors_used.add(new_color)
        n_colors_used += 1


# c = np.array([2, 0, 3, 4, 1, 4, 4, 0, 1, 5, 2, 3, 1, 2, 6, 3, 0, 5, 6, 3, 0, 1, 
#     4, 6, 0, 1, 5, 6, 3, 2, 5, 6, 2, 0, 0, 4])

# c = np.array([5, 2, 6, 3, 4, 1, 0, 1, 4, 3, 2, 0, 3, 6, 2, 5, 1, 6, 2, 0, 3, 4, 3, 5,
#         1, 4, 5, 2, 0, 6, 6, 2, 0, 1, 5, 3])

# c = np.array([2, 1, 6, 3, 0, 5, 3, 4, 0, 1, 6, 6, 0, 6, 5, 2, 4, 1, 5, 6, 1, 6, 3, 2,
#         4, 3, 2, 0, 1, 6, 6, 0, 6, 4, 2, 3])

# c = np.array([ 4,  8,  0,  7, 11,  1, 10,  5,  2,  9,  6, 10,  5,  3, 11, 10,  5,  2,
#          6,  8, 12,  1,  0, 11,  2,  4,  2,  4,  9,  6, 12,  3,  0, 10,  7,  8,
#          1,  5,  0,  1,  3,  8,  4,  9,  7,  0,  8,  5,  6,  2,  4,  6, 12,  7,
#          5,  1,  5,  4,  2,  6,  0, 12, 11,  3,  8,  5,  8,  2,  6,  7, 10, 12,
#          4, 11, 10,  9,  1,  7,  0,  6,  1, 12, 10,  0,  9,  7,  8,  4,  3, 11,
#          5,  7,  0,  2,  4, 10,  8,  6,  1,  5, 11,  8,  9,  6,  8,  5,  3,  9,
#          3,  4,  7,  2, 12,  1,  6,  0,  2,  8,  1,  4,  2,  0,  5, 11,  9, 10,
#          6,  7,  6,  9,  9,  2,  7,  8,  6,  8,  6,  3,  4,  2, 10,  0,  1,  7,
#         12,  1,  3,  1,  9,  4,  0,  6,  1,  5,  8, 11,  5,  9,  6, 10,  4, 12,
#          8, 11,  9,  2,  0, 12,  3])

# from utility import generate_queens_graph

# graph = generate_queens_graph(6, 6)

# correct_coloring(c, graph)
# print('new coloring: ', c)
# print('n_colors_used:', len(set(c)))
# print('is proper: ', is_proper_coloring(c, graph.adj_list))
# import sys; sys.exit(0)

def compute_pairwise_distances(vectors1, vectors2):
    n1 = vectors1.shape[0]
    n2 = vectors2.shape[0]

    vectors1_repeated_in_chunks = vectors1.repeat_interleave(n2, dim = 0)
    vectors2_repeated_alternating = vectors2.repeat(n1, 1)
    distances = torch.norm(vectors1_repeated_in_chunks - vectors2_repeated_alternating, dim=1)
    return distances.view(n1, n2)

def highlight_neighborhood(node, graph):
    neighbors = graph.adj_list[node]
    non_neighbors = list(set(range(graph.n_vertices)).difference(neighbors).difference([node]))
    colors = [''] * graph.n_vertices
    for neighbor in neighbors: colors[neighbor] = 'r'
    for non_neighbor in non_neighbors: colors[non_neighbor] = 'b'
    colors[node] = 'g'
    return colors

def to_numpy(x):
    if isinstance(x, np.ndarray): return x
    elif isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    else: raise Exception("unrecognized type for x in to_numpy")

def convert_points_to_xy(points):
    points = to_numpy(points)
    x = np.squeeze(points[:, 0])
    y = np.squeeze(points[:, 1])
    return x, y

def plot_2d_points(points, annotate=False, **kwargs):
    x, y = convert_points_to_xy(points)
    plt.scatter(x, y, **kwargs)
    if annotate:
        for i in range(len(points)):
            plt.annotate(str(i), (x[i], y[i]), xytext=(5,5), textcoords="offset pixels", fontsize='xx-small')
        # plt.annotate("1", embeddings[0].detach().cpu().numpy(), xytext=(5,5), textcoords="offset pixels")

def plot_points(points, annotate=False, **kwargs):
    if points.ndim > 2:
        points = PCA(n_components=2).fit_transform(points)
    plot_2d_points(points, annotate=annotate, **kwargs)


def initialize_embeddings(N, embedding_dim, mode="normal", device="cuda:0"):
    if mode == "normal":
        embeddings = torch.normal(0, 1., (N, embedding_dim), device=device)
    elif mode == "uniform":
        embeddings = torch.empty(N, embedding_dim, device=device).uniform_(-2, 2)
    elif mode == "sphere":
        embeddings = torch.normal(0, 1., (N, embedding_dim), device=device)
        embeddings /= torch.norm(embeddings, dim=1, keepdim=True)
    return nn.Parameter(embeddings)

def compute_neighborhood_losses(embeddings, adj_matrix, precomputed_distances = None):
    if precomputed_distances is None:
        distances = compute_pairwise_distances(embeddings, embeddings)
    else:
        distances = precomputed_distances
    inverse_distances = 1. / (distances + 1e-10)
    return torch.sum(inverse_distances * adj_matrix.float() / torch.sum(adj_matrix, dim=1, keepdim=True), dim=1)

def reinitialize_embeddings(embeddings, loss_function, ratio = 0.1, n_candidates = 10):
    with torch.no_grad():
        top_k = int(embeddings.shape[0] * ratio)
        if top_k == 0:
            return embeddings
        _, topk_indices = torch.topk(loss_function(embeddings), top_k, largest=True)

        low = torch.min(embeddings, dim=0)[0].cpu().numpy()
        high = torch.max(embeddings, dim=0)[0].cpu().numpy()
        candidate_embeddings = np.random.uniform(low, high, size=(n_candidates, top_k, embedding_dim))
        candidate_embeddings = torch.from_numpy(candidate_embeddings).type(torch.float32).to(embeddings.device)
        
        best_loss, best_embeddings = float('+inf'), None
        for i in range(n_candidates):
            new_embeddings = embeddings.clone()
            new_embeddings[topk_indices] = candidate_embeddings[i]

            new_neighborhood_loss = loss_function(new_embeddings).sum()

            if new_neighborhood_loss < best_loss:
                best_loss = new_neighborhood_loss
                best_embeddings = new_embeddings

    embeddings = best_embeddings
    embeddings.requires_grad_(True)
    return embeddings

def learn_embeddings(graph, n_clusters, embedding_dim, verbose):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results = DataDump()

    embeddings = initialize_embeddings(graph.n_vertices, embedding_dim, mode="normal", device=device)

    with torch.no_grad():
        adj_matrix = torch.tensor(adj_list_to_matrix(graph.adj_list)).to(device) 
        inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(device)

        overlap_matrix = torch.zeros(graph.n_vertices, graph.n_vertices).to(device)
        for i in range(graph.n_vertices):
            for j in range(graph.n_vertices):
                if i == j: continue
                n_i = set(graph.adj_list[i])
                n_j = set(graph.adj_list[j])
                overlap_matrix[i][j] = len(n_i.intersection(n_j)) / (len(n_i.union(n_j)) + 1e-8)

            overlap_matrix[i][graph.adj_list[i]] = 0. # suppress entries of neighbors

        global_overlap_matrix = torch.zeros_like(overlap_matrix)
        n_terms = 10
        beta = 0.9
        for i in range(1, n_terms + 1):
            global_overlap_matrix += (beta ** (i-1)) * torch.matrix_power(overlap_matrix, i)
        
        for i in range(graph.n_vertices):
            global_overlap_matrix[i][i] = 0 
            global_overlap_matrix[i][graph.adj_list[i]] = 0 # suppress entries of neighbors

        lambda_3 = 5.
        similarity_matrix = inverted_adj_matrix + lambda_3 * global_overlap_matrix
        
    optimizer = torch.optim.Adam([embeddings], lr=0.1)

    # phase 1
    for i in range(200):
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

        lambda_1 = 0.05
        loss = (1 - lambda_1) * neighborhood_loss + lambda_1 * compactness_loss
        neighborhood_losses_p1.append((1 - lambda_1) * neighborhood_loss)
        compactness_losses_p1.append(lambda_1 * compactness_loss)
        losses_p1.append(loss)

        loss.backward()
        optimizer.step()

        # if i % 10 == 0:
            # plt.figure()
            # plot_points(embeddings, title='epoch {}'.format(i), annotate=True)

    if verbose:
        print('end of phase 1:')
        plt.figure()
        plot_points(embeddings, annotate=True)
        plt.title('end of phase 1')

    # phase 2
    kmeans = KMeans(n_clusters)
    kmeans.fit(embeddings.detach().cpu().numpy())
    cluster_centers = torch.tensor(kmeans.cluster_centers_).to(device)

    if verbose:
        plot_points(cluster_centers, c='orange')

    for i in range(100):
        optimizer.zero_grad()

        # if i % 20 == 0 and i != 0: 
        #     embeddings = reinitialize_embeddings(embeddings,
        #         loss_function=lambda emb: compute_neighborhood_losses(emb, adj_matrix))

        neighborhood_loss = compute_neighborhood_losses(embeddings, adj_matrix).sum()
        
        distances_from_centers = compute_pairwise_distances(embeddings, cluster_centers)
        compactness_loss = torch.sum(torch.min(distances_from_centers, dim=1)[0] ** 2)

        # _lambda = 0.1
        lambda_2 = 0.1
        loss = (1 - lambda_2) * neighborhood_loss + lambda_2 * compactness_loss
        neighborhood_losses_p2.append((1 - lambda_2) * neighborhood_loss)
        compactness_losses_p2.append(lambda_2 * compactness_loss)
        losses_p2.append(loss)

        loss.backward()
        optimizer.step()

    if verbose:
        print('end of phase 2:')
 
    colors = torch.argmin(compute_pairwise_distances(embeddings, cluster_centers), dim=1)
    colors = colors.detach().cpu().numpy()
    properties = coloring_properties(colors, graph.adj_list)
    results.violation_ratio = properties[2]

    if verbose:
        print('colors:')
        print(colors)
        print('properties:')
        print(properties)

        plt.figure()
        plot_points(embeddings, annotate=True)
        plot_points(cluster_centers, c='orange', annotate=True)
        plt.title('end of phase 2')

    # if verbose:
    #     violators = set([])
    #     for v1, row in enumerate(graph.adj_list):
    #         for v2 in row:
    #             if colors[v1] == colors[v2]:
    #                 violators.add(v1)
    #                 print('violation: ({}, {})'.format(v1, v2))

    #     for v in sorted(list(violators)):
    #         plt.figure()
    #         c = highlight_neighborhood(v, graph)
    #         plot_points(embeddings, annotate=True, c=c)
    #         plot_points(cluster_centers, c='orange', annotate=True)

    correct_coloring(colors, graph)

    if verbose:
        print('corrected_colors:')
        print(colors)
        print(is_proper_coloring(colors, graph.adj_list))
        print('n_used_colors:', len(set(colors)))

        plt.show()

    results.n_used_colors = len(set(colors))
    return embeddings, results

# mode = "single_run"
mode = "dataset_run"

if mode == "single_run":

    embedding_dim = 10
    n_clusters = 7

    graph = generate_queens_graph(7,7)
    # graph = generate_queens_graph(20, 20)

    # # knesser graph:
    # n, k = 10, 3
    # graph = generate_kneser_graph(n, k)
    # print(graph.n_vertices, graph.n_edges, kneser_graph_chromatic_number(n, k))

    # graph = Graph(petersen_graph)
    # graph = GraphDatasetEager('../data/erdos_renyi_100/train')[0]
    # greedy_color = greedy_color(graph.get_nx_graph(), strategy="DSATUR") 
    # print(len(set(greedy_color.values())))

    seed = np.random.randint(0, 1000000)
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
        (generate_queens_graph(13, 13), 13)
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
            print('n_used stats: {}, {}'.format(np.mean(n_used_colors_list), np.std(n_used_colors_list)), file=out)
            print('n_used: {}'.format(n_used_colors_list), file=out)
            summary.append([np.mean(n_used_colors_list), np.std(n_used_colors_list), np.mean(violation_ratio_list)])
            print('\n\n', file=out)
            out.flush()

        print('summary:', file=out)
        for item in summary:
            print(', '.join(str(i) for i in item), file=out)
        
        




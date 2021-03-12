from graph import Graph
from utility import *
from graph_utility import *
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import sys



def correct_coloring(coloring, graph: Graph):
    coloring = np.array(coloring)
    for color in coloring:
        if color >= graph.n_vertices:
            raise ValueError("coloring uses colors of larger index than the number of vertices.")
    colors_used = set(coloring)
    colors_unused = set(range(graph.n_vertices)) - colors_used

    # TESTCOUNTER = 0
    while(True):
        # print('finding violators, counter: {}'.format(TESTCOUNTER))
        # TESTCOUNTER += 1
        n_violations = [0] * graph.n_vertices
        found_violations = False
        for v1, row in enumerate(graph.adj_list):
            for v2 in row:
                if coloring[v1] == coloring[v2]:
                    n_violations[v1] += 1
                    found_violations = True

        if not found_violations:
            break

        fixed_a_vertex = False
        for v in range(graph.n_vertices):
            if n_violations[v] > 0:
                forbidden_colors = set(coloring[graph.adj_list[v]])
                available_colors = colors_used - forbidden_colors
                if len(available_colors) > 0:
                    # print('p: {}, n_used: {}, used:{}, forbidden: {}'.format(possible_colors, n_colors_used, colors_used, forbidden_colors))
                    # prev_color = coloring[v] # TEST
                    coloring[v] = next(iter(available_colors)) # maybe choose something other than 0 ? (the one with least loss maybe)
                    fixed_a_vertex = True
                    # print('vertex fixed without new color: {}, from {} to {}, with possible colors: {}'.format(
                        # v, prev_color, coloring[v], possible_colors
                    # ))
                    break
        
        if fixed_a_vertex:
            continue
        
        max_violator = np.argmax(n_violations)
        new_color = next(iter(colors_unused))
        # print('choosing new color: {} for the max violator: {}'.format(new_color, max_violator))
        coloring[max_violator] = new_color
        colors_used.add(new_color)
        colors_unused.remove(new_color)
        # print('added oclor:', new_color)

    return coloring

# def correct_coloring(coloring, graph):
#     coloring = np.array(coloring)
#     colors_used = set(coloring)
#     n_colors_used = len(colors_used)

#     # TESTCOUNTER = 0
#     while(True):
#         # print('finding violators, counter: {}'.format(TESTCOUNTER))
#         # TESTCOUNTER += 1
#         n_violations = [0] * graph.n_vertices
#         found_violations = False
#         for v1, row in enumerate(graph.adj_list):
#             for v2 in row:
#                 if coloring[v1] == coloring[v2]:
#                     n_violations[v1] += 1
#                     found_violations = True

#         if not found_violations:
#             break

#         fixed_a_vertex = False
#         for v in range(graph.n_vertices):
#             if n_violations[v] > 0:
#                 forbidden_colors = set(coloring[graph.adj_list[v]])
#                 if len(forbidden_colors) < n_colors_used:
#                     possible_colors = colors_used - forbidden_colors
#                     print('p: {}, n_used: {}, used:{}, forbidden: {}'.format(possible_colors, n_colors_used, colors_used, forbidden_colors))
#                     # prev_color = coloring[v] # TEST
#                     coloring[v] = next(iter(possible_colors)) # maybe choose something other than 0 ? (the one with least loss maybe)
#                     fixed_a_vertex = True
#                     # print('vertex fixed without new color: {}, from {} to {}, with possible colors: {}'.format(
#                         # v, prev_color, coloring[v], possible_colors
#                     # ))
#                     break
        
#         if fixed_a_vertex:
#             continue
        
#         max_violator = np.argmax(n_violations)
#         new_color = n_colors_used
#         # print('choosing new color: {} for the max violator: {}'.format(new_color, max_violator))
#         coloring[max_violator] = new_color
#         colors_used.add(new_color)
#         n_colors_used += 1
#         print('added oclor:', new_color)

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

def plot_points(points, annotate=False, **kwargs):
    if points.ndim > 2:
        points = PCA(n_components=2).fit_transform(points)
    plot_2d_points(points, annotate=annotate, **kwargs)


def classes_to_colors(classes):
    palette = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    max_class = np.max(classes)
    n_needed_colors = max_class + 1
    if n_needed_colors > len(palette):
        raise ValueError('classes requires more colors than exists in palette.')
    return list(map(lambda c: palette[c], classes))

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
    epsilon = 1e-10
    # inverse_distances_old = 1. / (distances + epsilon)
    inverse_distances = torch.pow(distances + epsilon, -1)
    # print('inverse:')
    # print(inverse_distances)
    n_neighbors = torch.sum(adj_matrix, dim=1, keepdim=True)
    return torch.sum(inverse_distances * adj_matrix.float() / (n_neighbors + epsilon), dim=1)

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

    embeddings.data = best_embeddings.data

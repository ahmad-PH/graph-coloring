from graph import Graph
from heuristics import slf_heuristic, colorize_using_heuristic
import numpy as np
import torch

def find_k_coloring(graph: Graph, k, random_ordering=False):
    if k <= 0:
        raise ValueError('k should be greater than 0.')

    if random_ordering:
        ordering = slf_heuristic(graph.adj_list)
        ordering_len = len(ordering)
        for i in range(int(ordering_len / 3)):
            a, b = torch.randint(0, ordering_len, (2,)).tolist()
            ordering[a], ordering[b] = ordering[b], ordering[a]
    else:
        ordering = slf_heuristic(graph.adj_list)

    coloring = np.array([-1] * graph.n_vertices)

    if _find_k_coloring_rec(graph, k, ordering, 0, 0, coloring):
        return coloring.tolist()
    else:
        return None

def _find_k_coloring_rec(graph: Graph, k, ordering, curr_ind, n_used_colors, coloring):
    if curr_ind == graph.n_vertices: # if all nodes have been colored
        return True

    curr_node = ordering[curr_ind]
    neighbors = graph.adj_list[curr_node]
    forbidden_colors = set(coloring[neighbors]) - set([-1])
    available_colors = set(set(range(n_used_colors)) - forbidden_colors)

    # print('ind: {}, node: {}, \nneighbs: {}, \ncolors:{}, \nforb: {}, \navail: {}\n\n'.format(
    #     curr_ind, curr_node, neighbors, coloring, forbidden_colors, available_colors
    # ))

    for color in available_colors:
        coloring[curr_node] = color
        if _find_k_coloring_rec(graph, k, ordering, curr_ind + 1, n_used_colors, coloring):
            return True

    if n_used_colors < k:
        coloring[curr_node] = n_used_colors
        if _find_k_coloring_rec(graph, k, ordering, curr_ind + 1, n_used_colors + 1, coloring):
            return True

    coloring[curr_node] = -1 # if all attempts fail, cleanup the coloring.
    return False

def find_chromatic_number(graph: Graph, verbose=False):
    _, n_colors_used = colorize_using_heuristic(graph.adj_list, slf_heuristic)
    if verbose: print('color from heuristic: ', n_colors_used)

    k = n_colors_used - 1
    while True:
        if verbose: print(f'k: {k}')
        k_coloring = find_k_coloring(graph, k)
        if k_coloring == None:
            return k + 1
        else:
            k -= 1


# from graph_utility import *
# from manual_emb_utility import calculate_similarity_matrix
# from utility import tensor_correlation
# import sys
# from matplotlib import pyplot as plt

# if __name__=="__main__":

#     # ============================ create and save =============================

#     # graph = generate_erdos_renyi_graph(50, 0.3)
#     # graph.save("test.graph"); sys.exit(0)

#     graph = Graph.load("test.graph")
#     # chi = find_chromatic_number(graph)
#     # print(chi); sys.exit(0)

#     similarity_matrix = torch.zeros(graph.n_vertices, graph.n_vertices)

#     colorize_n_times = 10
#     for coloring_counter in range(colorize_n_times):
#         print(coloring_counter)
#         coloring = find_k_coloring(graph, 6, random_ordering=True)
#         clustering_matrix = torch.zeros(graph.n_vertices, graph.n_vertices)

#         for i in range(graph.n_vertices):
#             for j in range(graph.n_vertices):
#                 if i == j: continue
#                 clustering_matrix[i][j] = 1 if (coloring[i] == coloring[j]) else 0
                
#         print('coloring:')
#         print([(i,v) for i,v in enumerate(coloring)])
#         print('clust matrxi:')
#         print(clustering_matrix)

#         similarity_matrix += clustering_matrix
#         print('new sim:')
#         print(similarity_matrix)

#     similarity_matrix /= colorize_n_times
#     print('final sim:')
#     print(similarity_matrix)

#     torch.save(similarity_matrix, 'sim_mat.pt')

#     sys.exit(0)

#     # ============================ load and experiment =============================

#     graph = Graph.load("test.graph")
#     similarity_matrix = torch.load('sim_mat.pt')

#     calculate_similarity_matrix(graph, )
#     tensor_correlation()

#     # clamped = torch.clamp_min(similarity_matrix, 0.5)
#     # torch.set_printoptions(profile="full", precision=1)
#     # print(similarity_matrix)
#     # plt.imshow(clamped)
#     # plt.colorbar()
#     # plt.show()

#     # for i in range(similarity_matrix.shape[0]):
#     #     for j in range(similarity_matrix.shape[1]):
#     #         if similarity_matrix[i][j] > 0.8:
#     #             print(f'sim of {similarity_matrix[i][j]} between {i}, {j}')

#     sys.exit(0)

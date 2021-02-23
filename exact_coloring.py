from graph import Graph
from heuristics import slf_heuristic
import numpy as np

def find_k_coloring(graph: Graph, k):
    if k <= 0:
        raise ValueError('k should be greater than 0.')

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
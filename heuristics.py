from graph import Graph

import numpy as np
import heapq
from networkx.algorithms.coloring import strategy_saturation_largest_first 
import itertools

# def slf_heuristic(adj_list):
    # n_vertices = len(adj_list)
    # degrees = [len(row) for row in adj_list]
    # satur_degrees = [0] * n_vertices
    # remaining_vertices = list(range(n_vertices))

    # ordering = []
    # for i in range(n_vertices):
    #     chosen_vertex = remaining_vertices[0] # any of the remaining_vertices can be used for initialization
    #     for v in remaining_vertices:
    #         if satur_degrees[v] > satur_degrees[chosen_vertex]:
    #             chosen_vertex = v

    #         elif satur_degrees[v] == satur_degrees[chosen_vertex]:
    #             if degrees[v] > degrees[chosen_vertex]:
    #                 chosen_vertex = v

    #     ordering.append(chosen_vertex)
    #     remaining_vertices.remove(chosen_vertex)

    #     for neighb in adj_list[chosen_vertex]:

    # return ordering

def slf_heuristic(adj_list):
    G = Graph(adj_list).get_nx_graph()
    colors = {}
    nodes = strategy_saturation_largest_first(G, colors)
    ordering = []
    for u in nodes:
        ordering.append(u)
        neighbour_colors = {colors[v] for v in G[u] if v in colors}
        for color in itertools.count():
            if color not in neighbour_colors:
                break
        colors[u] = color
    return ordering
        

# OPTIMIZE: replace np.argmax with heap
def highest_colored_neighbor_heuristic(adj_list):
    n_vertices = len(adj_list)

    degrees = [len(row) for row in adj_list]
    first_vertex = np.argmax(degrees)
    color_degrees = np.zeros(n_vertices)
    color_degrees[first_vertex] = float('+inf')

    result = []
    while len(result) < n_vertices:
        next_vertex = np.argmax(color_degrees)
        result.append(next_vertex)
        color_degrees[adj_list[next_vertex]] += 1
        color_degrees[next_vertex] = float('-inf')

    return result


def unordered_heuristic(adj_list):
    return list(range(len(adj_list)))

def ordered_heuristic(adj_list):
    degrees = [len(row) for row in adj_list]
    vertex_degrees = [(i, degrees[i]) for i in range(len(adj_list))]
    vertex_degrees.sort(key=lambda x: x[1], reverse=True) # sort based on degrees, descending
    return [x[0] for x in vertex_degrees]

# OPTIMIZE: replace np.argmax with heap
def dynamic_ordered_heuristic(adj_list): 
    dynamic_degrees = [len(row) for row in adj_list]

    result = []
    for _ in range(len(adj_list)):
        next_vertex = np.argmax(dynamic_degrees)
        result.append(next_vertex)
        dynamic_degrees[next_vertex] = float('-inf')
        for neighbor in adj_list[next_vertex]:
            dynamic_degrees[neighbor] -= 1
    return result


def colorize_using_heuristic(adj_list, heuristic):
    colors = np.array([None] * len(adj_list))
    next_color = 0

    vertex_order = heuristic(adj_list)
    for i, vertex in enumerate(vertex_order): # remove enumerate
        colors_used = set(colors[adj_list[vertex]])
        for color in range(next_color):
            if color not in colors_used:
                colors[vertex] = color
                break
        new_color = False #remove 
        if colors[vertex] == None: # all colors were used
            new_color = True # remove
            colors[vertex] = next_color
            next_color += 1

        # print("i: {} \nvertex: {} \ndegree: {} \nneighbors: {} \ncolors_used: {} \ncolor_chosen: {} {}\n".format(i, vertex, len(adj_list[vertex]), adj_list[vertex], colors_used, colors[vertex], "(New)" if new_color else ""))
        # if i == 200:
            # break
    
    n_colors_used = next_color
    return colors, n_colors_used
        
def run_heuristic_on_dataset(heuristic, ds):
    results = []
    for i, graph in enumerate(ds):
        coloring, n_colors = colorize_using_heuristic(graph.adj_list, heuristic)

        # is_proper = is_proper_coloring(coloring, graph.adj_list)
        print('graph {}, n_colors: {}, is_proper: {}'.format(i, n_colors, None))

        results.append(n_colors)
    avg = np.mean(results)
    print('avg:', avg)
    return avg

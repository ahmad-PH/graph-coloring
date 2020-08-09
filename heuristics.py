import numpy as np
import heapq

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
    for vertex in vertex_order:
        colors_used = set(colors[adj_list[vertex]])
        for color in range(next_color):
            if color not in colors_used:
                colors[vertex] = color
                break
        if colors[vertex] == None: # all colors were used
            colors[vertex] = next_color
            next_color += 1
    
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

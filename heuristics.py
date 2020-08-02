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



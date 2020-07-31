import numpy as np

a = [
    [1, 3],
    [0, 2, 3, 4],
    [1, 4],
    [0, 1, 4],
    [1, 2, 3],
]

def adj_list_to_matrix(adj_list):
    n = len(adj_list)
    adj_matrix = np.zeros((n,n))
    for i in range(n):
        adj_matrix[i, adj_list[i]] = 1
    return adj_matrix

def adj_matrix_to_list(adj_matrix):
    adj_list = []
    for i in range(adj_matrix.shape[0]):
        adj_list.append([])
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1: 
                adj_list[-1].append(j)
    return adj_list

    
def vertex_order_heuristic(adj_list):
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

if __name__=='__main__':
    order = vertex_order_heuristic(a)
    print(order)


import math
import heapq
import numpy as np
import itertools
from graph import Graph

def n_combinations(n, k):
    if n < k:
        raise ValueError('n cannot be smaller than k in combination(n, k)')
    return int(math.factorial(n) / (math.factorial(n-k) * math.factorial(k)))

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

def is_proper_coloring(coloring, adj_list):
    for v1, row in enumerate(adj_list):
        for v2 in row:
            if coloring[v1] == coloring[v2]:
                return False
    return True
            
class EWMAWithCorrection:
    def __init__(self, beta=0.95):
        self._avg = 0.
        self._corrected = 0.
        self._beta = beta
        self._counter = 0

    def update(self, new_value):
        self._avg = self._avg * self._beta + new_value * (1 - self._beta)
        self._counter += 1
        self._corrected = self._avg / (1 - self._beta ** self._counter)

    def get_value(self):
        return self._corrected


class EWMA:
    def __init__(self, beta=0.95):
        self._avg = 0.
        self._beta = beta

    def update(self, new_value):
        self._avg = self._avg * self._beta + new_value * (1 - self._beta)

    def reset(self, reset_value=0):
        self._avg = reset_value

    def get_value(self):
        return self._avg



def generate_kneser_graph(n, k):
    subsets = [frozenset(subset) for subset in itertools.combinations(range(n), k)]
    adj_list = [[] for _ in range(len(subsets))]
    for i in range(len(subsets)):
        for j in range(i+1, len(subsets)):
            if subsets[i].isdisjoint(subsets[j]):
                adj_list[i].append(j)
                adj_list[j].append(i)
    return Graph(adj_list)

def generate_queens_graph(m, n):
    chess_coord_to_vertex_ind = lambda i, j: i * n + j
    adj_list = [[] for _ in range(m * n)]
    for i1 in range(m):
        for j1 in range(n):
            index = chess_coord_to_vertex_ind(i1, j1)

            # add the row
            i2 = i1
            for j2 in [x for x in range(n) if x != j1]:
                adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))

            # add the column
            j2 = j1
            for i2 in [x for x in range(m) if x != i1]:
                adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))

            # add the two diagonals
            for i2 in [x for x in range(m) if x != i1]:
                j2 = j1 + (i2-i1)
                if j2 >= 0 and j2 < n:
                    adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))

                j2 = j1 - (i2-i1)
                if j2 >= 0 and j2 < n:
                    adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))


    return Graph(adj_list)
            

def sort_graph_adj_list(adj_list):
    for neighborhood in adj_list:
        neighborhood.sort()

def n_edges(adj_list):
    sum_of_degrees =  sum([len(neighborhood) for neighborhood in adj_list])
    if sum_of_degrees % 2 != 0:
        raise ValueError('sum of degrees in adjacency list is not even.')
    return int(sum_of_degrees / 2)


# class ComparableContainer:
#     def __init__(self, item, key):
#         self.item = item
#         self.key = key

#     def __lt__(self, other):
#         return self.key(self.item) < self.key(other.item)

#     def __le__(self, other):
#         return self.key(self.item) <= self.key(other.item)

#     def __gt__(self, other):
#         return self.key(self.item) > self.key(other.item)

#     def __ge__(self, other):
#         return self.key(self.item) >= self.key(other.item)

#     def __eq__(self, other):
#         return self.key(self.item) == self.key(other.item)

#     def __str__(self):
#         return str(self.item)

# class MinHeap:
#     def __init__(self, _list, key):
#         self.heap = [ComparableContainer(item, key) for item in _list]
#         self.heap = heapq.heapify(self.heap)
#         self.key = key

    

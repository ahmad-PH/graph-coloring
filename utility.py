import math
import heapq
import numpy as np

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

class DataDump:
    def __init__(self):
        object.__setattr__(self, "data", {})

    def __setattr__(self, name, value):
        self.data[name] = value

    def __getattr__(self, name):
        return self.data[name]

    def __repr__(self):
        return self.data.__repr__()


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

    

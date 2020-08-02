import math
import heapq

def combinations(n, k):
    if n < k:
        raise ValueError('n cannot be smaller than k in combination(n, k)')
    return int(math.factorial(n) / (math.factorial(n-k) * math.factorial(k)))


class ComparableContainer:
    def __init__(self, item, key):
        self.item = item
        self.key = key

    def __lt__(self, other):
        return self.key(self.item) < self.key(other.item)

    def __le__(self, other):
        return self.key(self.item) <= self.key(other.item)

    def __gt__(self, other):
        return self.key(self.item) > self.key(other.item)

    def __ge__(self, other):
        return self.key(self.item) >= self.key(other.item)

    def __eq__(self, other):
        return self.key(self.item) == self.key(other.item)

    def __str__(self):
        return str(self.item)

class MinHeap:
    def __init__(self, _list, key):
        self.heap = [ComparableContainer(item, key) for item in _list]
        self.heap = heapq.heapify(self.heap)
        self.key = key

    

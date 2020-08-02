import unittest
import math

from utility import combinations
from heuristics import highest_colored_neighbor_heuristic, dynamic_ordered_heuristic

class TestHighestColoredNeighborHeuristic(unittest.TestCase):
    def test_graph1(self):
        adj_list = [
            [1, 3],
            [0, 2, 3, 4],
            [1, 4],
            [0, 1, 4],
            [1, 2, 3],
        ]

        self.assertEqual(highest_colored_neighbor_heuristic(adj_list), [1, 0, 3, 4, 2])


class TestDynamicOrderedHeuristic(unittest.TestCase):
    def test_graph1(self):
        adj_list = [
            [1, 3],
            [0, 2, 3, 4],
            [1, 4],
            [0, 1, 4],
            [1, 2, 3],
        ]

        self.assertEqual(dynamic_ordered_heuristic(adj_list), [1,3,2,0,4])

    def test_graph2(self):
        adj_list = [
            [1,2],
            [0],
            [0],
            [4],
            [3],
        ]

        self.assertEqual(dynamic_ordered_heuristic(adj_list), [0,3,1,2,4])
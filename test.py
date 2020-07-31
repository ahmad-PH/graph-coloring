import unittest
import math

from main import *
from utility import combinations

class TestVertexOrderHeuristic(unittest.TestCase):
    def test_graph1(self):
        adj_list = [
            [1, 3],
            [0, 2, 3, 4],
            [1, 4],
            [0, 1, 4],
            [1, 2, 3],
        ]

        self.assertEqual(vertex_order_heuristic(adj_list), [1, 0, 3, 4, 2])

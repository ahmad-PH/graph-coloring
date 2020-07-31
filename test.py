import unittest
import math

from main import *
from graph_generators import *
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


class TestErdosRenyiGenerator(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_probability(self):
        p = 0.3
        possible_edges = actual_edges = 0
        for _ in range(1000):
            n = np.random.randint(3, 30)
            adj_matrix = erdos_renyi(n, p)
            actual_edges += np.sum(adj_matrix) / 2
            possible_edges += combinations(n, 2)
        self.assertAlmostEqual(actual_edges / possible_edges, p, places=2)

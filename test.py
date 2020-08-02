import unittest
import math

from utility import combinations
from heuristics import *

graph1 =  [
    [1, 3],
    [0, 2, 3, 4],
    [1, 4],
    [0, 1, 4],
    [1, 2, 3],
]

graph2 = [
    [1,2],
    [0],
    [0],
    [4],
    [3],
]

class TestHighestColoredNeighborHeuristic(unittest.TestCase):
    def test_graph1(self):
        self.assertListEqual(highest_colored_neighbor_heuristic(graph1), [1, 0, 3, 4, 2])

    def test_graph2(self):
        self.assertListEqual(highest_colored_neighbor_heuristic(graph2), [0, 1, 2, 3, 4])


class TestDynamicOrderedHeuristic(unittest.TestCase):
    def test_graph1(self):
        self.assertListEqual(dynamic_ordered_heuristic(graph1), [1,3,2,0,4])

    def test_reduces_degrees_dynamically(self):
        self.assertListEqual(dynamic_ordered_heuristic(graph2), [0,3,1,2,4])


class TestColorizeUsingHeuristic(unittest.TestCase):
    def test_unordered_graph1(self):
        colors, n_colors = colorize_using_heuristic(graph1, unordered_heuristic)
        self.assertListEqual(list(colors), [0, 1, 0, 2, 3])
        self.assertEqual(n_colors, 4)

    def test_ordered_graph1(self):
        colors, n_colors = colorize_using_heuristic(graph1, ordered_heuristic)
        self.assertListEqual(list(colors), [2, 0, 1, 1, 2])
        self.assertEqual(n_colors, 3)

    def test_dynamic_ordered_graph2(self):
        colors, n_colors = colorize_using_heuristic(graph2, dynamic_ordered_heuristic)
        self.assertListEqual(list(colors), [0, 1, 1, 0, 1])
        self.assertEqual(n_colors, 2)

    def test_highest_colored_neighbor_graph2(self):
        colors, n_colors = colorize_using_heuristic(graph2, highest_colored_neighbor_heuristic)
        self.assertListEqual(list(colors), [0, 1, 1, 0, 1])
        self.assertEqual(n_colors, 2)
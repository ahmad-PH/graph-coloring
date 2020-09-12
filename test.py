import unittest
import math

from heuristics import *
from GAT import GraphAttentionLayer, GraphSingularAttentionLayer
from graph_colorizer import GraphColorizer, ColorClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F

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



class TestGSATImplementation(unittest.TestCase):
    def test_small_embedding(self):
        # the torch.manual_seed call is to ensure they both initialize parameters the same way.
        torch.manual_seed(0)
        GAT = GraphAttentionLayer(10, 10, 0, 0.1)

        torch.manual_seed(0)
        GSAT = GraphSingularAttentionLayer(10, 10, 0, 0.1)

        test_embeddings = torch.randn(4, 10)

        adj_matrix = torch.tensor([
            [1,1,1,0],
            [1,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
        ])

        gat_result = GAT.forward(test_embeddings, adj_matrix)
        gsat_result = GSAT.forward(test_embeddings[[0,1,2]], 0)

        self.assertTrue(torch.allclose(gat_result[0], gsat_result))

    def test_large_embedding(self):
        torch.manual_seed(0)
        GAT = GraphAttentionLayer(100, 100, 0, 0.1)

        torch.manual_seed(0)
        GSAT = GraphSingularAttentionLayer(100, 100, 0, 0.1)

        test_embeddings = torch.randn(4, 100)

        adj_matrix = torch.tensor([
            [1,1,1,0],
            [1,1,0,1],
            [1,0,1,0],
            [0,1,0,1],
        ])

        gat_result = GAT.forward(test_embeddings, adj_matrix)
        gsat_result = GSAT.forward(test_embeddings[[0,1,2]], 0)

        self.assertTrue(torch.allclose(gat_result[0], gsat_result, rtol=1e-4))

class TestColorClassifier(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(0)

    def _create_sample_data(self, embedding_size, n_possible_colors, sample_size = 10):
        layer1 = nn.Linear(embedding_size, 200).to(self.device)
        layer2 = nn.Linear(200, n_possible_colors + 1).to(self.device)
        nn.init.uniform_(layer1.weight.data, -1, 1)
        nn.init.uniform_(layer2.weight.data, -1, 1)

        with torch.no_grad():
            x = torch.randn(sample_size, embedding_size).to(self.device)
            y = F.softmax(layer2(torch.tanh(layer1(x))), dim = 1).to(self.device)

        return x, y

    def test_with_batch(self):
        embedding_size = 200
        n_possible_colors = 50
        x, y = self._create_sample_data(embedding_size, n_possible_colors)
        classifier = ColorClassifier(embedding_size, n_possible_colors, run_assertions=False).to(self.device)

        optimizer = torch.optim.Adam(classifier.parameters())
        loss_function = nn.MSELoss()

        for i in range(200):
            optimizer.zero_grad()
            y_hat = classifier(x, n_possible_colors).to(self.device)
            loss = loss_function(y_hat, y)
            loss.backward()
            optimizer.step()
        
        self.assertLess(loss.item(), 0.0005)

    def test_without_batch(self):
        embedding_size = 30
        n_possible_colors = 5
        x, y = self._create_sample_data(embedding_size, n_possible_colors, 5)
        classifier = ColorClassifier(embedding_size, n_possible_colors, run_assertions=False).to(self.device)

        optimizer = torch.optim.Adam(classifier.parameters())
        loss_function = nn.MSELoss()

        for i in range(200):
            for j in range(x.shape[0]):
                optimizer.zero_grad()
                y_hat = classifier(x[j], n_possible_colors).to(self.device)
                loss = loss_function(y_hat, y[j])
                loss.backward()
                optimizer.step()
        
        self.assertLess(loss.item(), 0.0005)

    def test_n_used_colors_masks_propery(self):
        embedding_size = 100
        n_possible_colors = 20
        x, _ = self._create_sample_data(embedding_size, n_possible_colors)
        classifier = ColorClassifier(embedding_size, n_possible_colors, run_assertions=False).to(self.device)

        y_hat = classifier(x[0], 10)
        self.assertEqual(torch.sum(y_hat[10:20]).item(), 0.)
        self.assertEqual(torch.sum(y_hat).item(), 1.)

        y_hat = classifier(x[0], 0)
        self.assertEqual(y_hat[20].item(), 1.0)

    def test_learns_when_half_the_colors_are_used(self):
        embedding_size = 100
        n_possible_colors = 20
        x, y = self._create_sample_data(embedding_size, n_possible_colors, 10)
        classifier = ColorClassifier(embedding_size, n_possible_colors, run_assertions=False).to(self.device)

        optimizer = torch.optim.Adam(classifier.parameters())

        n_used_colors = 10
        ignored_predictions = torch.tensor([0] * n_used_colors + [1] * (n_possible_colors-n_used_colors) + [0], dtype=torch.bool) \
                             .repeat(y.shape[0], 1) \
                             .to(self.device)

        for i in range(200):
            optimizer.zero_grad()
            y_hat = classifier(x, n_possible_colors).to(self.device)
            loss = ((y_hat - y)**2).masked_fill(ignored_predictions, 0.).mean()
            loss.backward()
            optimizer.step()
        
        self.assertLess(loss.item(), 0.00001)

    def test_neighboring_colors_masked_properly(self):
        embedding_size = 100
        n_possible_colors = 20
        x, _ = self._create_sample_data(embedding_size, n_possible_colors, 1)
        classifier = ColorClassifier(embedding_size, n_possible_colors, run_assertions=False).to(self.device)
        
        adj_colors = [1,3,5]
        y_hat = classifier(x[0], n_possible_colors, adj_colors=None)
        for adj_color in adj_colors:
            self.assertGreater(y_hat[adj_color].item(), 0.)

        y_hat = classifier(x[0], n_possible_colors, adj_colors=adj_colors)
        for adj_color in adj_colors:
            self.assertEqual(y_hat[adj_color].item(), 0.)
        







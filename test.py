import unittest
import math

from heuristics import *
from GAT import GraphAttentionLayer, GraphSingularAttentionLayer
from graph_colorizer1 import GraphColorizer, ColorClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import EWMA, generate_kneser_graph, generate_queens_graph, sort_graph_adj_list

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

petersen_graph_manual = [
    [1,4,5], # 0
    [0,2,6], # 1
    [1,3,7], # 2
    [2,4,8], # 3
    [0,3,9], # 4
    [0,7,8], # 5
    [1,8,9], # 6
    [2,5,9], # 7
    [3,5,6], # 8
    [4,6,7], # 9
]

petersen_graph = [
    [7, 8, 9], # 0
    [5, 6, 9], # 1
    [4, 6, 8], # 2
    [4, 5, 7], # 3
    [2, 3, 9], # 4
    [1, 3, 8], # 5
    [1, 2, 7], # 6
    [0, 3, 6], # 7
    [0, 2, 5], # 8
    [0, 1, 4], # 9
]

bipartite_10_vertices = [
    [5,6,7,8,9], # 0
    [5,6,7,8,9], # 1
    [5,6,7,8,9], # 2
    [5,6,7,8,9], # 3
    [5,6,7,8,9], # 4
    [0,1,2,3,4], # 5
    [0,1,2,3,4], # 6
    [0,1,2,3,4], # 7
    [0,1,2,3,4], # 8
    [0,1,2,3,4], # 9
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
        

class TestLossComputation(unittest.TestCase):
    def setUp(self):
        self.colorizer = GraphColorizer()
    
    def _one_hot(self, index, value):
        result = torch.zeros(self.colorizer.n_possible_colors + 1)
        result[index] = value
        return result

    def test_adj_penalty_decrease(self):
        adj_colors = [0]
        l1 = self.colorizer._compute_color_classifier_loss(self._one_hot(0, 0.9), adj_colors)
        l2 = self.colorizer._compute_color_classifier_loss(self._one_hot(0, 0.8), adj_colors)
        self.assertLess(l2, l1)

    def test_new_color_penalty_decrease(self):
        adj_colors = []
        l1 = self.colorizer._compute_color_classifier_loss(self._one_hot(self.colorizer.n_possible_colors, 0.9), adj_colors)
        l2 = self.colorizer._compute_color_classifier_loss(self._one_hot(self.colorizer.n_possible_colors, 0.8), adj_colors)
        self.assertLess(l2, l1)

    def test_adj_penalty_greater_than_new_color_penalty(self):
        adj_colors = [1]
        adj_penalty = self.colorizer._compute_color_classifier_loss(self._one_hot(1, 0.9), adj_colors)
        new_color_penalty = self.colorizer._compute_color_classifier_loss(self._one_hot(self.colorizer.n_possible_colors, 0.9), adj_colors)
        self.assertGreater(adj_penalty, new_color_penalty)    

    def test_no_penalty_for_correct_choice(self):
        adj_colors = [0,2]
        loss = self.colorizer._compute_color_classifier_loss(self._one_hot(1, 0.9), adj_colors)
        self.assertAlmostEqual(loss, 0.)



class TestEWMA(unittest.TestCase):
    def test_no_values(self):
        avg = EWMA()
        self.assertEqual(avg.get_value(), 0.)

    def test_single_value(self):
        avg = EWMA()
        avg.update(1.0)
        self.assertEqual(avg.get_value(), 1.0)

    def test_multiple_values(self):
        avg = EWMA(0.8)
        avg.update(5.)
        avg.update(10.)
        avg.update(11.)
        self.assertAlmostEqual(avg.get_value(), 9.098, places=2)


class TestKneserGraphGenerator(unittest.TestCase):
    def test_petersen_graph(self):
        graph = generate_kneser_graph(5,2)
        self.assertListEqual(graph.adj_list, petersen_graph)
    
    def test_complete_graph(self):
        graph = generate_kneser_graph(4, 1)
        self.assertListEqual(graph.adj_list, [
            [1,2,3],
            [0,2,3],
            [0,1,3],
            [0,1,2]
        ]) # assert that it is the complete graph with 4 vertices


class TestQueensGraphGenerator(unittest.TestCase):
    def test_2by2(self):
        graph = generate_queens_graph(2,2)
        sort_graph_adj_list(graph.adj_list)
        self.assertListEqual(graph.adj_list,
            [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        )

    def test_3by3(self):
        graph = generate_queens_graph(3,3)
        sort_graph_adj_list(graph.adj_list)
        self.assertListEqual(graph.adj_list[0], [1,2,3,4,6,8])
        self.assertListEqual(graph.adj_list[8], [0,2,4,5,6,7])

    def test_2by3(self):
        graph = generate_queens_graph(2,3)
        sort_graph_adj_list(graph.adj_list)
        self.assertListEqual(graph.adj_list[0], [1,2,3,4])
        self.assertListEqual(graph.adj_list[1], [0,2,3,4,5])
        self.assertListEqual(graph.adj_list[5], [1,2,3,4])
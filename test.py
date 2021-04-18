import unittest
import math
from io import StringIO
import os, shutil

from heuristics import *
from GAT import GraphAttentionLayer, GraphSingularAttentionLayer
from colorizer_1_plain import ColorClassifier
from colorizer_8_pointer_netw import GraphColorizer as PointerColorizer
from graph import Graph
import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import EWMAWithCorrection, LinearScheduler
from graph_utility import generate_kneser_graph, generate_queens_graph, sort_graph_adj_list, \
    is_proper_coloring, coloring_properties
from snap_utility import edgelist_eliminate_self_loops
from exact_coloring import find_k_coloring
from sim_matrix_registry import SimMatrixRegistry
from manual_emb_utility import correct_coloring, colorize_embedding_guided_slf_mean, colorize_embedding_guided_slf_max

graph1 = Graph([
    [1, 3],
    [0, 2, 3, 4],
    [1, 4],
    [0, 1, 4],
    [1, 2, 3],
], 'graph1')

graph2 = Graph([
    [1,2],
    [0],
    [0],
    [4],
    [3],
], 'graph2')

graph3 = Graph([
    [1,2,3,4,5],
    [0,2,3,7],
    [0,1,6],
    [0,1],
    [0],
    [0],
    [2],
    [1]
], 'graph3')

graph4 = Graph([
    [1,2,3],
    [0,4],
    [0,4],
    [0,4],
    [1,2,3,5,6],
    [4,7],
    [4,7],
    [5,6]
], 'graph4')

slf_hard = Graph([
    [1,2,3,4],
    [0,2,3,5],
    [0,1],
    [0,1],
    [0,6,7],
    [1,6,7],
    [4,5,7],
    [4,5,6],
], 'slf_hard')

petersen_graph_manual = Graph([
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
], 'petersen_manual')

petersen_graph = Graph([
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
], 'petersen')

bipartite_10_vertices = Graph([
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
], 'bipartite_10_vertices')

class TestHighestColoredNeighborHeuristic(unittest.TestCase):
    def test_graph1(self):
        self.assertListEqual(highest_colored_neighbor_heuristic(graph1.adj_list), [1, 0, 3, 4, 2])

    def test_graph2(self):
        self.assertListEqual(highest_colored_neighbor_heuristic(graph2.adj_list), [0, 1, 2, 3, 4])


class TestDynamicOrderedHeuristic(unittest.TestCase):
    def test_graph1(self):
        self.assertListEqual(dynamic_ordered_heuristic(graph1.adj_list), [1,3,2,0,4])

    def test_reduces_degrees_dynamically(self):
        self.assertListEqual(dynamic_ordered_heuristic(graph2.adj_list), [0,3,1,2,4])


class TestColorizeUsingHeuristic(unittest.TestCase):
    def test_unordered_graph1(self):
        colors, n_colors = colorize_using_heuristic(graph1.adj_list, unordered_heuristic)
        self.assertListEqual(list(colors), [0, 1, 0, 2, 3])
        self.assertEqual(n_colors, 4)

    def test_ordered_graph1(self):
        colors, n_colors = colorize_using_heuristic(graph1.adj_list, ordered_heuristic)
        self.assertListEqual(list(colors), [2, 0, 1, 1, 2])
        self.assertEqual(n_colors, 3)

    def test_dynamic_ordered_graph2(self):
        colors, n_colors = colorize_using_heuristic(graph2.adj_list, dynamic_ordered_heuristic)
        self.assertListEqual(list(colors), [0, 1, 1, 0, 1])
        self.assertEqual(n_colors, 2)

    def test_highest_colored_neighbor_graph2(self):
        colors, n_colors = colorize_using_heuristic(graph2.adj_list, highest_colored_neighbor_heuristic)
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
        
        self.assertLess(loss.item(), 1e-4)

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
        

class TestEWMAWithCorrection(unittest.TestCase):
    def test_no_values(self):
        avg = EWMAWithCorrection()
        self.assertEqual(avg.get_value(), 0.)

    def test_single_value(self):
        avg = EWMAWithCorrection()
        avg.update(1.0)
        self.assertEqual(avg.get_value(), 1.0)

    def test_multiple_values(self):
        avg = EWMAWithCorrection(0.8)
        avg.update(5.)
        avg.update(10.)
        avg.update(11.)
        self.assertAlmostEqual(avg.get_value(), 9.098, places=2)


class TestKneserGraphGenerator(unittest.TestCase):
    def test_petersen_graph(self):
        graph = generate_kneser_graph(5,2)
        self.assertListEqual(graph.adj_list, petersen_graph.adj_list)
    
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

    def test_on_queens6_6(self):
        graph = generate_queens_graph(6, 6)
        self.assertEqual(graph.n_edges, 290)

    def test_on_queens8_8(self):
        graph = generate_queens_graph(8, 8)
        self.assertEqual(graph.n_edges, 728)


class TestSLFHeuristic(unittest.TestCase):
    def test_graph3(self):
        ordering = slf_heuristic(graph3.adj_list)
        self.assertListEqual(ordering[:4], [0,1,2,3])


class TestPointerColorizer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.colorizer = PointerColorizer(device=self.device)
        self.n_possible_colors = self.colorizer.n_possible_colors

    def test_full_mask(self):
        mask = self.colorizer._get_pointer_mask(0, []).squeeze(0)
        for i in range(0, self.n_possible_colors):
            self.assertEqual(mask[i], True)
        self.assertEqual(mask[self.n_possible_colors], False)

    def test_some_used_no_adj(self):
        mask = self.colorizer._get_pointer_mask(3, []).squeeze(0)
        for i in range(0, 3):
            self.assertEqual(mask[i], False)
        for i in range(4, self.n_possible_colors):
            self.assertEqual(mask[i], True)
        self.assertEqual(mask[self.n_possible_colors], False)

    def test_some_used_some_adj(self):
        mask = self.colorizer._get_pointer_mask(4, [0,2]).squeeze(0)
        self.assertEqual(mask[0], True)
        self.assertEqual(mask[1], False)
        self.assertEqual(mask[2], True)
        self.assertEqual(mask[3], False)
        for i in range(4, self.n_possible_colors):
            self.assertEqual(mask[i], True)
        self.assertEqual(mask[self.n_possible_colors], False)

    def test_all_used(self):
        mask = self.colorizer._get_pointer_mask(self.n_possible_colors, []).squeeze(0)
        for i in range(0, self.n_possible_colors + 1):
            self.assertEqual(mask[i], False)


class TestColoringCheckers(unittest.TestCase):
    def test_is_proper_coloring_true(self):
        self.assertTrue(is_proper_coloring([0, 1, 1, 0, 1], graph2))

    def test_is_proper_coloring_false(self):
        self.assertFalse(is_proper_coloring([0, 0, 1, 0, 1], graph2))
    
    def test_coloring_properties_graph2_false(self):
        is_proper, n_violations, violation_ratio = coloring_properties([0, 0, 1, 0, 1], graph2)
        self.assertFalse(is_proper)
        self.assertEqual(n_violations, 1)
        self.assertAlmostEqual(violation_ratio, 1./3.)

    def test_coloring_properties_graph2_true(self):
        is_proper, n_violations, violation_ratio = coloring_properties([0, 1, 1, 0, 1], graph2)
        self.assertTrue(is_proper)
        self.assertEqual(n_violations, 0)
        self.assertEqual(violation_ratio, 0.)

    def test_coloring_properties_graph1(self):
        is_proper, n_violations, violation_ratio = coloring_properties([0, 1, 1, 1, 0], graph1)
        self.assertFalse(is_proper)
        self.assertEqual(n_violations, 2)
        self.assertAlmostEqual(violation_ratio, 2./7.)


class TestGraphSaveAndLoad(unittest.TestCase):
    def setUp(self):
        self.testIO = StringIO()

    def test_happy_scenario(self):
        petersen_graph.save(self.testIO)
        self.testIO.seek(0)
        loaded_graph = Graph.load(self.testIO)
        self.assertEqual(loaded_graph.n_vertices, petersen_graph.n_vertices)
        self.assertEqual(loaded_graph.adj_list, petersen_graph.adj_list)

    def test_with_isolated_vertices(self):
        graph = Graph([[2], [], [0]])
        graph.save(self.testIO)
        self.testIO.seek(0)
        loaded_graph = Graph.load(self.testIO)
        self.assertEqual(loaded_graph.n_vertices, graph.n_vertices)
        self.assertEqual(loaded_graph.adj_list, graph.adj_list)

class TestEdgeListEliminateSelfLoops(unittest.TestCase):
    def test_consecutive_self_loops(self):
        edge_list = [(0, 1), (1, 1), (3, 3), (2, 0), (2,2)]
        edgelist_eliminate_self_loops(edge_list)
        self.assertEqual(edge_list, [(0, 1), (2, 0)])


class TestFindKColoring(unittest.TestCase):
    def test_happy_scenario_graph1(self):
        coloring = find_k_coloring(graph1, 3)
        self.assertTrue(is_proper_coloring(coloring, graph1))
        n_used = len(set(coloring))
        self.assertEqual(n_used, 3)

    def test_boundary_k(self):
        coloring = find_k_coloring(graph1, 2)
        self.assertEqual(coloring, None)
    
    def test_excess_k_graph1(self):
        coloring = find_k_coloring(graph1, 4)
        self.assertTrue(is_proper_coloring(coloring, graph1))
        n_used = len(set(coloring))
        self.assertLessEqual(n_used, 4)

    def test_happy_scenario_petersen(self):
        coloring = find_k_coloring(petersen_graph, 3)
        self.assertTrue(is_proper_coloring(coloring, petersen_graph))
        n_used = len(set(coloring))
        self.assertEqual(n_used, 3)


class TestLinearScheduler(unittest.TestCase):
    def test_happy_scenario(self):
        scheduler = LinearScheduler(10, 12, 2)
        value1 = scheduler.get_next_value()
        value2 = scheduler.get_next_value()
        self.assertEqual(value1, 11)
        self.assertEqual(value2, 12)

class TestCorrectColoring(unittest.TestCase):
    def test_coloring_with_holes(self):
        coloring = [1, 1, 1, 1, 1] # hole: 0 is not used
        corrected_coloring = correct_coloring(coloring, graph2)
        self.assertTrue(is_proper_coloring(corrected_coloring, graph2))
        
class TestColorizeEmbeddingGuidedSlfMean(unittest.TestCase):
    def test_happy_scenario_1(self):
        embeddings = torch.tensor([[0.4963, 0.7682],
            [0.0885, 0.1320],
            [0.3074, 0.6341],
            [0.4901, 0.8964],
            [0.4556, 0.6323],
            [0.3489, 0.4017],
            [0.0223, 0.1689],
            [0.2939, 0.5185]])

        coloring = colorize_embedding_guided_slf_mean(embeddings, slf_hard)
        self.assertListEqual(coloring, [0, 1, 2, 2, 2, 2, 1, 0])


class TestColorizeEmbeddingGuidedSlfMax(unittest.TestCase):
    def test_happy_scenario_1(self):
        embeddings = torch.tensor([[0.4963, 0.7682],
            [0.0885, 0.1320],
            [0.3074, 0.6341],
            [0.4901, 0.8964],
            [0.4556, 0.6323],
            [0.3489, 0.4017],
            [0.0223, 0.1689],
            [0.2939, 0.5185]])

        coloring = colorize_embedding_guided_slf_max(embeddings, slf_hard)
        self.assertListEqual(coloring, [0, 1, 2, 2, 2, 0, 1, 3])

class TestSimMatrixRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = SimMatrixRegistry.get_instance()
        os.mkdir('/tmp/test')
        self.registry.working_dir_path = '/tmp/test'
        self.dummy_tensor = torch.empty(10, 10).uniform_(0, 2)

    def test_happy_scenario_1(self):
        self.registry.register_similarity_matrix(self.dummy_tensor, 'test_mat')
        new_dummy = self.registry.get_similarity_matrix('test_mat')
        self.assertTrue(torch.all(self.dummy_tensor == new_dummy))

    def test_doesnt_override_existing(self):
        self.registry.register_similarity_matrix(self.dummy_tensor, 'test_mat')
        with self.assertRaises(Exception):
            self.registry.register_similarity_matrix(self.dummy_tensor, 'test_mat')

    def test_non_existing(self):
        self.assertEqual(self.registry.get_similarity_matrix('non_existing_name'), None)

    def tearDown(self):
        shutil.rmtree(self.registry.working_dir_path)


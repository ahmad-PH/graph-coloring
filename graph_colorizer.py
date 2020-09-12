
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from GAT import SGAT
from heuristics import highest_colored_neighbor_heuristic

# first trial: don't mask neighborhood colors
class GraphColorizer(nn.Module):
    def __init__(self, dropout, alpha, nheads):
        super().__init__()
        self.embedding_size = 512
        self.n_possible_colors = 256
        self.sgat = SGAT(self.embedding_size + self.n_possible_colors, self.embedding_size, self.embedding_size, dropout, alpha, nheads) 
        self.color_classifier = ColorClassifier()

    def forward(self, graph: Graph):
        # initializations
        colors = np.array([-1] * graph.n_vertices)
        one_hot_colors = torch.zeros(graph.n_vertices, self.n_possible_colors)
        embeddings = torch.zeros(graph.n_vertices, self.embedding_size)
        n_used_colors = 0
        vertex_order = highest_colored_neighbor_heuristic(graph.adj_list)

        # the first vertex is processed separately:
        self._assign_color(0, vertex_order[0], colors, one_hot_colors)
        for neighbor in graph.adj_list[vertex_order[0]]:
            self._update_vertex_embedding(neighbor, graph, embeddings, one_hot_colors)

        for vertex in vertex_order[1:]: # first vertex was processed before
            self._update_vertex_embedding(vertex, graph, embeddings, one_hot_colors)

            adjacent_colors = self._get_adjacent_colors(vertex, graph, colors)
            classifier_out = self.color_classifier(embeddings[vertex], adjacent_colors, n_used_colors)

            # use output to determine next color (decode it)
            max_color_prob = torch.max(classifier_out)
            max_color_index = torch.argmax(classifier_out).item()
            if max_color_index == self.n_possible_colors:
                chosen_color = n_used_colors
                n_used_colors += 1
            else:
                chosen_color = max_color_index

            self._assign_color(chosen_color, vertex, colors, one_hot_colors)

            for neighbor in graph.adj_list[vertex]:
                self._update_vertex_embedding(neighbor, graph, embeddings, one_hot_colors)

            # compute a loss somehow and backprop it
            

        # print(vertex_order)
        # print(embeddings)

    def _update_vertex_embedding(self, vertex, graph, embeddings, one_hot_colors):
        neighbors = vertex + graph.adj_list[vertex] # because the adj_list is not self-inclusive
        sgat_input = torch.cat([embeddings[neighbors], one_hot_colors[neighbors]], dim=1)
        embeddings[vertex] = self.sgat(sgat_input, 0)  # 0, because we put the vertex itself first in list.

    def _get_adjacent_colors(self, vertex, graph, colors):
        return list(set(colors[graph.adj_list[vertex]]))

    def _assign_color(self, color, vertex, colors, one_hot_colors):
        colors[vertex] = color
        one_hot_colors[vertex] = nn.functional.one_hot(torch.tensor(color), num_classes = self.n_possible_colors)

class ColorClassifier(nn.Module):
    def __init__(self, embedding_size, n_possible_colors, run_assertions = True):
        super().__init__()
        self.embedding_size, self.n_possible_colors = embedding_size, n_possible_colors
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, 400)
        self.fc3 = nn.Linear(400, n_possible_colors + 1) # +1 is for the "new color" case
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)
        if run_assertions:
            assert(self.embedding_size == 512)
            assert(self.n_possible_colors == 256)

    def _mask_irrelevant_colors(self, activations, n_used_colors):
        mask = torch.tensor([0] * n_used_colors + [1] * (self.n_possible_colors - n_used_colors) + [0], \
            dtype=torch.bool, requires_grad=False).to(activations.device)
        mask = mask.repeat(activations.shape[0], 1)
        return activations.masked_fill(mask, float("-inf"))

    def _mask_colors(self, activations, colors):
        mask = torch.tensor([False] * (self.n_possible_colors + 1), dtype=torch.bool, requires_grad=False).to(activations.device)
        mask[colors] = True
        mask = mask.repeat(activations.shape[0], 1)
        return activations.masked_fill(mask, float("-inf"))

    def forward(self, x, n_used_colors, adj_colors = None):
        if x.ndim == 1: 
            no_batch = True
            x = torch.unsqueeze(x, 0)
        else:
            no_batch = False
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = self._mask_irrelevant_colors(x, n_used_colors)
        if adj_colors is not None:
            x = self._mask_colors(x, adj_colors)
        x = self.softmax(x)
        if no_batch: x = x.squeeze(0)
        return x
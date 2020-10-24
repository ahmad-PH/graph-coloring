import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from GAT import GraphAttentionLayer, GraphSingularAttentionLayer
from heuristics import highest_colored_neighbor_heuristic
from typing import *
from matplotlib import pyplot as plt
from utility import *

class GraphColorizer(nn.Module):
    def __init__(self, dropout=0.6, alpha=0.2, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = 512
        self.n_possible_colors = 256
        if loss_type not in ["reinforce"]:
            raise ValueError("unrecognized `loss` value at GraphColorizer: {}".format(loss))
        self.loss_type = loss_type
        self.add_module("attn", 
            GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        )
        self.add_module("attn2", 
            GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        ) # TEST: exists to test whether using non-neighbors helps
        self.add_module("color_classifier", 
            ColorClassifier(self.embedding_size, self.n_possible_colors)
        )
        self.device = device
        self.to(device)

        self.attn_w_norms : List[float] = [] # TEST
        self.attn_a_norms : List[float] = [] # TEST
        self.classif_norms: List[List[float]] = [[], [], []] # TEST


    def forward(self, graph: Graph, baseline=None):
        embeddings = torch.normal(0, 0.1, (graph.n_vertices, self.embedding_size)).to(self.device)
        adj_matrix = torch.tensor(adj_list_to_matrix(graph.adj_list) + np.eye(graph.n_vertices), requires_grad=False).to(self.device) 
        # + np.eye is to make the matrix self-inclusive
        inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix

        for i in range(10):
            embeddings = self.attn.forward(embeddings, adj_matrix)
            embeddings = self.attn2.forward(embeddings, inverted_adj_matrix)

        vertex_order = highest_colored_neighbor_heuristic(graph.adj_list)
        colors = np.array([-1] * graph.n_vertices)
        self._assign_color(0, vertex_order[0], colors)
        n_used_colors = 1
        log_partial_prob = 0.

        for vertex in vertex_order[1:]:
            adjacent_colors = self._get_adjacent_colors(vertex, graph, colors)
            classifier_out = self.color_classifier.forward(embeddings[vertex], n_used_colors, adjacent_colors)

            # use output to determine next color (decode it)
            max_color_index = int(torch.argmax(classifier_out).item())
            if max_color_index == self.n_possible_colors:
                if n_used_colors == self.n_possible_colors:
                    raise RuntimeError('All colors are used, but the network still chose new color.')
                chosen_color = n_used_colors
                n_used_colors += 1
            else:
                chosen_color = max_color_index
            self._assign_color(chosen_color, vertex, colors)

            prob_part = torch.max(classifier_out)
            log_prob_part = torch.log(prob_part + 1e-8) - torch.log(torch.tensor(1e-8)) # TEST # [0, 18.42]
            log_partial_prob += log_prob_part


        if baseline is None: raise ValueError('baseline cannot be None if loss type is `reinforcement`.')
        print('log_partial_prob: {}'.format(log_partial_prob))
        reinforce_loss = (n_used_colors - baseline) * log_partial_prob / graph.n_vertices
        # regularization_term = torch.norm(self.attn.W) + \
        #         torch.norm(self.color_classifier.fc1.weight) + \
        #         torch.norm(self.color_classifier.fc2.weight) + \
        #         torch.norm(self.color_classifier.fc3.weight)
        # print('reinforce loss: {}, regularization term: {}'.format(reinforce_loss, regularization_term))
        print('reinforce loss: {}'.format(reinforce_loss))
        loss = reinforce_loss #+ 0.05 * regularization_term

        return colors, loss

    def _get_adjacent_colors(self, vertex, graph, colors):
        # "-1" is removed from the answer because -1 indicates "no color yet"
        return list(set(colors[graph.adj_list[vertex]]).difference([-1]))

    def _assign_color(self, color, vertex, colors):
        colors[vertex] = color

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
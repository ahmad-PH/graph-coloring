import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from GAT import GraphAttentionLayer, GraphSingularAttentionLayer
from heuristics import slf_heuristic
from typing import *
from matplotlib import pyplot as plt
from utility import *
from globals import data

class GraphColorizer(nn.Module):
    def __init__(self, embedding_size, dropout=0.6, alpha=0.2, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_possible_colors = 20

        self.loss_type = loss_type

        self.attn_neighbors = GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        self.add_module("attn_neighbors", self.attn_neighbors)

        self.attn_non_neighbors = GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        self.add_module("attn_non_neighbors", self.attn_non_neighbors ) # TEST: exists to test whether using non-neighbors helps

        self.color_classifier = ColorClassifier(self.embedding_size, self.n_possible_colors)
        self.add_module("color_classifier", self.color_classifier)

        self.update_embedding = nn.Linear(2 * self.embedding_size, self.embedding_size)
        self.add_module("update_embedding", self.update_embedding)

        self.device = device
        self.to(device)

    def forward(self, graph: Graph, embeddings: torch.Tensor, baseline=None):
        with torch.no_grad():
            adj_matrix = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32, requires_grad=False).to(self.device) 
            inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(self.device)

        # # disabled temporarily to test the simplest possible case
        # for i in range(2): # used to be 10
        #     neighbor_updates = self.attn_neighbors.forward(embeddings, adj_matrix)
        #     non_neighbor_updates = self.attn_non_neighbors.forward(embeddings, inverted_adj_matrix)
        #     concatenated = torch.cat((neighbor_updates, non_neighbor_updates), dim=1)
        #     embeddings = torch.tanh(embeddings + self.update_embedding(concatenated))

        vertex_order = slf_heuristic(graph.adj_list)
        colors = np.array([-1] * graph.n_vertices)
        self._assign_color(0, vertex_order[0], colors)
        n_used_colors = 1
        # log_partial_prob = 0.
        loss = torch.tensor(0.).to(self.device)

        for vertex in vertex_order[1:]:
            adjacent_colors = self._get_adjacent_colors(vertex, graph, colors)

            # print('vertex: {}, adj_colors: {}'.format(vertex, adjacent_colors))
            classifier_out = self.color_classifier.forward(embeddings[vertex], n_used_colors, adjacent_colors)
            # print('classifier outputs: {}'.format(classifier_out.detach().cpu()))

            # use output to determine next color (decode it)
            raw_chosen_color = np.random.choice(self.n_possible_colors , p = classifier_out.detach().cpu().numpy()) # TODO: put back the +1 on n_possible_colors 

            # if raw_chosen_color == self.n_possible_colors:
            #     if n_used_colors == self.n_possible_colors:
            #         raise RuntimeError('All colors are used, but the network still chose new color.')
            #     chosen_color = n_used_colors
            #     n_used_colors += 1
            # else:
            #     chosen_color = raw_chosen_color

            chosen_color = raw_chosen_color
            self._assign_color(chosen_color, vertex, colors)

            loss_part = 1 - torch.log(classifier_out[data.optimal_coloring[vertex]] + 1e-10)
            loss += loss_part

            # print('optimal_sol: {}, probabilities: {}, loss: {} selected: {}'.format(data.optimal_coloring[vertex],
                # classifier_out.data, loss_part, raw_chosen_color))
            
            # prob_part = classifier_out[raw_chosen_color]
            # log_prob_part = torch.log(prob_part + 1e-8) - torch.log(torch.tensor(1e-8)) # TEST # [0, 18.42]
            # log_partial_prob += log_prob_part


        # if baseline is None: raise ValueError('baseline cannot be None if loss type is `reinforcement`.')
        # print('log_partial_prob: {}'.format(log_partial_prob))
        # reinforce_loss = (n_used_colors - baseline) * log_partial_prob / graph.n_vertices
        # print('reinforce loss: {}'.format(reinforce_loss))
        # loss = reinforce_loss #+ 0.05 * regularization_term

        return colors, loss

    def _get_adjacent_colors(self, vertex, graph, colors):
        # "-1" is removed from the answer because -1 indicates "no color yet"
        return list(set(colors[graph.adj_list[vertex]]).difference([-1]))

    def _assign_color(self, color, vertex, colors):
        colors[vertex] = color

class ColorClassifier(nn.Module):
    def __init__(self, embedding_size, n_possible_colors):
        super().__init__()
        self.embedding_size, self.n_possible_colors = embedding_size, n_possible_colors
        self.fc1 = nn.Linear(embedding_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, int((embedding_size + n_possible_colors) / 2)) 
        self.fc3 = nn.Linear(int((embedding_size + n_possible_colors) / 2), n_possible_colors) # TODO: put back the +1 which was for the "new color" case
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

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
        # x = self._mask_irrelevant_colors(x, n_used_colors)
        # if adj_colors is not None:
        #     x = self._mask_colors(x, adj_colors)
        x = self.softmax(x)
        if no_batch: x = x.squeeze(0)
        return x
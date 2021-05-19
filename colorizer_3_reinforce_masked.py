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
from graph_utility import *

class GraphColorizer(nn.Module):
    def __init__(self, embedding_size, dropout=0.6, alpha=0.2, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = embedding_size
        self.n_possible_colors = 50

        self.loss_type = loss_type

        self.color_classifier = ColorClassifier(self.embedding_size, self.n_possible_colors)
        self.add_module("color_classifier", self.color_classifier)

        self.device = device
        self.to(device)

    def forward(self, graph: Graph, embeddings: torch.Tensor, baseline=None):
        with torch.no_grad():
            adj_matrix = torch.tensor(graph.get_adj_matrix(), dtype=torch.float32, requires_grad=False).to(self.device) 
            inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(self.device)

        vertex_order = slf_heuristic(graph.adj_list)
        colors = np.array([-1] * graph.n_vertices)
        self._assign_color(0, vertex_order[0], colors)
        n_used_colors = 1
        partial_log_prob = torch.tensor(0.).to(self.device)
        extra = DataDump()

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

            # print('optimal_sol: {}, probabilities: {}, loss: {} selected: {}'.format(data.optimal_coloring[vertex],
            #     classifier_out.data, loss_part, raw_chosen_color))
            
            # print('optimal sol: {}, selected: {}'.format(data.optimal_coloring[vertex], raw_chosen_color))
            
            prob = classifier_out[raw_chosen_color]
            log_prob = torch.log(prob + 1e-8) - torch.log(torch.tensor(1e-8)) # TEST # [0, 18.42]
            partial_log_prob += log_prob


        # if baseline is None: raise ValueError('baseline cannot be None if loss type is `reinforcement`.')
        # print('log_partial_prob: {}'.format(log_partial_prob))
        # reinforce_loss = (n_used_colors - baseline) * log_partial_prob / graph.n_vertices

        confidence = partial_log_prob / graph.n_vertices
        _, _, violation_ratio = coloring_properties(colors, graph)
        violation_percent = violation_ratio * 100.
        n_used_colors = len(set(colors))
        # n_used_colors_ratio = n_used_colors / graph.n_vertices
        # _lambda = data.n_used_lambda_scheduler.get_next_value()
        _lambda = 1.
        cost = (n_used_colors * _lambda + violation_percent)
        print('lambda: ', _lambda)

        reinforce_loss = cost * confidence
        print('violation_percent: {}, n_used_colors: {}, cost: {}, confidence: {}, loss: {}'.format(
            violation_percent, n_used_colors, cost, confidence.item(), reinforce_loss.item()
        ))
        
        extra.cost = cost
        extra.violation_ratio = violation_ratio
        return colors, reinforce_loss, extra

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
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, int((embedding_size + n_possible_colors) / 2)) 
        self.fc4 = nn.Linear(int((embedding_size + n_possible_colors) / 2), n_possible_colors) # TODO: put back the +1 which was for the "new color" case
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def _mask_irrelevant_colors(self, activations, n_used_colors):
        mask = torch.tensor([0] * n_used_colors + [1] * (self.n_possible_colors - n_used_colors) + [0], \
            dtype=torch.bool, requires_grad=False).to(activations.device)
        mask = mask.repeat(activations.shape[0], 1)
        return activations.masked_fill(mask, float("-inf"))

    def _mask_colors(self, activations, colors):
        mask = torch.tensor([False] * (self.n_possible_colors), dtype=torch.bool, requires_grad=False).to(activations.device)
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
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        # x = self._mask_irrelevant_colors(x, n_used_colors)
        if adj_colors is not None:
            x = self._mask_colors(x, adj_colors)
        x = self.softmax(x)
        if no_batch: x = x.squeeze(0)
        return x
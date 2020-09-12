import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from GAT import GraphSingularAttentionLayer
from heuristics import highest_colored_neighbor_heuristic

class GraphColorizer(nn.Module):
    def __init__(self, dropout=0.6, alpha=0.2, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = 512
        self.n_possible_colors = 256
        if loss_type not in ["reinforce", "direct"]:
            raise ValueError("unrecognized `loss` value at GraphColorizer: {}".format(loss))
        self.loss_type = loss_type
        self.add_module("attn", 
            GraphSingularAttentionLayer(self.embedding_size + self.n_possible_colors, self.embedding_size, dropout, alpha) 
        )
        # self.sgat = SGAT(self.embedding_size + self.n_possible_colors, self.embedding_size, self.embedding_size, dropout, alpha, nheads) 
        self.add_module("color_classifier", 
            ColorClassifier(self.embedding_size, self.n_possible_colors)
        )
        self.device = device
        self.to(device)

    def forward(self, graph: Graph, baseline=None):
        # initializations
        colors = np.array([-1] * graph.n_vertices)
        one_hot_colors = torch.zeros(graph.n_vertices, self.n_possible_colors).to(self.device)
        embeddings = torch.zeros(graph.n_vertices, self.embedding_size).to(self.device)
        n_used_colors = 0
        vertex_order = highest_colored_neighbor_heuristic(graph.adj_list)
        if self.loss_type == "direct":
            loss = torch.tensor(0.).to(self.device)
        elif self.loss_type == "reinforce":
            partial_prob = torch.tensor(1.).to(self.device)

        # the first vertex is processed separately:
        self._assign_color(0, vertex_order[0], colors, one_hot_colors)
        n_used_colors = 1
        for neighbor in graph.adj_list[vertex_order[0]]:
            self._update_vertex_embedding(neighbor, graph, embeddings, one_hot_colors)

        # print('vertex order:', vertex_order)
        # print('after initial assign:')
        # print(colors)
        # print(one_hot_colors)
        # print(embeddings)
        # print(n_used_colors)

        for vertex in vertex_order[1:]: # first vertex was processed before
            # print('vertex {}'.format(vertex))

            self._update_vertex_embedding(vertex, graph, embeddings, one_hot_colors)
            if self.loss_type == "direct":
                adjacent_colors = None
            elif self.loss_type == "reinforce":
                adjacent_colors = self._get_adjacent_colors(vertex, graph, colors)
            classifier_out = self.color_classifier(embeddings[vertex].clone(), n_used_colors, adjacent_colors)

            # use output to determine next color (decode it)
            max_color_index = int(torch.argmax(classifier_out).item())
            if max_color_index == self.n_possible_colors:
                chosen_color = n_used_colors
                n_used_colors += 1
            else:
                chosen_color = max_color_index
            self._assign_color(chosen_color, vertex, colors, one_hot_colors)

            # print('classifier out:', classifier_out[[0,1,2,3,self.n_possible_colors]].data)
            if self.loss_type == "direct":
                loss_p = self._compute_color_classifier_loss(classifier_out, adjacent_colors)
                loss += loss_p
            elif self.loss_type == "reinforce":
                partial_prob *= torch.max(classifier_out)

            for neighbor in graph.adj_list[vertex]:
                self._update_vertex_embedding(neighbor, graph, embeddings, one_hot_colors)

        if self.loss_type == "reinforce":
            if baseline is None: raise ValueError('baseline can not be None if loss type is `reinforcement`.')
            print('n_used: {}, partial_prob: {}'.format(n_used_colors, partial_prob))
            loss = (n_used_colors - baseline) * partial_prob            

        return colors, loss

    def _update_vertex_embedding(self, vertex, graph, embeddings, one_hot_colors):
        cloned_neighbor_embeddings = torch.cat([embeddings[vertex].unsqueeze(0).clone(), embeddings[graph.adj_list[vertex]]], dim=0)
        cloned_neighbor_one_hot_colors = torch.cat([one_hot_colors[vertex].unsqueeze(0).clone(), one_hot_colors[graph.adj_list[vertex]]], dim=0)
        attn_input = torch.cat([cloned_neighbor_embeddings, cloned_neighbor_one_hot_colors], dim=1)
        embeddings[vertex] = self.attn(attn_input, 0)  # 0, because we put the vertex itself first in list.

    def _compute_color_classifier_loss(self, classifier_out, adj_colors, epsilon = 1e-15):
        max_color_prob = torch.max(classifier_out)
        max_color_index = int(torch.argmax(classifier_out).item())
        if max_color_index in adj_colors:
            return 5. * - torch.log(1. - max_color_prob + epsilon)
        elif max_color_index == self.n_possible_colors:
            return - torch.log(1. - max_color_prob + epsilon)
        else:
            return 0.

    def _get_adjacent_colors(self, vertex, graph, colors):
        # "-1" is removed from the answer because -1 indicates "no color yet"
        return list(set(colors[graph.adj_list[vertex]]).difference([-1]))

    def _assign_color(self, color, vertex, colors, one_hot_colors):
        colors[vertex] = color
        one_hot_colors[vertex] = F.one_hot(torch.tensor(color), num_classes = self.n_possible_colors)

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
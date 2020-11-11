import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from GAT import GraphAttentionLayer, GraphSingularAttentionLayer
from multi_head_attention import MHAAdapter, MultiHeadAttention
from heuristics import slf_heuristic
from typing import *
from matplotlib import pyplot as plt
from utility import *

class GraphColorizer(nn.Module):
    def __init__(self, dropout=0.6, alpha=0.2, graph: Graph = None, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = 512 
        self.n_possible_colors = 256 
        self.loss_type = loss_type

        if loss_type not in ["reinforce"]:
            raise ValueError("unrecognized `loss` value at GraphColorizer: {}".format(loss))

        self.n_attn_layers = 3 # TEST: used to be 10
        self.neighb_attns = nn.ModuleList([MHAAdapter(8, self.embedding_size, self.embedding_size) \
            for _ in range(self.n_attn_layers)])
        self.non_neighb_attns = nn.ModuleList([MHAAdapter(8, self.embedding_size, self.embedding_size) \
            for _ in range(self.n_attn_layers)])
        # self.attn_combiner = nn.ModuleList([nn.Linear(2*self.embedding_size, self.embedding_size) \
            # for _ in range(self.n_attn_layers)])

        self.attn_combiner = nn.ModuleList([nn.Sequential(
                nn.Linear(2*self.embedding_size, 2*self.embedding_size),
                nn.ReLU(),
                nn.Linear(2*self.embedding_size, self.embedding_size)
            ) for _ in range(self.n_attn_layers)
        ])

        self.color_classifier = ColorClassifierWithGraphEmbedding(self.embedding_size, self.n_possible_colors)
        
        # self.embeddings = nn.Parameter(torch.normal(0, 1., (graph.n_vertices, self.embedding_size)))

        self.device = device
        self.to(device)

    def forward(self, graph: Graph, baseline=None):
        embeddings = torch.normal(0, 1., (graph.n_vertices, self.embedding_size)).to(self.device)
        # embeddings = self.embeddings

        with torch.no_grad():
            adj_matrix = torch.tensor(adj_list_to_matrix(graph.adj_list), requires_grad=False).to(self.device) 
            inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(self.device)

        for i in range(self.n_attn_layers):
            # neighbor_updates = F.relu(self.neighb_attns[i].forward(embeddings, adj_matrix))
            # non_neighbor_updates = F.relu(self.non_neighb_attns[i].forward(embeddings, inverted_adj_matrix))
            # concatenated = torch.cat((neighbor_updates, non_neighbor_updates), dim=1)
            # embeddings = embeddings + self.attn_combiner[i].forward(concatenated) 
            neighbor_updates = self.neighb_attns[i].forward(embeddings, adj_matrix) + embeddings
            non_neighbor_updates = self.non_neighb_attns[i].forward(embeddings, inverted_adj_matrix) + embeddings
            concatenated = torch.cat((neighbor_updates, non_neighbor_updates), dim=1)
            embeddings = embeddings + self.attn_combiner[i].forward(concatenated)


        vertex_order = slf_heuristic(graph.adj_list)
        colors = np.array([-1] * graph.n_vertices)
        self._assign_color(0, vertex_order[0], colors)
        n_used_colors = 1
        log_partial_prob = 0.
        node_confidences : List[torch.Tensor]= []

        for vertex in vertex_order[1:]:
            adjacent_colors = self._get_adjacent_colors(vertex, graph, colors)
            classifier_out = self.color_classifier.forward(embeddings[vertex], torch.mean(embeddings, dim=0), n_used_colors, adjacent_colors)
            # classifier_out = self.color_classifier.forward(embeddings[vertex], n_used_colors, adjacent_colors)

            # use output to determine next color (decode it)
            raw_chosen_color = np.random.choice(self.n_possible_colors + 1, p = classifier_out.detach().cpu().numpy())
            if raw_chosen_color == self.n_possible_colors:
                if n_used_colors == self.n_possible_colors:
                    raise RuntimeError('All colors are used, but the network still chose new color.')
                chosen_color = n_used_colors
                n_used_colors += 1
            else:
                chosen_color = raw_chosen_color
            self._assign_color(chosen_color, vertex, colors)

            prob_part = classifier_out[raw_chosen_color]
            log_prob_part = torch.log(prob_part + 1e-8) - torch.log(torch.tensor(1e-8)) # TEST # [0, 18.42]
            log_partial_prob += log_prob_part

        if baseline is None: raise ValueError('baseline cannot be None if loss type is `reinforcement`.')
        # print('log_partial_prob: {}'.format(log_partial_prob))
        reinforce_loss = (n_used_colors - baseline) * log_partial_prob / graph.n_vertices
        loss = reinforce_loss #+ 0.01 * (neighborhood_loss + compactness_loss)
        # loss = neighborhood_loss + 0.01 * compactness_loss
        print('loss: {}'.format(loss))

        return colors, loss

    def _get_adjacent_colors(self, vertex, graph, colors):
        # "-1" is removed from the answer because -1 indicates "no color yet"
        return list(set(colors[graph.adj_list[vertex]]).difference([-1]))

    def _assign_color(self, color, vertex, colors):
        colors[vertex] = color

class ColorClassifierWithGraphEmbedding(nn.Module):
    def __init__(self, embedding_size, n_possible_colors, run_assertions = True):
        super().__init__()
        self.embedding_size, self.n_possible_colors = embedding_size, n_possible_colors
        self.fc1 = nn.Linear(embedding_size * 2, embedding_size)
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

    def forward(self, node_embedding, graph_embedding, n_used_colors, adj_colors = None):
        if node_embedding.ndim == 1:
            no_batch = True
            node_embedding = torch.unsqueeze(node_embedding, 0)
        else:
            no_batch = False
        graph_embedding = graph_embedding.unsqueeze(0).expand(node_embedding.shape[0], -1)
        x = torch.cat([node_embedding, graph_embedding], dim=1)
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = self._mask_irrelevant_colors(x, n_used_colors)
        if adj_colors is not None:
            x = self._mask_colors(x, adj_colors)
        x = self.softmax(x)
        if no_batch: x = x.squeeze(0)
        return x


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
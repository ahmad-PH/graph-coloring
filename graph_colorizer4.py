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

class GraphColorizer(nn.Module):
    def __init__(self, dropout=0.6, alpha=0.2, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = 512 
        self.n_possible_colors = 256 
        if loss_type not in ["reinforce"]:
            raise ValueError("unrecognized `loss` value at GraphColorizer: {}".format(loss))
        self.loss_type = loss_type
        self.add_module("attn_neighbors", 
            GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        )
        self.add_module("attn_non_neighbors", 
            GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        ) # TEST: exists to test whether using non-neighbors helps
        self.add_module("update_embedding", # TEST
            nn.Linear(2 * self.embedding_size, self.embedding_size)
        )
        self.glimpse = GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True, True)
        self.device = device
        self.to(device)

        self.attn_w_norms : List[float] = [] # TEST
        self.attn_a_norms : List[float] = [] # TEST
        self.classif_norms: List[List[float]] = [[], [], []] # TEST

    def forward(self, graph: Graph, baseline=None):
        embeddings = torch.normal(0, 0.1, (graph.n_vertices, self.embedding_size)).to(self.device)
        with torch.no_grad():
            adj_matrix = torch.tensor(adj_list_to_matrix(graph.adj_list), requires_grad=False).to(self.device) 
            inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(self.device)

        for i in range(10):
            neighbor_updates = self.attn_neighbors.forward(embeddings, adj_matrix)
            non_neighbor_updates = self.attn_non_neighbors.forward(embeddings, inverted_adj_matrix)
            concatenated = torch.cat((neighbor_updates, non_neighbor_updates), dim=1)
            embeddings = F.relu(embeddings + self.update_embedding(concatenated))

        N = embeddings.size()[0]

        embeddings_repeated_in_chunks = embeddings.repeat_interleave(N, dim=0)
        embeddings_repeated_alternating = embeddings.repeat(N, 1)
        distances = torch.norm(embeddings_repeated_in_chunks - embeddings_repeated_alternating, dim=1)
        all_distances = distances.view(N, N)

        with torch.no_grad():
            # similarity_matrix = (2 * -adj_matrix + 1) - torch.eye(graph.n_vertices).to(self.device)
            similarity_matrix = -adj_matrix
            similarity_matrix = similarity_matrix.float()

        embedding_loss = torch.sum(all_distances * similarity_matrix)
        alpha = 400.
        triplet_loss = torch.max(embedding_loss + alpha, torch.tensor(0.).to(self.device))

        print('embedding_loss: {}, triplet_loss: {}'.format(embedding_loss, triplet_loss))

        glimpse_attentions = self.glimpse.forward(embeddings, adj_matrix)
        glimpse_links = torch.argmax(glimpse_attentions, dim=1)
        adj_list = self._create_adj_list_from_glimpse_links(glimpse_links.cpu().numpy(), adj_matrix)

        # print('glimpse attentions:', glimpse_attentions)
        # print('glimpse links:', glimpse_links)
        # print('adj list:', adj_list)

        coloring = self._colorize_cluster_graph(adj_list)
        n_used_colors = len(set(coloring))
        print('coloring', coloring)

        confidence_vector, _ = torch.max(glimpse_attentions, dim=1)
        log_confidence_vector = torch.log(confidence_vector + 1e-8) - torch.log(torch.full_like(confidence_vector, 1e-8))
        log_confidence = torch.mean(log_confidence_vector)

        if baseline is None: raise ValueError('baseline cannot be None if loss type is `reinforcement`.')
        reinforce_loss = (n_used_colors - baseline) * log_confidence
        loss = reinforce_loss #+ 0.1 * triplet_loss #+ 0.05 * regularization_term
        print('loss: {}, reinforce loss: {}, triplet loss: {}'.format(loss, reinforce_loss, triplet_loss))

        return coloring, loss

    def _get_adjacent_colors(self, vertex, graph, colors):
        # "-1" is removed from the answer because -1 indicates "no color yet"
        return list(set(colors[graph.adj_list[vertex]]).difference([-1]))

    def _assign_color(self, color, vertex, colors):
        colors[vertex] = color

    def _create_adj_list_from_glimpse_links(self, links, adj_matrix):
        n_vertices = len(links)
        skip_vertex = [False] * n_vertices 
        adj_list = [[] for _ in range(n_vertices)]
        for v in range(n_vertices):
            u = links[v]
            if skip_vertex[v] or u == v or adj_matrix[u,v] == 1:
                continue
            adj_list[v].append(u)
            adj_list[u].append(v)
            if links[u] == v:
                skip_vertex[u] = True
        return adj_list

    def _colorize_cluster_graph(self, adj_list):
        n_vertices = len(adj_list)
        colors = [-1] * n_vertices
        next_color = 0
        for v in range(n_vertices):
            if colors[v] != -1: # already colored
                continue
            self._colorize_cluster_graph_rec(adj_list, v, colors, next_color)
            next_color += 1
        return colors

    def _colorize_cluster_graph_rec(self, adj_list, vertex, colors, color):
        if colors[vertex] != -1: # already colored
            return
        colors[vertex] = color
        for v in adj_list[vertex]:
            self._colorize_cluster_graph_rec(adj_list, v, colors, color)



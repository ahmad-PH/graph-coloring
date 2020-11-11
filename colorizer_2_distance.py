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
    def __init__(self, dropout=0.6, alpha=0.2, graph: Graph = None, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = 512 
        self.n_possible_colors = 256 
        self.loss_type = loss_type

        if loss_type not in ["reinforce"]:
            raise ValueError("unrecognized `loss` value at GraphColorizer: {}".format(loss))

        if graph == None:
            raise ValueError("graph cannot be None in constructor.")

        # self.add_module("attn_neighbors", 
        #     GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        # )
        # self.add_module("attn_non_neighbors", 
        #     GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True)
        # ) # TEST: exists to test whether using non-neighbors helps

        # self.add_module("update_embedding", # TEST
        #     nn.Linear(2 * self.embedding_size, self.embedding_size)
        # )

        self.n_attn_layers = 3 # TEST: used to be 10
        self.neighb_attns = nn.ModuleList([GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True) \
            for _ in range(self.n_attn_layers)])
        self.non_neighb_attns = nn.ModuleList([GraphAttentionLayer(self.embedding_size, self.embedding_size, dropout, alpha, True) \
            for _ in range(self.n_attn_layers)])
        self.attn_combiner = nn.ModuleList([nn.Linear(2*self.embedding_size, self.embedding_size) \
            for _ in range(self.n_attn_layers)])

        self.color_classifier = ColorClassifier(self.embedding_size, self.n_possible_colors)
        
        self.embeddings = nn.Parameter(torch.normal(0, 1., (graph.n_vertices, self.embedding_size)))

        self.device = device
        self.to(device)

        self.attn_w_norms : List[float] = [] # TEST
        self.attn_a_norms : List[float] = [] # TEST
        self.classif_norms: List[List[float]] = [[], [], []] # TEST

    def forward(self, graph: Graph, baseline=None):
        # embeddings = torch.normal(0, 0.1, (graph.n_vertices, self.embedding_size)).to(self.device)

        # embeddings = torch.empty(graph.n_vertices, self.embedding_size).uniform_(-1,1)
        # torch.save(embeddings, 'a.pt')
        # import sys; sys.exit()

        # embeddings = torch.load('a.pt')

        embeddings = self.embeddings

        with torch.no_grad():
            adj_matrix = torch.tensor(adj_list_to_matrix(graph.adj_list), requires_grad=False).to(self.device) 
            inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(self.device)

        N = embeddings.size()[0] 

        embeddings_repeated_in_chunks = embeddings.repeat_interleave(N, dim=0)
        embeddings_repeated_alternating = embeddings.repeat(N, 1)
        distances = torch.norm(embeddings_repeated_in_chunks - embeddings_repeated_alternating, dim=1)
        all_distances = distances.view(N, N)

        # classifier_outputs = self.color_classifier.forward(embeddings, self.n_possible_colors, [])
        # embeddings_repeated_in_chunks = classifier_outputs.repeat_interleave(N, dim=0)
        # embeddings_repeated_alternating = classifier_outputs.repeat(N, 1)
        # distances = torch.norm(embeddings_repeated_in_chunks - embeddings_repeated_alternating, dim=1)
        # all_distances = distances.view(N, N)

        # with torch.no_grad():
        #     # similarity_matrix = (2 * -adj_matrix + 1) - torch.eye(graph.n_vertices).to(self.device)
        #     similarity_matrix = -adj_matrix
        #     similarity_matrix = similarity_matrix.float()

        # embedding_loss = torch.sum(all_distances * similarity_matrix)


        neighborhood_loss = - torch.sum(all_distances * adj_matrix.float() / torch.sum(adj_matrix, dim=1, keepdim=True))

        center = torch.mean(embeddings, dim=0)
        center_repeated = center.unsqueeze(0).expand(N, -1)

        print('\nall_distances:')
        print(all_distances)
        print('adj matrix:')
        print(adj_matrix)
        print('\n')

        compactness_loss = torch.sum(torch.norm(embeddings - center_repeated, dim=1))

        with torch.no_grad():
            # neighborhood_dist = - neighborhood_loss.item()
            # non_neighborhood_dist = torch.sum(all_distances * inverted_adj_matrix.float() / torch.sum(adj_matrix, dim=1, keepdim=True)).item()
            neighborhood_dist = - neighborhood_loss.item()
            cent_dist = compactness_loss.item()

            print('neighbs_dist: {}, cent_dist: {}, ratio: {}'.format(neighborhood_dist, cent_dist, neighborhood_dist/cent_dist))

            import globals 
            globals.data.last_ratio = neighborhood_dist/cent_dist
            # from globals import file_writer
            # print('WRITING TO FILE:', file_writer)
            # print(neighborhood_dist / non_neighborhood_dist, file=file_writer)

        # alpha = 400.
        # triplet_loss = torch.max(embedding_loss + alpha, torch.tensor(0.).to(self.device))

        # print('embedding_loss: {}, triplet_loss: {}'.format(embedding_loss, triplet_loss))

        # for i in range(self.n_attn_layers):
        #     neighbor_updates = F.relu(self.neighb_attns[i].forward(embeddings, adj_matrix))
        #     non_neighbor_updates = F.relu(self.non_neighb_attns[i].forward(embeddings, inverted_adj_matrix))
        #     concatenated = torch.cat((neighbor_updates, non_neighbor_updates), dim=1)
        #     embeddings = F.relu(embeddings + self.attn_combiner[i].forward(concatenated))

        # vertex_order = slf_heuristic(graph.adj_list)
        # colors = np.array([-1] * graph.n_vertices)
        # self._assign_color(0, vertex_order[0], colors)
        # n_used_colors = 1
        # log_partial_prob = 0.
        # node_confidences : List[torch.Tensor]= []

        # for vertex in vertex_order[1:]:
        #     adjacent_colors = self._get_adjacent_colors(vertex, graph, colors)
        #     classifier_out = self.color_classifier.forward(embeddings[vertex], n_used_colors, adjacent_colors)

        #     # use output to determine next color (decode it)
        #     max_color_index = int(torch.argmax(classifier_out).item())
        #     if max_color_index == self.n_possible_colors:
        #         if n_used_colors == self.n_possible_colors:
        #             raise RuntimeError('All colors are used, but the network still chose new color.')
        #         chosen_color = n_used_colors
        #         n_used_colors += 1
        #     else:
        #         chosen_color = max_color_index
        #     self._assign_color(chosen_color, vertex, colors)

        #     prob_part = torch.max(classifier_out)
        #     log_prob_part = torch.log(prob_part + 1e-8) - torch.log(torch.tensor(1e-8)) # TEST # [0, 18.42]
        #     log_partial_prob += log_prob_part

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters = 2).fit(embeddings.detach().cpu())
        print('labels:', kmeans.labels_)

        if baseline is None: raise ValueError('baseline cannot be None if loss type is `reinforcement`.')
        # print('log_partial_prob: {}'.format(log_partial_prob))
        # reinforce_loss = (n_used_colors - baseline) * log_partial_prob / graph.n_vertices
        # loss = reinforce_loss + 0.01 * (neighborhood_loss + compactness_loss)
        loss = neighborhood_loss + 0.01 * compactness_loss
        print('neighb loss: {}, compactness loss: {}, reinforce loss: {} loss: {}'.format(
            neighborhood_loss, compactness_loss, None, loss))

        return kmeans.labels_, loss

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
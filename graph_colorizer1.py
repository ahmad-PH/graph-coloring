import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from graph import Graph
from GAT import GraphSingularAttentionLayer
from heuristics import highest_colored_neighbor_heuristic
from typing import *
from matplotlib import pyplot as plt

class GraphColorizer(nn.Module):
    def __init__(self, dropout=0.6, alpha=0.2, device="cpu", loss_type="reinforce"):
        super().__init__()
        self.embedding_size = 512
        self.n_possible_colors = 256
        if loss_type not in ["reinforce", "direct"]:
            raise ValueError("unrecognized `loss` value at GraphColorizer: {}".format(loss))
        self.loss_type = loss_type
        # print('memory before attn: {}'.format(torch.cuda.memory_allocated(0)))
        self.add_module("attn", 
            GraphSingularAttentionLayer(self.embedding_size + self.n_possible_colors, self.embedding_size, dropout, alpha).to('cuda:0') # TEST
        )
        # print('memory before classifier: {}'.format(torch.cuda.memory_allocated(0)))
        self.add_module("color_classifier", 
            ColorClassifier(self.embedding_size, self.n_possible_colors).to('cuda:0') # TEST
        )
        # print('memory after classifier: {}'.format(torch.cuda.memory_allocated(0)))
        self.device = device
        self.to(device)

        self.attn_w_norms : List[float] = [] # TEST
        self.attn_a_norms : List[float] = [] # TEST
        self.classif_norms: List[List[float]] = [[], [], []] # TEST


    def forward(self, graph: Graph, baseline=None):
        # initializations
        # print('initial memory: ', torch.cuda.memory_allocated(0))
        colors = np.array([-1] * graph.n_vertices)
        one_hot_colors = torch.zeros(graph.n_vertices, self.n_possible_colors).to(self.device)
        # print('memory after one-hot colors: ', torch.cuda.memory_allocated(0))
        embeddings = torch.zeros(graph.n_vertices, self.embedding_size).to(self.device)
        # print('memory after initializations: ', torch.cuda.memory_allocated(0))
        n_used_colors = 0
        vertex_order = highest_colored_neighbor_heuristic(graph.adj_list)
        if self.loss_type == "direct":
            loss = torch.tensor(0.).to(self.device)
        elif self.loss_type == "reinforce":
            partial_prob = torch.tensor(1.).to(self.device)
            log_partial_prob = 0. # TEST

        # the first vertex is processed separately:
        self._assign_color(0, vertex_order[0], colors, one_hot_colors)
        n_used_colors = 1
        # TEMPORARY REMOVAL DUE TO MEM CONSTRAINTS:
        # for neighbor in graph.adj_list[vertex_order[0]]:
        #     self._update_vertex_embedding(neighbor, graph, embeddings, one_hot_colors)

        # print('vertex order:', vertex_order)
        # print('after initial assign:')
        # print(colors)
        # print(one_hot_colors)
        # print(embeddings)
        # print(n_used_colors)
        hist = [] # TEST
        classif_inp_norms: List[float] = [] # TEST

        i = 0 # TEST
        for vertex in vertex_order[1:]: # first vertex was processed before
            i += 1 # TEST
            # print('vertex {}'.format(vertex))
            # print('memory: {}'.format(torch.cuda.memory_allocated(0)))

            self._update_vertex_embedding(vertex, graph, embeddings, one_hot_colors)
            # print('after emb update: {}'.format(torch.cuda.memory_allocated(0)))

            if self.loss_type == "direct":
                adjacent_colors = None
            elif self.loss_type == "reinforce":
                adjacent_colors = self._get_adjacent_colors(vertex, graph, colors)
            classifier_out = self.color_classifier.forward(embeddings[vertex].clone(), n_used_colors, adjacent_colors)
            classif_inp_norms.append(torch.norm(embeddings[vertex]).item())
            # print('after classifier fw: {}'.format(torch.cuda.memory_allocated(0)))
            # print('classifier out:', classifier_out[[0,1,2,3,self.n_possible_colors]].data)
            if torch.any(torch.isnan(classifier_out)):
                print('nan detected')
                print('classifier input: {}, {}, {}'.format(embeddings[vertex], n_used_colors, adjacent_colors))
                print('classif inp norms:', classif_inp_norms)
                plt.title('classifier input norms')
                plt.plot(classif_inp_norms)
                plt.show()
            #     plt.title('w norms')
            #     plt.plot(self.attn.W_norms)
            #     plt.show()
                # import sys; sys.exit(0)


            # use output to determine next color (decode it)
            max_color_index = int(torch.argmax(classifier_out).item())
            if max_color_index == self.n_possible_colors:
                if n_used_colors == self.n_possible_colors:
                    raise RuntimeError('All colors are used, but the network still chose new color.')
                chosen_color = n_used_colors
                n_used_colors += 1
            else:
                chosen_color = max_color_index
            self._assign_color(chosen_color, vertex, colors, one_hot_colors)

            hist.append(torch.max(classifier_out).item())

            if self.loss_type == "reinforce":
                prob_part = torch.max(classifier_out)
                log_prob_part = torch.log(prob_part + 1e-8) - torch.log(torch.tensor(1e-8)) # TEST # [0, 18.42]
                hist.append(log_prob_part.item()) # TEST
                # partial_prob *= prob_part
                log_partial_prob += log_prob_part

            elif self.loss_type == "direct":
                loss_p = self._compute_color_classifier_loss(classifier_out, adjacent_colors)
                loss += loss_p
            
            # TEMPORARY REMOVAL DUE TO MEM CONSTRAINTS:
            # for neighbor in graph.adj_list[vertex]:
                # self._update_vertex_embedding(neighbor, graph, embeddings, one_hot_colors)

        self.attn_w_norms.append(torch.norm(self.attn.W).item()) # TEST
        self.attn_a_norms.append(torch.norm(self.attn.a).item()) # TEST
        self.classif_norms[0].append(torch.norm(self.color_classifier.fc1.weight).item()) # TEST 
        self.classif_norms[1].append(torch.norm(self.color_classifier.fc2.weight).item()) # TEST 
        self.classif_norms[2].append(torch.norm(self.color_classifier.fc3.weight).item()) # TEST

        if self.loss_type == "reinforce":
            if baseline is None: raise ValueError('baseline can not be None if loss type is `reinforcement`.')
            print('log_partial_prob: {}'.format(log_partial_prob))
            # loss = (n_used_colors - baseline) * partial_prob #- 100.*torch.log(partial_prob + 1e-8)   # TEST     
            # print('log input:', n_used_colors - baseline + 1e-8)
            # loss = np.log(n_used_colors - baseline + 1e-8) + log_partial_prob # TEST
            reinforce_loss = (n_used_colors - baseline) * log_partial_prob / graph.n_vertices
            regularization_term = torch.norm(self.attn.W) + \
                 torch.norm(self.color_classifier.fc1.weight) + \
                 torch.norm(self.color_classifier.fc2.weight) + \
                 torch.norm(self.color_classifier.fc3.weight)
            print('reinforce loss: {}, regularization term: {}'.format(reinforce_loss, regularization_term))
            loss = reinforce_loss + 0.05 * regularization_term

        # print('hist:', hist)
        # print('hist multiplied:', torch.prod(torch.tensor(hist)))
        # plt.hist(hist, bins=10)
        # import sys
        # sys.exit()

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
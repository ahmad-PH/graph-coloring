from graph import Graph
from test import graph1
import torch
import torch.nn as nn
from utility import adj_list_to_matrix, generate_queens_graph
from matplotlib import pyplot as plt
from test import *
import numpy as np


def learn_embeddings(graph, embedding_dim = 2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N = graph.n_vertices
    embeddings = nn.Parameter(torch.normal(0, 1., (N, embedding_dim), device=device))

    with torch.no_grad():
        adj_matrix = torch.tensor(adj_list_to_matrix(graph.adj_list)).to(device) 
        inverted_adj_matrix = torch.ones_like(adj_matrix) - adj_matrix - torch.eye(graph.n_vertices).to(device)

    optimizer = torch.optim.Adam([embeddings], lr=0.1)

    for i in range(400):
        print('epoch: {}'.format(i))

        optimizer.zero_grad()

        embeddings_repeated_in_chunks = embeddings.repeat_interleave(N, dim=0)
        embeddings_repeated_alternating = embeddings.repeat(N, 1)
        distances = torch.norm(embeddings_repeated_in_chunks - embeddings_repeated_alternating, dim=1)
        all_distances = distances.view(N, N)

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
        # print('adj matrix:')
        # print(adj_matrix)
        print('\n')

        compactness_loss = torch.sum(torch.norm(embeddings - center_repeated, dim=1) ** 2)

        # with torch.no_grad():
        #     # neighborhood_dist = - neighborhood_loss.item()
        #     # non_neighborhood_dist = torch.sum(all_distances * inverted_adj_matrix.float() / torch.sum(adj_matrix, dim=1, keepdim=True)).item()
        #     neighborhood_dist = - neighborhood_loss.item()
        #     cent_dist = compactness_loss.item()

        #     print('neighbs_dist: {}, cent_dist: {}, ratio: {}'.format(neighborhood_dist, cent_dist, neighborhood_dist/cent_dist))

        #     import globals 
        #     globals.data.last_ratio = neighborhood_dist/cent_dist

        # alpha = 400.
        # triplet_loss = torch.max(embedding_loss + alpha, torch.tensor(0.).to(self.device))

        # print('embedding_loss: {}, triplet_loss: {}'.format(embedding_loss, triplet_loss))

        loss = neighborhood_loss + 0.1 * compactness_loss
        print('neighb loss: {}, compactness loss: {}, loss: {}'.format(
            neighborhood_loss, compactness_loss, loss))

        loss.backward()
        optimizer.step()

        print('gradient:', embeddings.grad)

    return embeddings

emb_test_graph1 = [
    [1,2],
    [0],
    [0]
]

emb_test_graph2 = [
    [2,3],
    [2,3],
    [0,1],
    [0,1]
]


# graph = Graph(bipartite_10_vertices)
graph = generate_queens_graph(13, 13)

# seed = np.random.randint(0, 1000000)
# seed = 348266 # peterson 1: 
seed = 266412
torch.random.manual_seed(seed)

embeddings = learn_embeddings(graph, 5)
embeddings = embeddings.detach().cpu()
print('adj list:')
for i in range(len(graph.adj_list)):
    print('{}: {}'.format(i, graph.adj_list[i]))

print('embeddings')
for i in range(embeddings.shape[0]):
    print('{}: {}'.format(i, embeddings[i].data))
print(embeddings.shape)

print('seed is: ', seed)

torch.save(embeddings, 'q13_13.pt')

# # plot the result:
# x = embeddings[:, 0].squeeze().numpy()
# y = embeddings[:, 1].squeeze().numpy()

# for i in range(10):
#     selected_node = i
#     neighbors = graph.adj_list[selected_node]
#     non_neighbors = list(set(range(graph.n_vertices)).difference(neighbors).difference([selected_node]))
#     colors = [''] * graph.n_vertices
#     for neighbor in neighbors: colors[neighbor] = 'r'
#     for non_neighbor in non_neighbors: colors[non_neighbor] = 'b'
#     colors[selected_node] = 'g'

#     # plt.scatter(x, y, c=['r','r','r','r','r', 'b','b','b','b','b'])
#     plt.scatter(x, y, c=colors)
#     plt.figure()

# plt.show()

import torch
from typing import List
from matplotlib import pyplot as plt
import numpy as np

def generate_cluster_centers(n_centers: int, min_center_distance: float, embedding_dim: int):
    growth_rate = np.exp(np.log(n_centers) / embedding_dim)
    # print('growth rate: ', growth_rate)
    lower_bound = (-3 * min_center_distance) * growth_rate
    upper_bound = (+3 * min_center_distance) * growth_rate
    # print('bounds: ', lower_bound, upper_bound)

    centers = torch.empty(n_centers, embedding_dim)
    centers[0] = torch.empty(embedding_dim).uniform_(lower_bound, upper_bound)
    n_created_centers = 1
    loop_limit = 100
    for i in range(1, n_centers):
        loop_counter = 0
        warned_user = False
        while True: # loop until you've created the right cluster_center
            new_center = torch.empty(embedding_dim).uniform_(lower_bound, upper_bound)
            new_center_repeated = new_center.unsqueeze(0).repeat(n_created_centers, 1)
            distances = torch.norm(new_center_repeated - centers[:n_created_centers], dim=1)

            if torch.min(distances) > min_center_distance:
                break

            loop_counter += 1
            if loop_counter >= loop_limit:
                if not warned_user:
                    print("WARNING: generate_cluster_centers looping too many times: n_centers: {}, min_center_distance: {}".format(
                        n_centers, min_center_distance
                    ))
                    warned_user = True
                else:
                    print('looping after hitting the limit: ', loop_counter)

        centers[n_created_centers] = new_center
        n_created_centers += 1

    return centers


def generate_clustered_embeddings(embedding_dim: int, n_embeddings: int,
    n_clusters: int, embedding_cluster_map: List[int], min_center_distance: float):

    cluster_centers = generate_cluster_centers(n_clusters, min_center_distance, embedding_dim)

    embeddings = torch.empty(n_embeddings, embedding_dim)

    for i in range(n_embeddings):
        target_center = cluster_centers[embedding_cluster_map[i]]
        noise = torch.normal(0., min_center_distance/8., target_center.shape)
        embeddings[i] = target_center + noise

    return embeddings


def generate_random_mapping(n_vertices, n_colors):
    mapping =  np.random.randint(0, n_colors, n_vertices)
    color_selected_n_times = [0] * n_colors
    for c in mapping:
        color_selected_n_times[c] += 1

    # print('n_vertices: {}, n_colors: {}, mapping: {}'.format(n_vertices, n_colors, mapping))
    # print(color_selected_n_times)
    # print('checking for 0s')
    for c in range(n_colors):
        while color_selected_n_times[c] < 1:
            # print('c: {} has zero colors'.format(c))
            random_position = np.random.randint(0, n_vertices)
            # print('random pos:', random_position)
            # print('mapping:', mapping[random_position])
            # print('n_times:', color_selected_n_times[mapping[random_position]])
            if color_selected_n_times[mapping[random_position]] > 1:
                # print('went into if')
                color_selected_n_times[mapping[random_position]] -= 1
                mapping[random_position] = c
                color_selected_n_times[c] += 1
            # print('after if:')
            # print(color_selected_n_times)
            # import time; time.sleep(1)
                
    return mapping
    

# ============================== visual tests: ==============================

# if __name__ == '__main__':
#     from manual_emb_utility import plot_points

#     centers = generate_cluster_centers(100, 10, 2)
#     plot_points(centers, annotate=True)
#     plt.show()

#     emb = generate_clustered_embeddings(5, 10, 3, [0, 0, 0, 1, 1, 1, 2, 2, 2, 2], 1)
#     plot_points(emb, annotate=True)
#     plt.show()

# ============================== generate data: ==============================

# if __name__ == '__main__':
#     embedding_dim = 10

#     for i in range(100):
#         n_clusters = np.random.randint(2, 10)
#         n_embeddings = 25
#         embedding_to_cluster_map = generate_random_mapping(n_embeddings, n_clusters)
#         embeddings = generate_clustered_embeddings(embedding_dim, n_embeddings, n_clusters, embedding_to_cluster_map, 2)
#         torch.save(embeddings, '../data/dummy_embeddings/{}.pt'.format(i))


# ============================== full coloring test: ==============================

# if __name__ == '__main__':
    # from exact_coloring import find_k_coloring
    # import networkx as nx
    # from graph import Graph

    # import time
    # t1 = time.time()
    # g = Graph.from_networkx_graph(nx.erdos_renyi_graph(100, 0.5))
    # c = find_k_coloring(g, 11)
    # t2 = time.time()
    # print(t2 - t1)
    # print(c)
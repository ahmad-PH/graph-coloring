from graph import Graph
import os
import networkx as nx
from graph_utility import generate_erdos_renyi_graph
from heuristics import colorize_using_heuristic, slf_heuristic
from exact_coloring import find_chromatic_number_upper_bound

def generate_erdos_renyi(foldername, n_train, n_valid, n_test, n_vertices, p = 0.5):
    os.mkdir(foldername)
    for subfoldername, graph_count in [('train', n_train), ('valid', n_valid), ('test', n_test)]:
        os.mkdir(os.path.join(foldername, subfoldername))
        for i in range(graph_count):
            G = Graph.from_networkx_graph(nx.erdos_renyi_graph(n_vertices, p))
            G.save("{}/{}/{}.graph".format(foldername, subfoldername, i))

def generate_watts_strogatz(foldername, n_train, n_valid, n_test, n_vertices, k = 4, beta = 0.5):
    os.mkdir(foldername)
    for subfoldername, graph_count in [('train', n_train), ('valid', n_valid), ('test', n_test)]:
        os.mkdir(os.path.join(foldername, subfoldername))
        for i in range(graph_count):
            G = Graph.from_networkx_graph(nx.watts_strogatz_graph(n_vertices, k, beta))
            G.save("{}/{}/{}.graph".format(foldername, subfoldername, i))


def generate_erdos_renyi_improvable_by(foldername, n_graphs, n_vertices, p = 0.5, improvable_by = 2, verbose = False):
    os.mkdir(foldername)
    differences = []
    for i in range(n_graphs):
        while True:
            graph = generate_erdos_renyi_graph(n_vertices, p)
            _, slf_n_colors = colorize_using_heuristic(graph.adj_list, slf_heuristic)
            optimal_n_colors = find_chromatic_number_upper_bound(graph, 15, verbose)
            difference = slf_n_colors - optimal_n_colors
            differences.append(difference)
            if verbose: print('slf: {}, optimal: {}, diff: {}'.format(slf_n_colors, optimal_n_colors, difference))
            if difference >= improvable_by:
                graph.save('{}/{}.graph'.format(foldername, i))
                break
            else:
                if verbose: print('too little difference, continuing.')
            print('')

    if verbose:
        print('diffs:', differences)
        print('mean:', np.mean(differences))
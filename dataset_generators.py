from graph import Graph
import os
import networkx as nx

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

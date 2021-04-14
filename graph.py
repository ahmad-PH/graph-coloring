import networkx as nx
from scipy.io import mmread, mminfo
from typing import List, Union
from io import TextIOBase, StringIO, IOBase

from utility import adj_list_to_matrix


def networkx_graph_to_adj_list(graph):
    adj_list = []
    for i in range(len(graph.adj)):
        adj_list.append(sorted(graph.adj[i].keys()))
    return adj_list

class Graph:
    def __init__(self, adj_list, name=None):
        self.set_name(name)
        self.adj_list = adj_list
        self.nx_graph = None
        self.n_vertices = len(adj_list)
        self.n_edges = int(sum(len(row) for row in adj_list) / 2)
        self.adj_matrix = None
    
    @staticmethod
    def from_networkx_graph(graph):
        G = Graph(networkx_graph_to_adj_list(graph))
        G.nx_graph = graph
        return G

    def save(self, target: Union[str, TextIOBase]):
        if isinstance(target, str):
            with open(target, 'w') as f:
                self._save(f)
        elif isinstance(target, TextIOBase):
            self._save(target)

    def _save(self, iobase):
        for row in self.adj_list:
            for neighbor in row:
                iobase.write(str(neighbor) + ' ')
            iobase.write('\n')

    @staticmethod
    def load(source: Union[str, TextIOBase]):
        if isinstance(source, str):
            with open(source, 'r') as f:
                return Graph._load(f).set_name(source)
        elif isinstance(source, TextIOBase):
            return Graph._load(source)

    @staticmethod
    def _load(source):
        adj_list = []
        line = source.readline()
        while line != "": # EOF not reached
            row = [int(x) for x in line.split()]
            adj_list.append(row)
            line = source.readline()
        return Graph(adj_list)

    @staticmethod
    def from_mtx_file(path):
        coo_matrix = mmread(path)
        coo_matrix.eliminate_zeros()
        coo_matrix.sum_duplicates()

        rows, cols, entries, _format, field, symmetry = mminfo(path)
        assert rows == cols, "rows != cols"
        assert _format == "coordinate", "format = array not yet tested"

        adj_list : List[List[int]] = [[] for _ in range(rows)]
        for row, column in zip(coo_matrix.row, coo_matrix.col):
            if row != column:
                adj_list[row].append(column)

        return Graph(adj_list)

    def get_adj_matrix(self):
        if self.adj_matrix is None:
            self.adj_matrix = adj_list_to_matrix(self.adj_list)
        return self.adj_matrix

    def get_nx_graph(self):
        if self.nx_graph is None:
            self.nx_graph = self._calculate_nx_graph()
        return self.nx_graph

    def set_name(self, name):
        self.name = name
        return self

    def _calculate_nx_graph(self):
        G = nx.Graph()
        for v1, row in enumerate(self.adj_list):
            for v2 in row:
                G.add_edge(v1, v2)
        return G

    def __str__(self):
        return 'graph: ' + str(self.adj_list)

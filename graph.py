
import networkx as nx

def networkx_graph_to_adj_list(graph):
    adj_list = []
    for i in range(len(graph.adj)):
        adj_list.append(sorted(graph.adj[i].keys()))
    return adj_list


class Graph:
    def __init__(self, adj_list):
        self.adj_list = adj_list
        self.nx_graph = None
    
    @staticmethod
    def from_networkx_graph(graph):
        G = Graph(networkx_graph_to_adj_list(graph))
        G.nx_graph = graph
        return G

    def save(self, path):
        with open(path, 'w') as f:
            for row in self.adj_list:
                for neighbor in row:
                    f.write(str(neighbor) + ' ')
                f.write('\n')
        
    @staticmethod
    def load(path):
        adj_list = []
        with open(path, 'r') as f:
            line = f.readline()
            while line != "": # EOF not reached
                row = [int(x) for x in line.split()]
                adj_list.append(row)
                line = f.readline()
        return Graph(adj_list)
        
    def get_nx_graph(self):
        if self.nx_graph is None:
            self.nx_graph = self._calculate_nx_graph()
        return self.nx_graph

    def _calculate_nx_graph(self):
        G = nx.Graph()
        for v1, row in enumerate(self.adj_list):
            for v2 in row:
                G.add_edge(v1, v2)
        return G

    def __str__(self):
        return 'graph: ' + str(self.adj_list)


def networkx_graph_to_adj_list(graph):
    adj_list = []
    for i in range(len(graph.adj)):
        adj_list.append(sorted(graph.adj[i].keys()))
    return adj_list


class Graph:
    def __init__(self, adj_list):
        self.adj_list = adj_list
    
    @staticmethod
    def from_networkx_graph(graph):
        return Graph(networkx_graph_to_adj_list(graph))

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

    
    def __str__(self):
        return str(self.adj_list)

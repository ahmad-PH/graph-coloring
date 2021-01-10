from graph import Graph
import itertools

def is_proper_coloring(coloring, graph):
    for v1, row in enumerate(graph.adj_list):
        for v2 in row:
            if coloring[v1] == coloring[v2]:
                return False
    return True

def coloring_properties(coloring, graph):
    n_violations = 0
    for v1, row in enumerate(graph.adj_list):
        for v2 in row:
            if coloring[v1] == coloring[v2]:
                n_violations += 1
                # print('violation: ({}, {})'.format(v1, v2))

    n_violations /= 2 # because we counted each twice
    return n_violations == 0, n_violations, float(n_violations) / graph.n_edges


def generate_kneser_graph(n, k):
    subsets = [frozenset(subset) for subset in itertools.combinations(range(n), k)]
    adj_list = [[] for _ in range(len(subsets))]
    for i in range(len(subsets)):
        for j in range(i+1, len(subsets)):
            if subsets[i].isdisjoint(subsets[j]):
                adj_list[i].append(j)
                adj_list[j].append(i)
    return Graph(adj_list, 'k{}_{}'.format(n, k))

def kneser_graph_chromatic_number(n, k):
    if n < 2 * k:
        return 1
    else:
        return n - 2*k + 2

def generate_queens_graph(m, n):
    chess_coord_to_vertex_ind = lambda i, j: i * n + j
    adj_list = [[] for _ in range(m * n)]
    for i1 in range(m):
        for j1 in range(n):
            index = chess_coord_to_vertex_ind(i1, j1)

            # add the row
            i2 = i1
            for j2 in [x for x in range(n) if x != j1]:
                adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))

            # add the column
            j2 = j1
            for i2 in [x for x in range(m) if x != i1]:
                adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))

            # add the two diagonals
            for i2 in [x for x in range(m) if x != i1]:
                j2 = j1 + (i2-i1)
                if j2 >= 0 and j2 < n:
                    adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))

                j2 = j1 - (i2-i1)
                if j2 >= 0 and j2 < n:
                    adj_list[index].append(chess_coord_to_vertex_ind(i2, j2))


    return Graph(adj_list, 'q{}_{}'.format(m, n))
            

def sort_graph_adj_list(adj_list):
    for neighborhood in adj_list:
        neighborhood.sort()

def n_edges(adj_list):
    sum_of_degrees =  sum([len(neighborhood) for neighborhood in adj_list])
    if sum_of_degrees % 2 != 0:
        raise ValueError('sum of degrees in adjacency list is not even.')
    return int(sum_of_degrees / 2)
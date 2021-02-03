from graph import Graph
# import snap

# def convert(graph: snap.TNGraph):
#     adj_list = [[] for _ in range(graph.GetNodes())]
#     for edge in graph.Edges():
#         v1, v2 = edge.GetId()
#         print(v1, v2)
#         print(graph.GetNodes())
#         if v2 in adj_list[v1]:
#             print('{} is on adj_list of {}'.format(v2, v1))
#         if v1 in adj_list[v2]:
#             print('{} is on adj_list of {}'.format(v1, v2))
#         adj_list[v1].append(v2)
#         adj_list[v2].append(v1)

def read_edge_list(filepath):
    edge_list = []
    with open(filepath) as f:
        for line in f.readlines():

            if line.startswith('#'):
                continue

            tokens = line.split()
            if len(tokens) != 2:
                raise Exception("snap graph line contains more than two tokens: " + str(tokens))
            v1, v2 = int(tokens[0]), int(tokens[1])
            edge_list.append((v1, v2))

    return edge_list

def edgelist_find_max_node_index(edge_list):
    max_node_index = float('-inf')
    for v1, v2 in edge_list:
        if v1 > max_node_index:
            max_node_index = v1
        if v2 > max_node_index:
            max_node_index = v2
    return max_node_index

def edgelist_find_untouched_nodes(edge_list, max_node_index):
    touched = [False] * (max_node_index + 1)
    for v1, v2 in edge_list:
        touched[v1] = True
        touched[v2] = True

    untouched_nodes = []
    for i, is_touched in enumerate(touched):
        if is_touched == False:
            untouched_nodes.append(i)

    return untouched_nodes, touched

def edgelist_eliminate_isolated_nodes(edge_list, max_node_index, touched_map):
    shift_amounts = [0] * (max_node_index + 1)

    shift_amount = 0
    for i in range(len(shift_amounts)):
        shift_amounts[i] = shift_amount
        if touched_map[i] == False:
            shift_amount -= 1

    for i, edge in enumerate(edge_list):
        v1, v2 = edge
        edge_list[i] = (v1 + shift_amounts[v1], v2 + shift_amounts[v2])

def edgelist_eliminate_self_loops(edge_list):
    i = 0
    length = len(edge_list)
    while i < length:
        if edge_list[i][0] == edge_list[i][1]:
            edge_list.pop(i)
            i -= 1
            length -= 1
        i += 1

def load_snap_graph(filepath):
    edges = read_edge_list(filepath)

    max_node_index = edgelist_find_max_node_index(edges)
    untouched_nodes, touched_map = edgelist_find_untouched_nodes(edges, max_node_index)

    edgelist_eliminate_isolated_nodes(edges, max_node_index, touched_map)

    max_node_index = edgelist_find_max_node_index(edges)
    untouched_nodes, _ = edgelist_find_untouched_nodes(edges, max_node_index)
    assert len(untouched_nodes) == 0, "graph contains untouched nodes after elimination step."

    edgelist_eliminate_self_loops(edges)

    n_nodes = max_node_index + 1
    adj_list = [[] for _ in range(n_nodes)]
    for v1, v2 in edges:
        adj_list[v1].append(v2)
        adj_list[v2].append(v1)
    
    for i, adj_list_row in enumerate(adj_list):
        adj_list[i] = list(set(adj_list_row))

    graph = Graph(adj_list)

    print('graph: nodes: {}, edges: {}'.format(graph.n_vertices, graph.n_edges))

    return graph


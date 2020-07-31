import numpy as np

def erdos_renyi(n, p):
    if p > 1. or p <0.:
        raise ValueError('expected p to be between 0. and 1. but got: {}'.format(p))
    
    adj_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.uniform(0, 1) < p:
                adj_matrix[i][j] = adj_matrix[j][i] = 1
    
    if not np.all(adj_matrix == adj_matrix.T):
        raise Exception('adjacency matrix is not symmetric')

    return adj_matrix

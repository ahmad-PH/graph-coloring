import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, dropout, alpha, concat=True, glimpse_mode=False):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.glimpse_mode = glimpse_mode

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # if glimpse_mode:
            # self.b = nn.Parameter(torch.empty(size=(2*out_features, 1)))
            # nn.init.xavier_uniform_(self.b.data, gain=1.414)

        # self.a = nn.Sequential(
        #     nn.Linear(2*out_features, out_features),
        #     nn.ReLU(),
        #     nn.Linear(out_features, 1)
        # )
        # def w_init(l):
        #     if isinstance(l, nn.Linear):
        #         nn.init.xavier_uniform_(l.weight.data, gain=1.414)

        # self.a.apply(w_init)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)

        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # e = self.leakyrelu(self.a(a_input).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)

        if self.glimpse_mode:
            # acceptance_scores = torch.sigmoid(torch.matmul(a_input, self.b).squeeze(2))
            # attention = torch.where((adj + torch.eye(adj.shape[0]).to(adj.get_device())) == 0, e, zero_vec)
            attention = torch.where(torch.eye(adj.shape[0]).to(adj.get_device()) == 0, e, zero_vec)
            return F.softmax(attention, dim=1) # , acceptance_scores

        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training) # TEST: to see if removing dropout improves performance
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks): 
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        # 
        # These are the rows of the second matrix (Wh_repeated_alternating): 
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN 
        # '----------------------------------------------------' -> N times
        # 
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class GraphSingularAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):
        super().__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, target_vertex):
        Wh = torch.mm(h, self.W)
        N = h.shape[0]
        expanded_Wh = Wh[target_vertex].unsqueeze(0).expand(N, -1)
        all_combinations_matrix = torch.cat([expanded_Wh, Wh], dim = 1)
        e = self.leakyrelu(torch.matmul(all_combinations_matrix, self.a)).squeeze(1)
        alpha = F.softmax(e, dim=0)
        alpha = F.dropout(alpha, self.dropout, training=self.training) # TEST: dropout is probably not good for graph col
        h_prime = torch.mm(alpha.unsqueeze(0), Wh)
        return F.elu(h_prime) # TEST : my idea: torch.tanh(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



# commented-out because it is bugged: multiple attention layers require the output of the first layer not to be singular
# class SGAT(nn.Module):
#     def __init__(self, nfeat, nhid, nout, dropout, alpha, nheads):
#         """Dense version of GAT."""
#         super(SGAT, self).__init__()
#         self.dropout = dropout

#         self.attentions = [GraphSingularAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
#         for i, attention in enumerate(self.attentions):
#             self.add_module('attention_{}'.format(i), attention)

#         self.out_att = GraphSingularAttentionLayer(nhid * nheads, nout, dropout=dropout, alpha=alpha, concat=False)
#         self.add_module('out_att', self.out_att)

#     def forward(self, x, target_index):
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = torch.cat([att(x, target_index) for att in self.attentions], dim=1)
#         x = F.dropout(x, self.dropout, training=self.training)
#         x = F.elu(self.out_att(x, target_index))
#         return x


import torch
import numpy as np
from torch import nn
import math
import globals

class MultiHeadAttentionOriginal(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttentionOriginal, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_q_dim,
            input_h_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None,
            pointer_mode=False,
            name=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        if pointer_mode:
            assert n_heads == 1, "n_heads must be 1 if pointer_mode is True"

        self.n_heads = n_heads
        self.input_q_dim = input_q_dim
        self.input_h_dim = input_h_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim
        self.pointer_mode = pointer_mode

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_q_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_h_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_h_dim, val_dim))

        if pointer_mode == False:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

        if name is not None:
            self.name = "{}-{}".format(self.__class__.__name__, name)
        else:
            self.name = self.__class__.__name__

    def _get_name(self):
        return self.name

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        """
        :param q: queries (batch_size, n_query, input_q_dim)
        :param h: data (batch_size, graph_size, input_h_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_h_dim = h.size()
        _, n_query, input_q_dim = q.size()
        assert q.size(0) == batch_size, "Batch sizes don't match"
        assert input_q_dim == self.input_q_dim, "Wrong dimensions for q"
        assert input_h_dim == self.input_h_dim, "Wrong dimensions for h"

        hflat = h.contiguous().view(-1, input_h_dim)
        qflat = q.contiguous().view(-1, input_q_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        self.Q, self.K, self.V = Q, K, V # for logging purposes

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1) 
        # attn.shape == (n_heads, batch_size, n_query, graph_size)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        if self.pointer_mode:
            return attn.squeeze(0) # remove the n_heads dimension since n_heads == 1

        heads = torch.matmul(attn, V)

        heads = heads.transpose(0, 1) # swap the dimensions for batch and heads to align it for the matmul
        projected_heads = torch.matmul(heads, self.W_out)
        out = torch.sum(projected_heads, dim=1) # sum across heads
        self.out = out # for logging purposes

        if globals.debug_mode:
            self.out.retain_grad()

        return out

class MHAWithoutBatch(MultiHeadAttention):
    def forward(self, q, h, mask=None):
        q = q.unsqueeze(0)
        h = h.unsqueeze(0)
        return super().forward(q, h, mask=mask).squeeze(0)

class MHAAdapter(MHAWithoutBatch):
    def __init__(self, n_heads, input_dim, output_dim, name=None):
        super().__init__(n_heads, input_dim, input_dim, output_dim, pointer_mode=False, name=name)

    def forward(self, h, mask=None):
        mask = (1 - mask).bool()
        return super().forward(h, h, mask=mask).squeeze(0)

# batch_size = 3
# n_query = 2
# graph_size = 4
# input_dim = 5

# h = torch.normal(0, 0.5, [batch_size, graph_size, input_dim])
# q = torch.normal(10., 1., [batch_size, n_query, input_dim])

# torch.random.manual_seed(0)
# attn_orig = MultiHeadAttentionOriginal(n_heads=6, input_dim=input_dim, key_dim=5, embed_dim=18)

# torch.random.manual_seed(0)
# attn = MultiHeadAttention(n_heads=6, input_q_dim=input_dim, input_h_dim=input_dim, key_dim=5, embed_dim=18)

# out_orig = attn_orig.forward(q, h)
# out = attn(q, h)

# print(torch.allclose(out_orig, out))
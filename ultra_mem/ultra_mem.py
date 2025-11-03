from math import sqrt

import torch
from torch import nn, tensor, randn
import torch.nn.functional as F
from torch.nn import Linear, Identity, Sequential, Parameter, Module, ModuleList

from einx import add, multiply
from einops import rearrange, repeat, reduce, einsum
from einops.layers.torch import Rearrange

# einstein notation
# b - batch
# n - sequence
# m - memories
# d - feature dimension
# i, j - row, col
# h - heads
# r - tucker decomposition rank

# helper function

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def is_odd(n):
    return not divisible_by(n, 2)

def scale_gradient(t, scale = 1.):
    # scales the gradient, controlling effective lr upstream

    if scale == 1.:
        return t

    scaled_t = t * scale
    return t.detach() - scaled_t + scaled_t.detach()

# classes

class UltraMem(Module):

    def __init__(
        self,
        dim,
        dim_out,
        dim_values = None,
        num_memories = 1_000_000,
        topk = 32,
        dim_queries_keys = 128,         # think this is what PKM uses
        core_rank = 2,                  # the tucker decomposition core rank
        core_heads = 2,                 # number of cores / heads
        core_aux_loss_margin = 0.15,
        aux_loss_weight = 0.1,
        value_expansion = 4,
        pre_query_causal_conv = True,
        query_conv_kernel_size = 5,
        qk_layernorm = True,
        prenorm = True,
        proj_out = None,
        mem_init_std = 1e-2,
        mem_lr_scale = 1e1               # the values / memories needed 10x the learning rate (~1e-3 compared with ~1e-4 base lr, this could be controlled without doing param groups with a trick)
    ):
        super().__init__()

        # variables

        assert sqrt(num_memories).is_integer()
        num_keys = int(sqrt(num_memories))

        dim_values = default(dim_values, dim // core_heads)

        # prenorm and queries

        self.prenorm = nn.RMSNorm(dim) if prenorm else Identity()

        assert is_odd(query_conv_kernel_size)

        self.pre_query_causal_conv = Sequential(
            Rearrange('b n d -> b d n'),
            nn.ZeroPad1d((query_conv_kernel_size - 1, 0)),
            nn.Conv1d(dim, dim, query_conv_kernel_size, groups = dim),
            Rearrange('b d n -> b n d')
        ) if pre_query_causal_conv else Identity()

        self.to_queries = Linear(dim, dim_queries_keys, bias = False)

        # memory related

        self.heads = core_heads
        self.rank = core_rank

        self.num_keys = num_keys
        self.topk = topk

        self.query_ln = nn.LayerNorm(dim_queries_keys, bias = False) if qk_layernorm else Identity()
        self.key_ln = nn.LayerNorm(dim_queries_keys, bias = False) if qk_layernorm else Identity()

        self.keys = Parameter(randn(2, core_rank, num_keys, dim_queries_keys) * 1e-2)

        # their tucker decomposed core is 2x2
        # learned e2e with an auxiliary loss

        self.core = Parameter(randn(core_heads, core_rank, core_rank) * 1e-2)

        # handle value expansion

        assert divisible_by(num_memories, value_expansion)

        self.value_expansion_proj = Parameter(randn(value_expansion, dim_values, dim_values) * 1e-2)

        batch_randperm = randn(num_memories // value_expansion, value_expansion).argsort(dim = -1)
        self.register_buffer('rand_proj_mapping', batch_randperm.flatten())

        # memories

        self.memories = Parameter(randn(num_memories, dim_values) * mem_init_std)

        # memories lr

        self.mem_lr_scale = mem_lr_scale

        # whether to have a projection from (head * dim_values) back to (dim)

        proj_out = default(proj_out, core_heads * dim_values != dim)
        self.combine_values_to_out = Linear(core_heads * dim_values, dim, bias = False) if proj_out else Identity()

        # auxiliary loss on the core

        self.aux_loss_weight = aux_loss_weight
        self.core_aux_loss_margin = core_aux_loss_margin

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens,
        trainable_sparse_mask = None, # bool[num_memories,]
        return_aux_loss = None
    ):

        tokens = self.prenorm(tokens)

        # svd

        u, s, t = torch.svd(self.core)

        u_vec = u[..., 0]
        t_vec = t[..., 0]

        # aux loss on singular values

        return_aux_loss = default(return_aux_loss, self.training)

        aux_loss = self.zero

        if return_aux_loss:
            non_first_singular_values = s[:, 1:]

            aux_loss = F.relu(non_first_singular_values - self.core_aux_loss_margin).pow(2).mean(dim = -1) # eq (12)

            aux_loss = aux_loss.sum() * self.aux_loss_weight

        # queries keys

        tokens = self.pre_query_causal_conv(tokens)
        queries = self.to_queries(tokens)

        keys = self.keys

        # query key layernorm for stability

        queries = self.query_ln(queries)
        keys = self.key_ln(keys)

        row_scores, col_scores = einsum(queries, self.keys, 'b n d, rc r m d -> rc b n m r')

        # tucker decompsed qk retrieval, following fig 4

        merged_row_scores = einsum(row_scores, u_vec, '... r, h r -> h ...')
        merged_col_scores = einsum(col_scores, t_vec, '... r, h r -> h ...')

        top_row_indices = merged_row_scores.topk(self.topk, dim = -1).indices
        top_col_indices = merged_col_scores.topk(self.topk, dim = -1).indices

        indices = add('... i, ... j -> ... (i j)', top_row_indices * self.num_keys, top_col_indices)

        # ready for filtered row / col scores

        top_row_indices, top_col_indices = tuple(repeat(t, '... -> ... r', r = self.rank) for t in (top_row_indices, top_col_indices))
        row_scores, col_scores = tuple(repeat(t, '... -> h ...', h = self.heads) for t in (row_scores, col_scores))

        filtered_row_scores = row_scores.gather(-2, top_row_indices)
        filtered_col_score = col_scores.gather(-2, top_col_indices)

        scores = einsum(filtered_row_scores, filtered_col_score, self.core, 'h ... i r1, h ... j r2, h r1 r2 -> h ... i j')
        scores = rearrange(scores, '... i j -> ... (i j)')

        # get final scores and memory indices - (head, batch, seq, sel mems)

        final_scores, top_merged_score_indices = scores.topk(self.topk, dim = -1)
        final_indices = indices.gather(-1, top_merged_score_indices)

        # they use non-competitive scores, corroborating Csordas et al. 2023

        final_scores = final_scores.sigmoid()

        # fetch the memories, and also handle sparse finetuning

        memories = self.memories[final_indices]

        # change gradients to memories if needed

        memories = scale_gradient(memories, self.mem_lr_scale)

        # sparse finetuning

        if exists(trainable_sparse_mask):
            assert len(trainable_sparse_mask) == self.num_memories

            grad_mask = trainable_sparse_mask[final_indices]

            masked_memories = grad_mask * memories
            memories = memories.detach() + masked_memories - masked_memories.detach()

        # multiply by the scores

        aggregated = einsum(memories, final_scores, '... m d, ... m -> ... d')

        # concat the MCS heads and maybe combine

        concatted_heads = rearrange(aggregated, 'h b n d -> b n (h d)')

        out = self.combine_values_to_out(concatted_heads)

        # returning

        return out, aux_loss

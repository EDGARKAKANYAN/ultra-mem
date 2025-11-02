import torch
from torch.nn import Module, ModuleList

from einops import rearrange

# helper function

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class UltraMem(Module):

    def __init__(
        self,
        dim,
        dim_values,
        dim_out,
        dim_queries_keys = 128, # think this is what PKM uses
        core_rank = 2,          # the tucker decomposition core rank
        core_heads = 2,         # number of cores / heads
    ):
        super().__init__()

    def forward(
        self,
        x,
        trainable_sparse_mask = None # bool[num_memories,]
    ):
        return x

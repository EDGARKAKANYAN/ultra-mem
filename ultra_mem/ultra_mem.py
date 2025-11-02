import torch
import torch.nn.functional as F
from torch.nn import Parameter, Module, ModuleList

from einops import rearrange

# helper function

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

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

        # their tucker decompose core is 2x2
        # learned e2e with an auxiliary loss

        self.core = Parameter(torch.randn(core_heads, core_rank, core_rank) * 1e-2)

    def forward(
        self,
        tokens,
        trainable_sparse_mask = None # bool[num_memories,]
    ):
        return tokens

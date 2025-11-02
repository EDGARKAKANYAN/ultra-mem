import torch
from torch import tensor
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
        core_aux_loss_margin = 0.15,
        aux_loss_weight = 0.1,
    ):
        super().__init__()

        # their tucker decomposed core is 2x2
        # learned e2e with an auxiliary loss

        self.core = Parameter(torch.randn(core_heads, core_rank, core_rank) * 1e-2)

        # auxiliary loss on the core

        self.core_aux_loss_margin = core_aux_loss_margin
        self.aux_loss_weight = aux_loss_weight

        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        tokens,
        trainable_sparse_mask = None, # bool[num_memories,]
        return_aux_loss = None
    ):

        # svd

        u, s, v = torch.svd(self.core)

        # aux loss on singular values

        return_aux_loss = default(return_aux_loss, self.training)

        aux_loss = self.zero

        if return_aux_loss:
            non_first_singular_values = s[:, 1:]

            aux_loss = F.relu(non_first_singular_values - self.core_aux_loss_margin).pow(2).mean() # eq (12)
            aux_loss = aux_loss * self.aux_loss_weight

        # returning

        return tokens, aux_loss

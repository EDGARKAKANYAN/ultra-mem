import pytest

import torch

def test_ultra_mem():
    from ultra_mem.ultra_mem import UltraMem

    x = torch.randn(1, 1024, 512)

    ultra_mem = UltraMem(
        512,
        core_heads = 2,
        dim_values = 64,
        dim_out = 256,
    )

    out, aux_loss = ultra_mem(x)
    assert out.shape == x.shape and aux_loss.numel() == 1

import pytest
param = pytest.mark.parametrize

import torch

@param('value_expansion', (1, 2))
@param('core_heads', (1, 2))
def test_ultra_mem(
    value_expansion,
    core_heads
):
    from ultra_mem.ultra_mem import UltraMem

    x = torch.randn(1, 1024, 512)

    ultra_mem = UltraMem(
        512,
        dim_values = 64,
        dim_out = 256,
        value_expansion = value_expansion,
        core_heads = core_heads
    )

    out, aux_loss = ultra_mem(x)
    assert out.shape == x.shape and aux_loss.numel() == 1

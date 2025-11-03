import pytest
param = pytest.mark.parametrize

import torch

@param('value_expansion', (1, 2))
@param('core_heads', (1, 2))
@param('layers_for_mem_init', (None, 12))
def test_ultra_mem(
    value_expansion,
    core_heads,
    layers_for_mem_init
):
    from ultra_mem.ultra_mem import UltraMem

    x = torch.randn(1, 1024, 512)

    ultra_mem = UltraMem(
        512,
        value_expansion = value_expansion,
        core_heads = core_heads,
        layers_for_mem_init = layers_for_mem_init
    )

    out, aux_loss = ultra_mem(x)
    assert out.shape == x.shape and aux_loss.numel() == 1

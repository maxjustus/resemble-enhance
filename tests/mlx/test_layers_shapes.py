import numpy as np
import torch
import mlx.core as mx

from resemble_enhance.mlx_backend.layers import ConvTranspose1dCF, UNet
from resemble_enhance.mlx_backend.ops import torch_conv_transpose1d_weight_to_mlx


def test_conv_transpose1d_parity():
    torch.manual_seed(0)
    x = np.random.default_rng(0).standard_normal((2, 3, 14), dtype=np.float32)
    torch_mod = torch.nn.ConvTranspose1d(3, 3, 14, stride=7, padding=4, output_padding=1)
    mlx_mod = ConvTranspose1dCF(3, 3, 14, stride=7, padding=4, output_padding=1)
    mlx_mod.layer.weight = mx.array(torch_conv_transpose1d_weight_to_mlx(torch_mod.weight.detach().numpy()), dtype=mx.float32)
    mlx_mod.layer.bias = mx.array(torch_mod.bias.detach().numpy(), dtype=mx.float32)

    with torch.no_grad():
        y_torch = torch_mod(torch.from_numpy(x)).numpy()
    y_mlx = np.array(mlx_mod(mx.array(x)))

    assert y_torch.shape == y_mlx.shape
    np.testing.assert_allclose(y_mlx, y_torch, rtol=1e-5, atol=1e-5)


def test_unet_shape():
    model = UNet(3, 3)
    x = mx.zeros((1, 3, 31, 47), dtype=mx.float32)
    y = model(x)
    mx.eval(y)
    assert y.shape == x.shape

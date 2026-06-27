import numpy as np

from resemble_enhance.hparams import HParams
from resemble_enhance.mlx_backend.denoiser import MLXDenoiser


def test_denoiser_shape_random_input():
    model = MLXDenoiser(HParams())
    x = np.random.default_rng(0).standard_normal((1, 2048), dtype=np.float32)
    y = model(x)
    assert y.shape == x.shape
    assert y.dtype == np.float32

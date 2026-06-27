import numpy as np

from resemble_enhance.enhancer.hparams import HParams
from resemble_enhance.mlx_backend.enhancer import MLXEnhancer


def test_enhancer_forward_runs_on_short_input():
    model = MLXEnhancer(HParams())
    model.configurate_(nfe=4, solver="midpoint", lambd=0.5, tau=0.5)
    x = np.random.default_rng(0).standard_normal(2048 + 441, dtype=np.float32)
    y = model(x, seed=0)
    assert y.ndim == 2
    assert y.shape[0] == 1
    assert y.dtype == np.float32

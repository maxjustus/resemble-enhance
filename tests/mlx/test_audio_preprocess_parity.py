import numpy as np
import torch

from resemble_enhance.enhancer.inference import load_enhancer as load_torch_enhancer
from resemble_enhance.inference import remove_weight_norm_recursively
from resemble_enhance.mlx_backend.checkpoint import export_enhancer_weights
from resemble_enhance.mlx_backend.enhancer import load_enhancer


def _normalized(x: np.ndarray) -> np.ndarray:
    return x / np.abs(x).max().clip(min=1e-7)


def test_mel_parity_on_synthetic_inputs():
    export_enhancer_weights()
    torch_model = load_torch_enhancer(None, "cpu")
    remove_weight_norm_recursively(torch_model)
    torch_model.eval()
    mlx_model = load_enhancer("artifacts/mlx/enhancer_stage2.safetensors")

    sr = torch_model.hp.wav_rate
    t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False, dtype=np.float32)
    signals = [
        np.sin(2 * np.pi * 440 * t).astype(np.float32),
        np.random.default_rng(0).standard_normal(len(t), dtype=np.float32),
        np.linspace(-1, 1, 128, dtype=np.float32),
    ]

    for signal in signals:
        signal = _normalized(signal)
        with torch.no_grad():
            mel_torch = torch_model.to_mel(torch.from_numpy(signal)[None]).cpu().numpy()
        mel_mlx = mlx_model.to_mel(signal[None])
        assert mel_torch.shape == mel_mlx.shape
        np.testing.assert_allclose(mel_mlx, mel_torch, rtol=1e-4, atol=1e-4)

import numpy as np
import torch
import mlx.core as mx

from resemble_enhance.enhancer.inference import load_enhancer as load_torch_enhancer
from resemble_enhance.inference import remove_weight_norm_recursively
from resemble_enhance.mlx_backend.checkpoint import export_denoiser_weights, export_enhancer_weights
from resemble_enhance.mlx_backend.denoiser import load_denoiser
from resemble_enhance.mlx_backend.enhancer import load_enhancer
from resemble_enhance.mlx_backend.parity import max_abs_diff, mean_abs_diff, si_sdr


def test_denoiser_torch_mlx_parity():
    export_denoiser_weights()
    x = np.random.default_rng(0).standard_normal(4096, dtype=np.float32)

    torch_model = load_torch_enhancer(None, "cpu").denoiser
    remove_weight_norm_recursively(torch_model)
    torch_model.eval()
    with torch.no_grad():
        y_torch = torch_model(torch.from_numpy(x)[None]).cpu().numpy()[0]

    mlx_model = load_denoiser("artifacts/mlx/denoiser.safetensors")
    y_mlx = mlx_model(x)[0]

    assert max_abs_diff(y_torch, y_mlx) < 2e-3
    assert mean_abs_diff(y_torch, y_mlx) < 1e-4


def test_enhancer_subcomponents_match_with_torch_noise():
    export_enhancer_weights()
    x = np.random.default_rng(0).standard_normal(2048 + 441, dtype=np.float32)

    torch_model = load_torch_enhancer(None, "cpu")
    remove_weight_norm_recursively(torch_model)
    torch_model.eval()
    torch_model.configurate_(nfe=4, solver="midpoint", lambd=0.5, tau=0.5)

    mlx_model = load_enhancer("artifacts/mlx/enhancer_stage2.safetensors")
    mlx_model.configurate_(nfe=4, solver="midpoint", lambd=0.5, tau=0.5)

    with torch.no_grad():
        x_t = torch.from_numpy(x)[None]
        x_norm = x_t / (x_t.abs().max(dim=-1, keepdim=True).values + 1e-7)
        mel_torch = torch_model.normalizer(torch_model.to_mel(x_norm), update=False)
        mel_np = mel_torch.cpu().numpy().astype(np.float32)
        enc_torch = torch_model.lcfm.ae.encode(mel_torch).cpu().numpy().astype(np.float32)

    psi0_noise = torch.randn(
        (1, mlx_model.hp.lcfm_latent_dim, mel_np.shape[-1]),
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).numpy().astype(np.float32)
    psi0_np = 0.5 * psi0_noise + 0.5 * (enc_torch * torch_model.lcfm.z_scale)

    with torch.no_grad():
        torch.manual_seed(0)
        decoded_torch = torch_model.lcfm(mel_torch, ψ0=mel_torch).cpu().numpy()

    decoded_mlx = np.asarray(
        mlx_model.lcfm.ae.decode(
            mlx_model.lcfm.cfm(
                mx.array(mel_np, dtype=mx.float32),
                psi0=mx.array(psi0_np, dtype=mx.float32),
                rng=np.random.default_rng(123),
            )
            / mlx_model.lcfm.z_scale
        )
    )

    assert max_abs_diff(decoded_torch, decoded_mlx) < 1e-4
    assert mean_abs_diff(decoded_torch, decoded_mlx) < 1e-5
    assert si_sdr(decoded_torch, decoded_mlx) > 80.0

    vocoder_noise = torch.randn(
        (1, mlx_model.vocoder.d_noise, decoded_torch.shape[-1] + 10),
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).numpy().astype(np.float32)

    with torch.no_grad():
        torch.manual_seed(0)
        wav_torch = torch_model.vocoder(torch.from_numpy(decoded_torch), y=None).cpu().numpy()

    wav_mlx = np.asarray(
        mlx_model.vocoder(mx.array(decoded_torch.astype(np.float32)), noise=mx.array(vocoder_noise, dtype=mx.float32), npad=10)
    )

    assert max_abs_diff(wav_torch, wav_mlx) < 5e-4
    assert mean_abs_diff(wav_torch, wav_mlx) < 5e-5
    assert si_sdr(wav_torch, wav_mlx) > 70.0

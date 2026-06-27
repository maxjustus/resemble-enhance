from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

from ..enhancer.download import download as download_source_checkpoint
from ..enhancer.inference import load_enhancer as load_torch_enhancer
from ..inference import remove_weight_norm_recursively
from .denoiser import DEFAULT_MLX_DENOISER_WEIGHTS, MLXDenoiser
from .enhancer import DEFAULT_MLX_ENHANCER_WEIGHTS, MLXEnhancer, NormalizerState
from .layers import (
    AMPBlock,
    AMPResidual,
    Conv1dCF,
    Conv2dCF,
    ConvTranspose1dCF,
    GroupNormCF,
    IRMAE,
    Identity,
    InstanceNorm1dCF,
    KernelPredictor,
    KernelPredictorResidual,
    LVCBlock,
    LVCConvBlock,
    LowPassFilter1d,
    PreactResBlock,
    ResBlock1d,
    SnakeBeta,
    UNet,
    UNetBlock,
    UnivNet,
    UpActDown,
    UpSample1d,
    WN,
    WNLayer,
)
from .ops import (
    torch_conv1d_weight_to_mlx,
    torch_conv2d_weight_to_mlx,
    torch_conv_transpose1d_weight_to_mlx,
)

DEFAULT_MLX_DIR = Path("artifacts/mlx")


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _set_array(obj, name: str, value, transform=None):
    arr = _to_np(value)
    if transform is not None:
        arr = transform(arr)
    setattr(obj, name, mx.array(arr.astype(np.float32)))


def _copy_conv1d(dst: Conv1dCF, src: torch.nn.Conv1d):
    _set_array(dst.layer, "weight", src.weight, transform=torch_conv1d_weight_to_mlx)
    if src.bias is not None:
        _set_array(dst.layer, "bias", src.bias)


def _copy_conv2d(dst: Conv2dCF, src: torch.nn.Conv2d):
    _set_array(dst.layer, "weight", src.weight, transform=torch_conv2d_weight_to_mlx)
    if src.bias is not None:
        _set_array(dst.layer, "bias", src.bias)


def _copy_conv_transpose1d(dst: ConvTranspose1dCF, src: torch.nn.ConvTranspose1d):
    _set_array(dst.layer, "weight", src.weight, transform=torch_conv_transpose1d_weight_to_mlx)
    if src.bias is not None:
        _set_array(dst.layer, "bias", src.bias)


def _copy_group_norm(dst: GroupNormCF, src: torch.nn.GroupNorm):
    _set_array(dst.layer, "weight", src.weight)
    _set_array(dst.layer, "bias", src.bias)


def _copy_instance_norm(dst: InstanceNorm1dCF, src: torch.nn.InstanceNorm1d):
    del dst, src


def _copy_snake_beta(dst: SnakeBeta, src):
    _set_array(dst, "log_alpha", src.log_alpha)
    _set_array(dst, "log_beta", src.log_beta)


def _copy_lowpass(dst: LowPassFilter1d, src):
    _set_array(dst, "filter", src.filter)


def _copy_upsample_filter(dst: UpSample1d, src):
    _set_array(dst, "filter", src.filter)


def _copy_layers(dst_layers, src_layers):
    for dst, src in zip(dst_layers, src_layers):
        _copy_module(dst, src)


def _copy_module(dst, src):
    if isinstance(dst, Conv1dCF):
        return _copy_conv1d(dst, src)
    if isinstance(dst, Conv2dCF):
        return _copy_conv2d(dst, src)
    if isinstance(dst, ConvTranspose1dCF):
        return _copy_conv_transpose1d(dst, src)
    if isinstance(dst, GroupNormCF):
        return _copy_group_norm(dst, src)
    if isinstance(dst, InstanceNorm1dCF):
        return _copy_instance_norm(dst, src)
    if isinstance(dst, SnakeBeta):
        return _copy_snake_beta(dst, src)
    if isinstance(dst, UpSample1d):
        return _copy_upsample_filter(dst, src)
    if isinstance(dst, LowPassFilter1d):
        return _copy_lowpass(dst, src)
    if isinstance(dst, ResBlock1d):
        return copy_resblock1d(dst, src)
    if isinstance(dst, AMPResidual):
        return copy_amp_residual(dst, src)
    if isinstance(dst, KernelPredictorResidual):
        return copy_kernel_predictor_residual(dst, src)
    if isinstance(dst, LVCConvBlock):
        return copy_lvc_conv_block(dst, src)
    if isinstance(dst, Identity) or isinstance(dst, nn.Module) and dst.__class__.__name__ in {"GELU", "LeakyReLU", "Tanh"}:
        return
    raise TypeError(f"Unhandled module copy: {type(dst)} <- {type(src)}")


def copy_preact_resblock(dst: PreactResBlock, src):
    _copy_layers(dst.layers, list(src))


def copy_unet_block(dst: UNetBlock, src):
    _copy_conv2d(dst.pre_conv, src.pre_conv)
    copy_preact_resblock(dst.res_block1, src.res_block1)
    copy_preact_resblock(dst.res_block2, src.res_block2)


def copy_unet(dst: UNet, src):
    _copy_conv2d(dst.input_proj, src.input_proj)
    for d, s in zip(dst.encoder_blocks, src.encoder_blocks):
        copy_unet_block(d, s)
    for d, s in zip(dst.middle_blocks, src.middle_blocks):
        copy_unet_block(d, s)
    for d, s in zip(dst.decoder_blocks, src.decoder_blocks):
        copy_unet_block(d, s)
    _copy_layers(dst.head, list(src.head))


def copy_resblock1d(dst: ResBlock1d, src):
    _copy_layers(dst.layers, list(src))


def copy_irmae(dst: IRMAE, src):
    _copy_layers(dst.encoder, list(src.encoder))
    _copy_layers(dst.decoder, list(src.decoder))
    _copy_layers(dst.head, list(src.head))


def copy_wn_layer(dst: WNLayer, src):
    if dst.gconv is not None and src.gconv is not None:
        _copy_conv1d(dst.gconv, src.gconv)
    if dst.lconv is not None and src.lconv is not None:
        _copy_conv1d(dst.lconv, src.lconv)
    _copy_conv1d(dst.dconv, src.dconv)
    _copy_conv1d(dst.out, src.out)


def copy_wn(dst: WN, src):
    _copy_conv1d(dst.start, src.start)
    if dst.local_norm is not None and src.local_norm is not None:
        _copy_instance_norm(dst.local_norm, src.local_norm)
    for d, s in zip(dst.layers, src.layers):
        copy_wn_layer(d, s)
    _copy_conv1d(dst.end, src.end)


def copy_amp_residual(dst: AMPResidual, src):
    _copy_conv1d(dst.conv1, src[0])
    copy_upactdown(dst.act, src[1])
    _copy_conv1d(dst.conv2, src[2])


def copy_amp_block(dst: AMPBlock, src):
    for d, s in zip(dst.layers, list(src)):
        copy_amp_residual(d, s)


def copy_upactdown(dst: UpActDown, src):
    _copy_snake_beta(dst.act, src.act)
    _copy_upsample_filter(dst.upsample, src.upsample)
    _copy_lowpass(dst.downsample.lowpass, src.downsample.lowpass)


def copy_kernel_predictor_residual(dst: KernelPredictorResidual, src):
    _copy_layers(dst.layers, list(src))


def copy_kernel_predictor(dst: KernelPredictor, src):
    _copy_layers(dst.input_conv, list(src.input_conv))
    for d, s in zip(dst.residual_convs, src.residual_convs):
        copy_kernel_predictor_residual(d, s)
    _copy_conv1d(dst.kernel_conv, src.kernel_conv)
    _copy_conv1d(dst.bias_conv, src.bias_conv)


def copy_lvc_conv_block(dst: LVCConvBlock, src):
    _copy_layers(dst.layers, list(src))


def copy_lvc_block(dst: LVCBlock, src):
    copy_kernel_predictor(dst.kernel_predictor, src.kernel_predictor)
    _copy_layers(dst.convt_pre, list(src.convt_pre))
    copy_amp_block(dst.amp_block, src.amp_block)
    for d, s in zip(dst.conv_blocks, src.conv_blocks):
        copy_lvc_conv_block(d, s)


def copy_univnet(dst: UnivNet, src):
    for d, s in zip(dst.blocks, src.blocks):
        copy_lvc_block(d, s)
    _copy_conv1d(dst.conv_pre, src.conv_pre)
    _copy_layers(dst.conv_post, list(src.conv_post))


def copy_normalizer(dst: NormalizerState, src):
    _set_array(dst, "running_mean_unsafe", src.running_mean_unsafe)
    _set_array(dst, "running_var_unsafe", src.running_var_unsafe)


def copy_mel_buffers(dst, src):
    _set_array(dst, "mel_window", src.mel_fn.melspec.spectrogram.window)
    _set_array(dst, "mel_filter_bank", src.mel_fn.melspec.mel_scale.fb)
    _set_array(dst, "mel_magnitude_min", src.mel_fn.stft_magnitude_min)


def copy_denoiser(dst: MLXDenoiser, src):
    copy_unet(dst.net, src.net)
    copy_mel_buffers(dst, src)


def copy_lcfm(dst, src):
    copy_irmae(dst.ae, src.ae)
    copy_wn(dst.cfm.net, src.cfm.net)
    dst.z_scale = src.z_scale


def copy_enhancer(dst: MLXEnhancer, src):
    copy_lcfm(dst.lcfm, src.lcfm)
    copy_univnet(dst.vocoder, src.vocoder)
    copy_denoiser(dst.denoiser, src.denoiser)
    copy_normalizer(dst.normalizer, src.normalizer)
    copy_mel_buffers(dst, src)


def get_source_run_dir(run_dir: str | Path | None = None) -> Path:
    return Path(download_source_checkpoint(run_dir))


def load_source_checkpoint(path: str | Path):
    return torch.load(Path(path), map_location="cpu")


def inspect_checkpoint_manifest(run_dir: str | Path | None = None):
    run_dir = get_source_run_dir(run_dir)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state = load_source_checkpoint(path)["module"]
    grouped = {}
    for key, value in state.items():
        if key.startswith("denoiser."):
            group = "denoiser"
        elif key.startswith("lcfm."):
            group = "enhancer"
        elif key.startswith("vocoder."):
            group = "vocoder"
        elif key.startswith("mel_fn."):
            group = "conditioning"
        elif key.startswith("normalizer."):
            group = "conditioning"
        else:
            group = "unknown"
        grouped.setdefault(group, []).append(
            {"key": key, "shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
        )
    return {
        "source_checkpoint": str(path),
        "components": grouped,
        "counts": {name: len(items) for name, items in grouped.items()},
    }


def write_manifest(path: str | Path, manifest: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def export_enhancer_weights(
    *,
    run_dir: str | Path | None = None,
    weights_path: str | Path = DEFAULT_MLX_ENHANCER_WEIGHTS,
    hparams_path: str | Path | None = None,
) -> Path:
    torch_model = load_torch_enhancer(run_dir, "cpu")
    remove_weight_norm_recursively(torch_model)
    torch_model.eval()

    del hparams_path
    mlx_model = MLXEnhancer(torch_model.hp)
    copy_enhancer(mlx_model, torch_model)
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    mlx_model.save_weights(str(weights_path))
    return weights_path


def export_denoiser_weights(
    *,
    run_dir: str | Path | None = None,
    weights_path: str | Path = DEFAULT_MLX_DENOISER_WEIGHTS,
    hparams_path: str | Path | None = None,
) -> Path:
    torch_model = load_torch_enhancer(run_dir, "cpu").denoiser
    torch_model.eval()
    del hparams_path
    mlx_model = MLXDenoiser(torch_model.hp)
    copy_denoiser(mlx_model, torch_model)
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    mlx_model.save_weights(str(weights_path))
    return weights_path


def checkpoint_prefix_counts(run_dir: str | Path | None = None):
    run_dir = get_source_run_dir(run_dir)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state = load_source_checkpoint(path)["module"]
    return Counter(key.split(".")[0] for key in state)

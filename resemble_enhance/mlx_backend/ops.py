from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np


def assert_shape(x, pattern: str, **dims):
    actual = tuple(x.shape)
    tokens = pattern.replace(",", " ").split()
    if len(tokens) != len(actual):
        raise AssertionError(f"Expected {pattern}, got {actual}")
    resolved = {}
    for token, size in zip(tokens, actual):
        if token == "_":
            continue
        if token.isdigit():
            if int(token) != size:
                raise AssertionError(f"Expected {pattern}, got {actual}")
            continue
        expected = dims.get(token)
        if expected is not None and expected != size:
            raise AssertionError(f"Expected {pattern} with {token}={expected}, got {actual}")
        prior = resolved.get(token)
        if prior is not None and prior != size:
            raise AssertionError(f"Expected repeated dim {token} in {pattern}, got {actual}")
        resolved[token] = size


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def to_mx(x, dtype=mx.float32) -> mx.array:
    if isinstance(x, mx.array):
        if x.dtype == dtype:
            return x
        return x.astype(dtype)
    return mx.array(np.asarray(x), dtype=dtype)


def to_mlx_audio_batch(wav, dtype=mx.float32) -> mx.array:
    wav = to_numpy(wav).astype(np.float32, copy=False)
    if wav.ndim == 1:
        wav = wav[None]
    return mx.array(wav, dtype=dtype)


def from_mlx_audio_batch(x) -> np.ndarray:
    x = to_numpy(x).astype(np.float32, copy=False)
    if x.ndim == 2 and x.shape[0] == 1:
        return x[0]
    return x


def torch_conv1d_weight_to_mlx(w):
    return np.asarray(w).transpose(0, 2, 1)


def torch_conv2d_weight_to_mlx(w):
    return np.asarray(w).transpose(0, 2, 3, 1)


def torch_conv_transpose1d_weight_to_mlx(w):
    return np.asarray(w).transpose(1, 2, 0)


def torch_conv_transpose2d_weight_to_mlx(w):
    return np.asarray(w).transpose(1, 2, 3, 0)


def row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    return tuple(strides)


def unfold_last(x: mx.array, size: int, step: int) -> mx.array:
    shape = tuple(x.shape)
    if size > shape[-1]:
        raise ValueError(f"Cannot unfold size={size} from shape={shape}")
    n = 1 + (shape[-1] - size) // step
    base_strides = row_major_strides(shape)
    view_shape = shape[:-1] + (n, size)
    view_strides = base_strides[:-1] + (step * base_strides[-1], base_strides[-1])
    return mx.as_strided(x, shape=view_shape, strides=view_strides)


def pad1d(x: mx.array, left: int, right: int, mode: str = "constant", value: float = 0.0) -> mx.array:
    if left == 0 and right == 0:
        return x
    if mode == "constant":
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode=mode, constant_values=value)
    if mode == "edge":
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode=mode)
    if mode != "reflect":
        raise ValueError(f"Unsupported pad mode: {mode}")
    if x.shape[-1] < 2:
        return pad1d(x, left, right, mode="edge")
    left_pad = x[..., 1 : left + 1][..., ::-1] if left else x[..., :0]
    right_pad = x[..., -right - 1 : -1][..., ::-1] if right else x[..., :0]
    return mx.concatenate([left_pad, x, right_pad], axis=-1)


def pad2d(x: mx.array, pad: tuple[int, int, int, int], mode: str = "constant", value: float = 0.0) -> mx.array:
    left, right, top, bottom = pad
    pads = [(0, 0)] * (x.ndim - 2) + [(top, bottom), (left, right)]
    if mode == "constant":
        return mx.pad(x, pads, mode=mode, constant_values=value)
    if mode == "edge":
        return mx.pad(x, pads, mode=mode)
    raise ValueError(f"Unsupported 2D pad mode: {mode}")


def same_padding(kernel_size: int, dilation: int = 1) -> tuple[int, int]:
    total = dilation * (kernel_size - 1)
    left = total // 2
    right = total - left
    return left, right


def flatten_tree(tree, prefix: str = "") -> list[tuple[str, mx.array]]:
    out = []
    if isinstance(tree, dict):
        for key, value in tree.items():
            name = f"{prefix}.{key}" if prefix else key
            out.extend(flatten_tree(value, name))
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            name = f"{prefix}.{i}" if prefix else str(i)
            out.extend(flatten_tree(value, name))
    else:
        out.append((prefix, tree))
    return out


def save_safetensors(path: str | Path, weights: list[tuple[str, mx.array]] | dict[str, mx.array]):
    if isinstance(weights, list):
        weights = {k: v for k, v in weights}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), weights)


def load_safetensors(path: str | Path) -> dict[str, mx.array]:
    loaded = mx.load(str(path))
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict from {path}, got {type(loaded)}")
    return loaded

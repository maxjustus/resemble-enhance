from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .ops import assert_shape, pad1d, pad2d, same_padding, unfold_last


def _run_layers(layers, x):
    for layer in layers:
        x = layer(x)
    return x


def _to_channel_last_1d(x: mx.array) -> mx.array:
    return mx.swapaxes(x, 1, 2)


def _from_channel_last_1d(x: mx.array) -> mx.array:
    return mx.swapaxes(x, 1, 2)


def _to_channel_last_2d(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def _from_channel_last_2d(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


class Identity(nn.Module):
    def __call__(self, x):
        return x


class Conv1dCF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | str = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "constant",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.padding_mode = padding_mode
        self.layer = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def _pad(self, x):
        if self.padding == "same":
            left, right = same_padding(self.kernel_size, self.layer.dilation)
            return pad1d(x, left, right, mode=self.padding_mode)
        if isinstance(self.padding, int) and self.padding > 0:
            return pad1d(x, self.padding, self.padding, mode=self.padding_mode)
        return x

    def __call__(self, x):
        x = self._pad(x)
        x = _to_channel_last_1d(x)
        x = self.layer(x)
        return _from_channel_last_1d(x)


class Conv2dCF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "constant",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.padding = padding
        self.padding_mode = padding_mode
        self.layer = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def _pad(self, x):
        ph, pw = self.padding
        if ph or pw:
            return pad2d(x, (pw, pw, ph, ph), mode=self.padding_mode)
        return x

    def __call__(self, x):
        x = self._pad(x)
        x = _to_channel_last_2d(x)
        x = self.layer(x)
        return _from_channel_last_2d(x)


class ConvTranspose1dCF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        output_padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.layer = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            output_padding=output_padding,
            bias=bias,
        )

    def __call__(self, x):
        x = _to_channel_last_1d(x)
        x = self.layer(x)
        return _from_channel_last_1d(x)


class GroupNormCF(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.layer = nn.GroupNorm(
            num_groups,
            num_channels,
            eps=eps,
            affine=True,
            pytorch_compatible=True,
        )

    def __call__(self, x):
        axes = list(range(x.ndim))
        axes.append(axes.pop(1))
        x = mx.transpose(x, axes)
        x = self.layer(x)
        axes = [0, x.ndim - 1] + list(range(1, x.ndim - 1))
        return mx.transpose(x, axes)


class InstanceNorm1dCF(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.layer = nn.InstanceNorm(num_channels, eps=eps)

    def __call__(self, x):
        x = _to_channel_last_1d(x)
        x = self.layer(x)
        return _from_channel_last_1d(x)


class Upsample1dCF(nn.Module):
    def __init__(self, scale_factor: float, mode: str = "nearest"):
        super().__init__()
        self.layer = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def __call__(self, x):
        x = _to_channel_last_1d(x)
        x = self.layer(x)
        return _from_channel_last_1d(x)


class Upsample2dCF(nn.Module):
    def __init__(self, scale_factor: float, mode: str = "nearest"):
        super().__init__()
        self.layer = nn.Upsample(scale_factor=scale_factor, mode=mode)

    def __call__(self, x):
        x = _to_channel_last_2d(x)
        x = self.layer(x)
        return _from_channel_last_2d(x)


class AvgPool1dCF(nn.Module):
    def __init__(self, kernel_size: int, stride: int):
        super().__init__()
        self.layer = nn.AvgPool1d(kernel_size, stride)

    def __call__(self, x):
        x = _to_channel_last_1d(x)
        x = self.layer(x)
        return _from_channel_last_1d(x)


class PreactResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layers = [
            GroupNormCF(dim // 16, dim),
            nn.GELU(),
            Conv2dCF(dim, dim, 3, padding=1),
            GroupNormCF(dim // 16, dim),
            nn.GELU(),
            Conv2dCF(dim, dim, 3, padding=1),
        ]

    def __call__(self, x):
        return x + _run_layers(self.layers, x)


class UNetBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None, scale_factor: float = 1.0):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.pre_conv = Conv2dCF(input_dim, output_dim, 3, padding=1)
        self.res_block1 = PreactResBlock(output_dim)
        self.res_block2 = PreactResBlock(output_dim)
        self.downsample = Identity()
        self.upsample = Identity()
        if scale_factor > 1:
            self.upsample = Upsample2dCF(scale_factor)
        elif scale_factor < 1:
            self.downsample = Upsample2dCF(scale_factor)

    def __call__(self, x, h=None):
        x = self.upsample(x)
        if h is not None:
            if tuple(x.shape) != tuple(h.shape):
                raise AssertionError(f"{x.shape} != {h.shape}")
            x = x + h
        x = self.pre_conv(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return self.downsample(x), x


class UNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 16, num_blocks: int = 4, num_middle_blocks: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_proj = Conv2dCF(input_dim, hidden_dim, 3, padding=1)
        self.encoder_blocks = [
            UNetBlock(hidden_dim * 2**i, hidden_dim * 2 ** (i + 1), scale_factor=0.5)
            for i in range(num_blocks)
        ]
        self.middle_blocks = [UNetBlock(hidden_dim * 2**num_blocks) for _ in range(num_middle_blocks)]
        self.decoder_blocks = [
            UNetBlock(hidden_dim * 2 ** (i + 1), hidden_dim * 2**i, scale_factor=2)
            for i in reversed(range(num_blocks))
        ]
        self.head = [
            Conv2dCF(hidden_dim, hidden_dim, 3, padding=1),
            nn.GELU(),
            Conv2dCF(hidden_dim, output_dim, 1),
        ]

    @property
    def scale_factor(self):
        return 2 ** len(self.encoder_blocks)

    def pad_to_fit(self, x):
        hpad = (self.scale_factor - x.shape[2] % self.scale_factor) % self.scale_factor
        wpad = (self.scale_factor - x.shape[3] % self.scale_factor) % self.scale_factor
        return pad2d(x, (0, wpad, 0, hpad))

    def __call__(self, x):
        shape = tuple(x.shape)
        x = self.pad_to_fit(x)
        x = self.input_proj(x)

        skips = []
        for block in self.encoder_blocks:
            x, s = block(x)
            skips.append(s)

        for block in self.middle_blocks:
            x, _ = block(x)

        for block, s in zip(self.decoder_blocks, reversed(skips)):
            x, _ = block(x, s)

        x = _run_layers(self.head, x)
        return x[..., : shape[2], : shape[3]]


class ResBlock1d(nn.Module):
    def __init__(self, channels: int, dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.layers = []
        for dilation in dilations:
            self.layers.extend(
                [
                    GroupNormCF(32, channels),
                    nn.GELU(),
                    Conv1dCF(channels, channels, 3, padding="same", dilation=dilation),
                ]
            )

    def __call__(self, x):
        return x + _run_layers(self.layers, x)


@dataclass
class IRMAEOutput:
    latent: mx.array
    decoded: mx.array | None


class IRMAE(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, latent_dim: int, hidden_dim: int = 1024, num_irms: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        self.encoder = [Conv1dCF(input_dim, hidden_dim, 3, padding="same")]
        self.encoder.extend(ResBlock1d(hidden_dim) for _ in range(4))
        self.encoder.extend(
            Conv1dCF(hidden_dim if i == 0 else latent_dim, latent_dim, 1, bias=False) for i in range(num_irms)
        )
        self.encoder.append(nn.Tanh())

        self.decoder = [Conv1dCF(latent_dim, hidden_dim, 3, padding="same")]
        self.decoder.extend(ResBlock1d(hidden_dim) for _ in range(4))
        self.decoder.append(Conv1dCF(hidden_dim, output_dim, 1))

        self.head = [
            Conv1dCF(output_dim, hidden_dim, 3, padding="same"),
            nn.GELU(),
            Conv1dCF(hidden_dim, input_dim, 1),
        ]

    def encode(self, x):
        return _run_layers(self.encoder, x)

    def decode(self, z):
        return _run_layers(self.decoder, z)

    def __call__(self, x, skip_decoding: bool = False):
        z = self.encode(x)
        decoded = None if skip_decoding else self.decode(z)
        return IRMAEOutput(latent=z, decoded=decoded)


def fused_tanh_sigmoid(h: mx.array) -> mx.array:
    a, b = mx.split(h, 2, axis=1)
    return mx.tanh(a) * mx.sigmoid(b)


class WNLayer(nn.Module):
    def __init__(self, hidden_dim: int, local_dim: int | None, global_dim: int | None, kernel_size: int, dilation: int):
        super().__init__()
        local_output_dim = hidden_dim * 2
        self.gconv = Conv1dCF(global_dim, hidden_dim, 1) if global_dim is not None else None
        self.lconv = Conv1dCF(local_dim, local_output_dim, 1) if local_dim is not None else None
        self.dconv = Conv1dCF(hidden_dim, local_output_dim, kernel_size, dilation=dilation, padding="same")
        self.out = Conv1dCF(hidden_dim, 2 * hidden_dim, 1)

    def __call__(self, z, l, g):
        identity = z
        if g is not None:
            if g.ndim == 2:
                g = g[:, :, None]
            z = z + self.gconv(g)
        z = self.dconv(z)
        if l is not None:
            z = z + self.lconv(l)
        z = fused_tanh_sigmoid(z)
        h = self.out(z)
        z_out, s = mx.split(h, 2, axis=1)
        return (z_out + identity) / math.sqrt(2.0), s


class WN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        local_dim: int | None = None,
        global_dim: int | None = None,
        n_layers: int = 30,
        kernel_size: int = 3,
        dilation_cycle: int = 5,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.start = Conv1dCF(input_dim, hidden_dim, 1)
        self.local_norm = InstanceNorm1dCF(local_dim) if local_dim is not None else None
        self.layers = [
            WNLayer(
                hidden_dim=hidden_dim,
                local_dim=local_dim,
                global_dim=global_dim,
                kernel_size=kernel_size,
                dilation=2 ** (i % dilation_cycle),
            )
            for i in range(n_layers)
        ]
        self.end = Conv1dCF(hidden_dim, output_dim, 1)

    def __call__(self, z, l=None, g=None):
        z = self.start(z)
        if l is not None and self.local_norm is not None:
            l = self.local_norm(l)
        skips = []
        for layer in self.layers:
            z, s = layer(z, l, g)
            skips.append(s)
        s = mx.stack(skips, axis=0).sum(axis=0) / math.sqrt(len(skips))
        return self.end(s)


class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, clamp: tuple[float, float] = (1e-2, 50.0)):
        super().__init__()
        self.log_alpha = mx.zeros((in_features,)) + math.log(alpha)
        self.log_beta = mx.zeros((in_features,)) + math.log(alpha)
        self.clamp = clamp

    def __call__(self, x):
        alpha = mx.clip(mx.exp(self.log_alpha), self.clamp[0], self.clamp[1])[None, :, None]
        beta = mx.clip(mx.exp(self.log_beta), self.clamp[0], self.clamp[1])[None, :, None]
        return x + (1.0 / beta) * mx.power(mx.sin(x * alpha), 2)


class UpSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.filter = mx.zeros((1, 1, self.kernel_size), dtype=mx.float32)

    def __call__(self, x):
        _, channels, _ = x.shape
        x = pad1d(x, self.pad, self.pad, mode="edge")
        x = _to_channel_last_1d(x)
        weight = mx.broadcast_to(mx.transpose(self.filter, (0, 2, 1)), (channels, self.kernel_size, 1))
        x = self.ratio * mx.conv_transpose1d(x, weight, stride=self.stride, groups=channels)
        x = _from_channel_last_1d(x)
        if self.pad_right == 0:
            return x[..., self.pad_left :]
        return x[..., self.pad_left : -self.pad_right]


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        *,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "edge",
        kernel_size: int = 12,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.filter = mx.zeros((1, 1, kernel_size), dtype=mx.float32)

    def __call__(self, x):
        _, channels, _ = x.shape
        if self.padding:
            x = pad1d(x, self.pad_left, self.pad_right, mode=self.padding_mode)
        x = _to_channel_last_1d(x)
        weight = mx.broadcast_to(mx.transpose(self.filter, (0, 2, 1)), (channels, self.kernel_size, 1))
        x = mx.conv1d(x, weight, stride=self.stride, groups=channels)
        return _from_channel_last_1d(x)


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None):
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(stride=ratio, kernel_size=self.kernel_size)

    def __call__(self, x):
        return self.lowpass(x)


class UpActDown(nn.Module):
    def __init__(self, act: nn.Module, up_ratio: int = 2, down_ratio: int = 2, up_kernel_size: int = 12, down_kernel_size: int = 12):
        super().__init__()
        self.act = act
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def __call__(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AMPResidual(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv1 = Conv1dCF(channels, channels, kernel_size, dilation=dilation, padding="same")
        self.act = UpActDown(SnakeBeta(channels))
        self.conv2 = Conv1dCF(channels, channels, kernel_size, padding="same")

    def __call__(self, x):
        return self.conv2(self.act(self.conv1(x)))


class AMPBlock(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 3, dilations: tuple[int, int, int] = (1, 3, 5)):
        super().__init__()
        self.layers = [AMPResidual(channels, kernel_size, d) for d in dilations]

    def __call__(self, x):
        return x + _run_layers(self.layers, x)


class KernelPredictorResidual(nn.Module):
    def __init__(self, channels: int, conv_size: int, negative_slope: float):
        super().__init__()
        padding = (conv_size - 1) // 2
        self.layers = [
            Identity(),
            Conv1dCF(channels, channels, conv_size, padding=padding),
            nn.LeakyReLU(negative_slope),
            Conv1dCF(channels, channels, conv_size, padding=padding),
            nn.LeakyReLU(negative_slope),
        ]

    def __call__(self, x):
        return _run_layers(self.layers, x)


class KernelPredictor(nn.Module):
    def __init__(
        self,
        cond_channels: int,
        conv_in_channels: int,
        conv_out_channels: int,
        conv_layers: int,
        *,
        conv_kernel_size: int = 3,
        kpnet_hidden_channels: int = 64,
        kpnet_conv_size: int = 3,
        negative_slope: float = 0.1,
    ):
        super().__init__()
        self.conv_in_channels = conv_in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_layers = conv_layers

        kernel_channels = conv_in_channels * conv_out_channels * conv_kernel_size * conv_layers
        bias_channels = conv_out_channels * conv_layers
        padding = (kpnet_conv_size - 1) // 2

        self.input_conv = [
            Conv1dCF(cond_channels, kpnet_hidden_channels, 5, padding=2),
            nn.LeakyReLU(negative_slope),
        ]
        self.residual_convs = [
            KernelPredictorResidual(kpnet_hidden_channels, kpnet_conv_size, negative_slope) for _ in range(3)
        ]
        self.kernel_conv = Conv1dCF(kpnet_hidden_channels, kernel_channels, kpnet_conv_size, padding=padding)
        self.bias_conv = Conv1dCF(kpnet_hidden_channels, bias_channels, kpnet_conv_size, padding=padding)

    def __call__(self, c):
        batch, _, cond_length = c.shape
        c = _run_layers(self.input_conv, c)
        for residual in self.residual_convs:
            c = c + residual(c)
        k = self.kernel_conv(c)
        b = self.bias_conv(c)
        kernels = mx.reshape(
            k,
            (
                batch,
                self.conv_layers,
                self.conv_in_channels,
                self.conv_out_channels,
                self.conv_kernel_size,
                cond_length,
            ),
        )
        bias = mx.reshape(
            b,
            (
                batch,
                self.conv_layers,
                self.conv_out_channels,
                cond_length,
            ),
        )
        return kernels, bias


class LVCConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, negative_slope: float):
        super().__init__()
        self.layers = [
            nn.LeakyReLU(negative_slope),
            Conv1dCF(channels, channels, kernel_size, dilation=dilation, padding="same"),
            nn.LeakyReLU(negative_slope),
        ]

    def __call__(self, x):
        return _run_layers(self.layers, x)


class LVCBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        stride: int,
        *,
        dilations: list[int] | tuple[int, ...] = (1, 3, 9, 27),
        lrelu_slope: float = 0.2,
        conv_kernel_size: int = 3,
        cond_hop_length: int = 256,
        kpnet_hidden_channels: int = 64,
        kpnet_conv_size: int = 3,
        add_extra_noise: bool = False,
        downsampling: bool = False,
    ):
        super().__init__()
        self.add_extra_noise = add_extra_noise
        self.cond_hop_length = cond_hop_length
        self.conv_layers = len(dilations)
        self.conv_kernel_size = conv_kernel_size
        self.kernel_predictor = KernelPredictor(
            cond_channels=cond_channels,
            conv_in_channels=in_channels,
            conv_out_channels=2 * in_channels,
            conv_layers=len(dilations),
            conv_kernel_size=conv_kernel_size,
            kpnet_hidden_channels=kpnet_hidden_channels,
            kpnet_conv_size=kpnet_conv_size,
            negative_slope=lrelu_slope,
        )

        if downsampling:
            self.convt_pre = [
                nn.LeakyReLU(lrelu_slope),
                Conv1dCF(in_channels, in_channels, 2 * stride + 1, padding="same"),
                AvgPool1dCF(stride, stride),
            ]
        elif stride == 1:
            self.convt_pre = [nn.LeakyReLU(lrelu_slope), Conv1dCF(in_channels, in_channels, 1)]
        else:
            self.convt_pre = [
                nn.LeakyReLU(lrelu_slope),
                ConvTranspose1dCF(
                    in_channels,
                    in_channels,
                    2 * stride,
                    stride=stride,
                    padding=stride // 2 + stride % 2,
                    output_padding=stride % 2,
                ),
            ]

        self.amp_block = AMPBlock(in_channels)
        self.conv_blocks = [LVCConvBlock(in_channels, conv_kernel_size, d, lrelu_slope) for d in dilations]

    def location_variable_convolution(self, x, kernel, bias, dilation: int = 1, hop_size: int = 256):
        batch, _, in_length = x.shape
        _, _, out_channels, kernel_size, kernel_length = kernel.shape
        expected = kernel_length * hop_size
        if in_length != expected:
            raise AssertionError(f"length of (x, kernel) is not matched, {in_length} != {kernel_length} * {hop_size}")

        padding = dilation * ((kernel_size - 1) // 2)
        x = pad1d(x, padding, padding)
        x = unfold_last(x, hop_size + 2 * padding, hop_size)
        if hop_size < dilation:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, dilation)], mode="constant")
        x = unfold_last(x, dilation, dilation)
        x = x[..., :hop_size]
        x = mx.transpose(x, (0, 1, 2, 4, 3))
        x = unfold_last(x, kernel_size, 1)
        o = mx.einsum("bildsk,biokl->bolsd", x, kernel)
        o = o + bias[:, :, :, None, None]
        return mx.reshape(o, (batch, out_channels, -1))

    def __call__(self, x, c):
        _, in_channels, _ = x.shape
        x = _run_layers(self.convt_pre, x)
        x = self.amp_block(x)
        kernels, bias = self.kernel_predictor(c)
        for i, conv in enumerate(self.conv_blocks):
            output = conv(x)
            k = kernels[:, i]
            b = bias[:, i]
            output = self.location_variable_convolution(output, k, b, hop_size=self.cond_hop_length)
            left, right = mx.split(output, 2, axis=1)
            x = x + mx.sigmoid(left) * mx.tanh(right)
        return x


class UnivNet(nn.Module):
    def __init__(self, hp, d_input: int):
        super().__init__()
        self.hp = hp
        self.d_input = d_input
        self.blocks = [
            LVCBlock(
                self.hp.univnet_nc,
                d_input,
                stride=stride,
                dilations=(1, 3, 9, 27),
                cond_hop_length=hop_length,
                kpnet_conv_size=3,
            )
            for stride, hop_length in zip(self.strides, self.cumprod_strides)
        ]
        self.conv_pre = Conv1dCF(self.d_noise, self.nc, 7, padding=3, padding_mode="reflect")
        self.conv_post = [
            nn.LeakyReLU(0.2),
            Conv1dCF(self.nc, 1, 7, padding=3, padding_mode="reflect"),
            nn.Tanh(),
        ]

    @property
    def d_noise(self):
        return 128

    @property
    def strides(self):
        return [7, 5, 4, 3]

    @property
    def cumprod_strides(self):
        out = []
        value = 1
        for stride in self.strides:
            value *= stride
            out.append(value)
        return out

    @property
    def nc(self):
        return self.hp.univnet_nc

    @property
    def scale_factor(self):
        return self.hp.hop_size

    def __call__(self, x, *, noise: mx.array | None = None, npad: int = 10):
        assert_shape(x, "b c t", c=self.d_input)
        x = pad1d(x, 0, npad)
        if noise is None:
            noise = mx.random.normal((x.shape[0], self.d_noise, x.shape[2]), dtype=x.dtype)
        elif noise.shape[2] != x.shape[2]:
            raise AssertionError(f"Expected noise length {x.shape[2]}, got {noise.shape[2]}")
        z = self.conv_pre(noise)
        for block in self.blocks:
            z = block(z, x)
        z = _run_layers(self.conv_post, z)
        z = z[..., : -self.scale_factor * npad]
        return mx.squeeze(z, axis=1)

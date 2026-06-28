from __future__ import annotations

import logging
from pathlib import Path

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..hparams import HParams
from .audio import (
    istft_torch_like,
    mel_spectrogram_mlx,
    merge_chunks,
    normalize_waveform,
    periodic_hann_window,
    resample_audio,
    stft_torch_like,
)
from .layers import UNet
from .ops import from_mlx_audio_batch, to_mlx_audio_batch

logger = logging.getLogger(__name__)

DEFAULT_MLX_DENOISER_WEIGHTS = Path("artifacts/mlx/denoiser.safetensors")


def _normalize_batch(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x), axis=-1, keepdims=True).clip(min=1e-7)
    return (x / peak).astype(np.float32)


class MLXDenoiser(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.net = UNet(input_dim=3, output_dim=3)
        self.mel_window = mx.array(periodic_hann_window(hp.win_size), dtype=mx.float32)
        mel_fb = librosa.filters.mel(
            sr=hp.wav_rate,
            n_fft=hp.n_fft,
            n_mels=hp.num_mels,
            fmin=0.0,
            fmax=hp.wav_rate / 2.0,
            htk=False,
            norm="slaney",
        ).T
        self.mel_filter_bank = mx.array(mel_fb.astype(np.float32))
        self.mel_magnitude_min = mx.array([hp.stft_magnitude_min], dtype=mx.float32)

    @property
    def stft_cfg(self) -> dict:
        hop_size = self.hp.hop_size
        return dict(hop_length=hop_size, n_fft=hop_size * 4, win_length=hop_size * 4)

    @property
    def n_fft(self):
        return self.stft_cfg["n_fft"]

    @property
    def eps(self):
        return 1e-7

    def to_mel(self, x, drop_last: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        out = []
        for wav in x:
            mel = mel_spectrogram_mlx(
                wav,
                n_fft=self.hp.n_fft,
                hop_length=self.hp.hop_size,
                win_length=self.hp.win_size,
                preemphasis_coeff=self.hp.preemphasis,
                mel_filter_bank=np.asarray(self.mel_filter_bank, dtype=np.float32),
                magnitude_min=float(np.asarray(self.mel_magnitude_min)[0]),
                window=np.asarray(self.mel_window, dtype=np.float32),
            )
            if drop_last:
                mel = mel[..., :-1]
            out.append(mel)
        return np.stack(out, axis=0).astype(np.float32)

    def _stft(self, x: np.ndarray):
        mags = []
        coss = []
        sins = []
        window = periodic_hann_window(self.stft_cfg["win_length"])
        for wav in x:
            spec = stft_torch_like(
                wav,
                n_fft=self.stft_cfg["n_fft"],
                hop_length=self.stft_cfg["hop_length"],
                win_length=self.stft_cfg["win_length"],
                pad_mode="reflect",
                center=True,
                window=window,
                drop_last=True,
            )
            mag = np.abs(spec).astype(np.float32)
            phase = np.angle(spec)
            mags.append(mag)
            coss.append(np.cos(phase).astype(np.float32))
            sins.append(np.sin(phase).astype(np.float32))
        return np.stack(mags), np.stack(coss), np.stack(sins)

    def _istft(self, mag: np.ndarray, cos: np.ndarray, sin: np.ndarray, length: int):
        outputs = []
        window = periodic_hann_window(self.stft_cfg["win_length"])
        for i in range(len(mag)):
            real = mag[i] * cos[i]
            imag = mag[i] * sin[i]
            spec = real + 1j * imag
            spec = np.pad(spec, ((0, 0), (0, 1)), mode="edge")
            wav = istft_torch_like(
                spec,
                n_fft=self.stft_cfg["n_fft"],
                hop_length=self.stft_cfg["hop_length"],
                win_length=self.stft_cfg["win_length"],
                center=True,
                window=window,
                length=length,
            )
            wav = np.nan_to_num(wav, copy=False)
            outputs.append(wav.astype(np.float32))
        return np.stack(outputs)

    def _magphase(self, real: mx.array, imag: mx.array):
        mag = mx.sqrt(mx.power(real, 2) + mx.power(imag, 2) + self.eps)
        cos = real / mag
        sin = imag / mag
        return mag, cos, sin

    def _predict(self, mag: np.ndarray, cos: np.ndarray, sin: np.ndarray):
        x = mx.stack(
            [
                mx.array(mag, dtype=mx.float32),
                mx.array(cos, dtype=mx.float32),
                mx.array(sin, dtype=mx.float32),
            ],
            axis=1,
        )
        out = self.net(x)
        mag_mask, real, imag = mx.split(out, 3, axis=1)
        mag_mask = mx.sigmoid(mx.squeeze(mag_mask, axis=1))
        real = mx.tanh(mx.squeeze(real, axis=1))
        imag = mx.tanh(mx.squeeze(imag, axis=1))
        _, cos_res, sin_res = self._magphase(real, imag)
        return mag_mask, sin_res, cos_res

    def _separate(self, mag, cos, sin, mag_mask, cos_res, sin_res):
        sep_mag = mx.maximum(mag * mag_mask, 0.0)
        sep_cos = cos * cos_res - sin * sin_res
        sep_sin = sin * cos_res + cos * sin_res
        return sep_mag, sep_cos, sep_sin

    def __call__(self, x):
        x = from_mlx_audio_batch(x) if isinstance(x, mx.array) else np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        x = _normalize_batch(x)
        mag, cos, sin = self._stft(x)
        mag_mask, sin_res, cos_res = self._predict(mag, cos, sin)
        sep_mag, sep_cos, sep_sin = self._separate(
            mx.array(mag, dtype=mx.float32),
            mx.array(cos, dtype=mx.float32),
            mx.array(sin, dtype=mx.float32),
            mag_mask,
            cos_res,
            sin_res,
        )
        mx.eval(sep_mag, sep_cos, sep_sin)
        out = self._istft(np.asarray(sep_mag), np.asarray(sep_cos), np.asarray(sep_sin), length=x.shape[-1])
        if out.shape[-1] < x.shape[-1]:
            out = np.pad(out, ((0, 0), (0, x.shape[-1] - out.shape[-1])))
        return out.astype(np.float32)


def _resolve_weights_path(weights_path: str | Path | None) -> Path:
    path = DEFAULT_MLX_DENOISER_WEIGHTS if weights_path is None else Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MLX denoiser weights at {path}. Run scripts/convert_resemble_checkpoint_to_mlx.py first."
        )
    return path


def load_denoiser(weights_path: str | Path | None = None, *, hparams_path: str | Path | None = None) -> MLXDenoiser:
    hp = HParams.from_yaml(Path(hparams_path)) if hparams_path is not None else HParams()
    model = MLXDenoiser(hp)
    model.load_weights(str(_resolve_weights_path(weights_path)), strict=False)
    model.eval()
    return model


def denoise_audio_with_model(
    model: MLXDenoiser,
    wav: np.ndarray,
    sample_rate: int,
    *,
    chunk_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    wav = resample_audio(np.asarray(wav, dtype=np.float32), sample_rate, model.hp.wav_rate)
    sr = model.hp.wav_rate

    chunk_seconds = 30.0 if chunk_seconds is None else chunk_seconds
    overlap_seconds = 1.0 if overlap_seconds is None else overlap_seconds

    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length

    chunks = []
    for start in range(0, wav.shape[-1], hop_length):
        chunk = wav[start : start + chunk_length]
        normed, peak = normalize_waveform(chunk)
        normed = np.pad(normed, (0, 441))
        out = model(normed)[0][: len(chunk)] * peak
        chunks.append(out.astype(np.float32))

    hwav = merge_chunks(chunks, chunk_length, hop_length, sr=sr, length=wav.shape[-1])
    return hwav.astype(np.float32), sr


def denoise_audio_mlx(
    wav: np.ndarray,
    sample_rate: int,
    *,
    weights_path: str | Path | None = None,
    hparams_path: str | Path | None = None,
    chunk_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    model = load_denoiser(weights_path, hparams_path=hparams_path)
    return denoise_audio_with_model(
        model,
        wav,
        sample_rate,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )

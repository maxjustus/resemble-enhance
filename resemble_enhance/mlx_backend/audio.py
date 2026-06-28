from __future__ import annotations

import math
from pathlib import Path

import librosa
import mlx.core as mx
import numpy as np
import scipy.signal
import soundfile as sf

from .parity import mean_abs_diff, max_abs_diff


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav, sr


def save_audio(path: str | Path, wav: np.ndarray, sr: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)


def resample_audio(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if orig_sr == target_sr:
        return wav
    return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr, res_type="soxr_hq").astype(np.float32)


def normalize_waveform(wav: np.ndarray, eps: float = 1e-7) -> tuple[np.ndarray, float]:
    wav = np.asarray(wav, dtype=np.float32)
    peak = float(np.clip(np.max(np.abs(wav)), eps, None))
    return wav / peak, peak


def preemphasis(wav: np.ndarray, coeff: float) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32)
    if coeff <= 0:
        return wav
    padded = np.pad(wav, (1, 0), mode="constant")
    return (padded[1:] - coeff * padded[:-1]).astype(np.float32)


def periodic_hann_window(win_length: int) -> np.ndarray:
    return scipy.signal.windows.hann(win_length, sym=False).astype(np.float32)


def stft_torch_like(
    wav: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    pad_mode: str = "reflect",
    center: bool = True,
    window: np.ndarray | None = None,
    drop_last: bool = False,
) -> np.ndarray:
    if window is None:
        window = periodic_hann_window(win_length)
    spec = librosa.stft(
        np.asarray(wav, dtype=np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    if drop_last:
        spec = spec[..., :-1]
    return spec


def istft_torch_like(
    spec: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    center: bool = True,
    window: np.ndarray | None = None,
    length: int | None = None,
) -> np.ndarray:
    if window is None:
        window = periodic_hann_window(win_length)
    return librosa.istft(
        spec,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    ).astype(np.float32)


def mel_spectrogram(
    wav: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    preemphasis_coeff: float,
    mel_filter_bank: np.ndarray,
    magnitude_min: float,
    window: np.ndarray | None = None,
) -> np.ndarray:
    if window is None:
        window = periodic_hann_window(win_length)
    wav = preemphasis(wav, preemphasis_coeff)
    spec = stft_torch_like(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode="constant",
        center=True,
        window=window,
        drop_last=False,
    )
    mag = np.abs(spec).astype(np.float32)
    mel = mel_filter_bank.T @ mag
    mel = np.maximum(mel, magnitude_min)
    mel = np.log10(mel) * 20.0
    min_level_db = 20.0 * math.log10(magnitude_min)
    mel = (mel - min_level_db) / (-min_level_db + 15.0)
    return mel.astype(np.float32)


def mel_spectrogram_mlx(
    wav: np.ndarray | mx.array,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int,
    preemphasis_coeff: float,
    mel_filter_bank: np.ndarray | mx.array,
    magnitude_min: float,
    window: np.ndarray | mx.array | None = None,
) -> np.ndarray:
    if window is None:
        window = periodic_hann_window(win_length)

    x = mx.array(wav, dtype=mx.float32)
    if preemphasis_coeff > 0:
        x = mx.concatenate([x[:1], x[1:] - preemphasis_coeff * x[:-1]], axis=0)

    fft_window = mx.array(window, dtype=mx.float32)
    if win_length != n_fft:
        left = (n_fft - win_length) // 2
        right = n_fft - win_length - left
        fft_window = mx.pad(fft_window, [(left, right)], mode="constant")

    pad = n_fft // 2
    x = mx.pad(x, [(pad, pad)], mode="constant")
    n_frames = 1 + (x.shape[0] - n_fft) // hop_length
    frames = mx.as_strided(x, shape=(n_frames, n_fft), strides=(hop_length, 1))
    spec = mx.fft.rfft(frames * fft_window, n=n_fft, axis=-1)
    mag = mx.transpose(mx.abs(spec), (1, 0))

    mel = mx.array(mel_filter_bank, dtype=mx.float32).T @ mag
    mel = mx.maximum(mel, magnitude_min)
    mel = mx.log10(mel) * 20.0
    min_level_db = 20.0 * math.log10(magnitude_min)
    mel = (mel - min_level_db) / (-min_level_db + 15.0)
    mx.eval(mel)
    return np.asarray(mel, dtype=np.float32)


def compute_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y))))


def _offset_mel(chunk: np.ndarray, sr: int) -> np.ndarray:
    hop_length = sr // 200
    win_length = hop_length * 4
    n_fft = 1 << (win_length - 1).bit_length()
    mel = librosa.feature.melspectrogram(
        y=np.asarray(chunk, dtype=np.float32),
        sr=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=80,
        fmin=0.0,
        fmax=sr / 2.0,
        power=1.0,
    )
    return np.log1p(mel).astype(np.float32)


def compute_offset(chunk1: np.ndarray, chunk2: np.ndarray, sr: int = 44100) -> int:
    hop_length = sr // 200
    spec1 = _offset_mel(chunk1, sr)
    spec2 = _offset_mel(chunk2, sr)
    corr = compute_corr(spec1, spec2).mean(axis=0)
    argmax = int(np.argmax(corr))
    if argmax > len(corr) // 2:
        argmax -= len(corr)
    return -argmax * hop_length


def merge_chunks(
    chunks: list[np.ndarray],
    chunk_length: int,
    hop_length: int,
    *,
    sr: int = 44100,
    length: int | None = None,
) -> np.ndarray:
    signal_length = (len(chunks) - 1) * hop_length + chunk_length
    overlap_length = chunk_length - hop_length
    signal = np.zeros(signal_length, dtype=np.float32)

    fadein = np.concatenate([np.linspace(0, 1, overlap_length, dtype=np.float32), np.ones(hop_length, dtype=np.float32)])
    fadeout = np.concatenate([np.ones(hop_length, dtype=np.float32), np.linspace(1, 0, overlap_length, dtype=np.float32)])

    for i, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=np.float32)
        if len(chunk) < chunk_length:
            chunk = np.pad(chunk, (0, chunk_length - len(chunk)))
        start = i * hop_length
        end = start + chunk_length

        if i > 0:
            pre_region = chunks[i - 1][-overlap_length:]
            cur_region = chunk[:overlap_length]
            offset = compute_offset(pre_region, cur_region, sr=sr)
            start -= offset
            end -= offset

        if i == 0:
            chunk = chunk * fadeout
        elif i == len(chunks) - 1:
            chunk = chunk * fadein
        else:
            chunk = chunk * fadein * fadeout

        left_clip = max(0, -start)
        right_clip = max(0, end - signal_length)
        if left_clip or right_clip:
            chunk = chunk[left_clip : len(chunk) - right_clip]
            start = max(0, start)
            end = min(signal_length, end)
        signal[start:end] += chunk[: end - start]

    if length is not None:
        signal = signal[:length]
    return signal.astype(np.float32)


def preprocessing_report(reference: np.ndarray, candidate: np.ndarray) -> dict:
    return {
        "shape_reference": list(reference.shape),
        "shape_candidate": list(candidate.shape),
        "max_abs_diff": max_abs_diff(reference, candidate),
        "mean_abs_diff": mean_abs_diff(reference, candidate),
    }

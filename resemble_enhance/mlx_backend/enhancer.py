from __future__ import annotations

import logging
import math
from pathlib import Path

import librosa
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import scipy.optimize

from ..enhancer.download import download as download_source_checkpoint
from ..enhancer.hparams import HParams
from .audio import mel_spectrogram, merge_chunks, normalize_waveform, periodic_hann_window, resample_audio
from .denoiser import MLXDenoiser
from .layers import IRMAE, UnivNet, WN

logger = logging.getLogger(__name__)

DEFAULT_MLX_ENHANCER_WEIGHTS = Path("artifacts/mlx/enhancer_stage2.safetensors")


class NormalizerState(nn.Module):
    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps
        self.running_mean_unsafe = mx.array([0.0], dtype=mx.float32)
        self.running_var_unsafe = mx.array([1.0], dtype=mx.float32)

    @property
    def running_mean(self):
        return self.running_mean_unsafe

    @property
    def running_std(self):
        return mx.sqrt(self.running_var_unsafe + self.eps)

    def __call__(self, x, update: bool = False):
        del update
        return (x - self.running_mean) / self.running_std

    def inverse(self, x):
        return x * self.running_std + self.running_mean


class Solver:
    def __init__(self, method: str = "midpoint", nfe: int = 32, time_mapping_divisor: int = 4):
        self.configurate_(nfe=nfe, method=method)
        self._time_mapping_divisor = time_mapping_divisor

    def configurate_(self, nfe: int | None = None, method: str | None = None):
        if nfe is not None:
            self.nfe = nfe
        if method is not None:
            self.method = method
        if self.nfe == 1 and self.method in {"midpoint", "rk4"}:
            self.method = "euler"

    @staticmethod
    def exponential_decay_mapping(t, n: int = 4):
        def h(x, a):
            return (a**x - 1) / (a - 1)

        a = float(scipy.optimize.fsolve(lambda a_: h(1 / n, a_) - 0.5, x0=0)[0])
        return h(np.asarray(t, dtype=np.float64), a=a).astype(np.float32)

    @property
    def time_mapping(self):
        return lambda t: self.exponential_decay_mapping(t, n=self._time_mapping_divisor)

    @property
    def n_steps(self):
        n = self.nfe
        if self.method == "midpoint":
            n //= 2
        elif self.method == "rk4":
            n //= 4
        return n

    @staticmethod
    def _euler_step(t, psi_t, dt, f):
        return psi_t + dt * f(t=t, psi_t=psi_t, dt=dt)

    @staticmethod
    def _midpoint_step(t, psi_t, dt, f):
        return psi_t + dt * f(t=t + dt / 2, psi_t=psi_t + dt * f(t=t, psi_t=psi_t, dt=dt) / 2, dt=dt)

    @staticmethod
    def _rk4_step(t, psi_t, dt, f):
        k1 = f(t=t, psi_t=psi_t, dt=dt)
        k2 = f(t=t + dt / 2, psi_t=psi_t + dt * k1 / 2, dt=dt)
        k3 = f(t=t + dt / 2, psi_t=psi_t + dt * k2 / 2, dt=dt)
        k4 = f(t=t + dt, psi_t=psi_t + dt * k3, dt=dt)
        return psi_t + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    @property
    def _step(self):
        if self.method == "euler":
            return self._euler_step
        if self.method == "midpoint":
            return self._midpoint_step
        if self.method == "rk4":
            return self._rk4_step
        raise ValueError(f"Unknown method: {self.method}")

    def __call__(self, f, psi0: mx.array, t0: float = 0.0, t1: float = 1.0) -> mx.array:
        ts = self.time_mapping(np.linspace(t0, t1, self.n_steps + 1))
        psi_t = psi0
        for i in range(self.n_steps):
            dt = float(ts[i + 1] - ts[i])
            t = float(ts[i])
            psi_t = self._step(t=t, psi_t=psi_t, dt=dt, f=f)
        return psi_t


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_embed: int):
        super().__init__()
        self.d_embed = d_embed
        if d_embed % 2 != 0:
            raise ValueError("d_embed must be even")

    def __call__(self, t):
        t = mx.expand_dims(t, -1)
        p = mx.linspace(0, 4, self.d_embed // 2, dtype=t.dtype)
        while p.ndim < t.ndim:
            p = mx.expand_dims(p, 0)
        sin = mx.sin(t * mx.power(10.0, p))
        cos = mx.cos(t * mx.power(10.0, p))
        return mx.concatenate([sin, cos], axis=-1)


class CFM(nn.Module):
    def __init__(
        self,
        *,
        cond_dim: int,
        output_dim: int,
        time_emb_dim: int = 128,
        solver_nfe: int = 32,
        solver_method: str = "midpoint",
        time_mapping_divisor: int = 4,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.solver = Solver(solver_method, solver_nfe, time_mapping_divisor)
        self.emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.net = WN(
            input_dim=output_dim,
            output_dim=output_dim,
            local_dim=cond_dim,
            global_dim=time_emb_dim,
        )

    @property
    def sigma(self):
        return 1e-4

    def _sample_psi0(self, x: mx.array, rng: np.random.Generator):
        shape = list(x.shape)
        shape[1] = self.output_dim
        return mx.array(rng.standard_normal(shape, dtype=np.float32))

    def _to_v(self, *, psi_t: mx.array, x: mx.array, t: float | mx.array):
        if isinstance(t, (float, int)):
            t = mx.full((psi_t.shape[0],), float(t), dtype=psi_t.dtype)
        t = mx.clip(t, 0.0, 1.0)
        g = self.emb(t)
        return self.net(psi_t, l=x, g=g)

    def sample(self, x: mx.array, *, psi0: mx.array | None = None, t0: float = 0.0, rng: np.random.Generator):
        if psi0 is None:
            psi0 = self._sample_psi0(x, rng)
        f = lambda t, psi_t, dt: self._to_v(psi_t=psi_t, t=t, x=x)
        return self.solver(f=f, psi0=psi0, t0=t0)

    def __call__(self, x: mx.array, *, psi0: mx.array | None = None, t0: float = 0.0, rng: np.random.Generator):
        return self.sample(x, psi0=psi0, t0=t0, rng=rng)


class LCFM(nn.Module):
    def __init__(self, ae: IRMAE, cfm: CFM, *, z_scale: float = 1.0):
        super().__init__()
        self.ae = ae
        self.cfm = cfm
        self.z_scale = z_scale
        self._mode = "ae"
        self._eval_tau = 0.5

    def set_mode_(self, mode: str):
        self._mode = mode

    def eval_tau_(self, tau: float):
        self._eval_tau = tau

    def _scale(self, z: mx.array):
        return z * self.z_scale

    def _unscale(self, z: mx.array):
        return z / self.z_scale

    def __call__(self, x: mx.array, *, y: mx.array | None = None, psi0: mx.array | None = None, rng: np.random.Generator):
        if psi0 is not None:
            psi0 = self._scale(self.ae.encode(psi0))
            psi0 = self._eval_tau * mx.array(rng.standard_normal(psi0.shape, dtype=np.float32)) + (1 - self._eval_tau) * psi0

        if y is None:
            if self._mode == "ae":
                z = self.ae.encode(x)
            else:
                z = self._unscale(self.cfm(x, psi0=psi0, rng=rng))
            return self.ae.decode(z)

        return self.ae(y).decoded


class MLXEnhancer(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        n_mels = hp.num_mels
        vocoder_input_dim = n_mels + hp.vocoder_extra_dim
        latent_dim = hp.lcfm_latent_dim

        self.lcfm = LCFM(
            IRMAE(input_dim=n_mels, output_dim=vocoder_input_dim, latent_dim=latent_dim),
            CFM(
                cond_dim=n_mels,
                output_dim=latent_dim,
                solver_nfe=hp.cfm_solver_nfe,
                solver_method=hp.cfm_solver_method,
                time_mapping_divisor=hp.cfm_time_mapping_divisor,
            ),
            z_scale=hp.lcfm_z_scale,
        )
        self.lcfm.set_mode_(hp.lcfm_training_mode)

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

        self.vocoder = UnivNet(hp, vocoder_input_dim)
        self.denoiser = MLXDenoiser(hp)
        self.normalizer = NormalizerState()
        self._eval_lambd = 0.0

    def configurate_(self, *, nfe: int, solver: str, lambd: float, tau: float):
        self.lcfm.cfm.solver.configurate_(nfe=nfe, method=solver)
        self.lcfm.eval_tau_(tau)
        self._eval_lambd = lambd

    def to_mel(self, x, drop_last: bool = True):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        out = []
        for wav in x:
            mel = mel_spectrogram(
                wav,
                sample_rate=self.hp.wav_rate,
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

    def _may_denoise(self, x: np.ndarray):
        if self.hp.lcfm_training_mode == "cfm":
            return self.denoiser(x)
        return x

    def __call__(self, x, *, seed: int = 0):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        x = (x / np.max(np.abs(x), axis=-1, keepdims=True).clip(min=1e-7)).astype(np.float32)

        x_mel_original = mx.array(self.to_mel(x), dtype=mx.float32)
        x_mel_original = self.normalizer(x_mel_original, update=False)

        if self.hp.lcfm_training_mode == "cfm" and self._eval_lambd != 0:
            x_denoised = self._may_denoise(x)
            x_mel_denoised = mx.array(self.to_mel(x_denoised), dtype=mx.float32)
            x_mel_denoised = self.normalizer(x_mel_denoised, update=False)
            x_mel_denoised = self._eval_lambd * x_mel_denoised + (1 - self._eval_lambd) * x_mel_original
        else:
            x_mel_denoised = x_mel_original

        rng = np.random.default_rng(seed)
        if self.hp.force_gaussian_prior:
            decoded = self.lcfm(x_mel_denoised, psi0=None, rng=rng)
        else:
            decoded = self.lcfm(x_mel_denoised, psi0=x_mel_original, rng=rng)

        vocoder_npad = 10
        noise = mx.array(
            rng.standard_normal((decoded.shape[0], self.vocoder.d_noise, decoded.shape[2] + vocoder_npad), dtype=np.float32),
            dtype=mx.float32,
        )
        out = self.vocoder(decoded, noise=noise, npad=vocoder_npad)
        mx.eval(out)
        return np.asarray(out, dtype=np.float32)


def _resolve_weights_path(weights_path: str | Path | None) -> Path:
    path = DEFAULT_MLX_ENHANCER_WEIGHTS if weights_path is None else Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MLX enhancer weights at {path}. Run scripts/convert_resemble_checkpoint_to_mlx.py first."
        )
    return path


def load_enhancer(weights_path: str | Path | None = None, *, hparams_path: str | Path | None = None) -> MLXEnhancer:
    if hparams_path is None:
        hp = HParams.load(Path(download_source_checkpoint(None)))
    else:
        hp = HParams.from_yaml(Path(hparams_path))
    model = MLXEnhancer(hp)
    model.load_weights(str(_resolve_weights_path(weights_path)), strict=False)
    model.eval()
    return model


def enhance_audio_mlx(
    wav: np.ndarray,
    sample_rate: int,
    *,
    weights_path: str | Path | None = None,
    hparams_path: str | Path | None = None,
    solver: str | None = None,
    nfe: int | None = None,
    lambd: float = 0.5,
    tau: float = 0.5,
    seed: int = 0,
    chunk_seconds: float | None = None,
    overlap_seconds: float | None = None,
) -> tuple[np.ndarray, int]:
    model = load_enhancer(weights_path, hparams_path=hparams_path)
    model.configurate_(
        nfe=model.hp.cfm_solver_nfe if nfe is None else nfe,
        solver=model.hp.cfm_solver_method if solver is None else solver,
        lambd=lambd,
        tau=tau,
    )

    wav = resample_audio(np.asarray(wav, dtype=np.float32), sample_rate, model.hp.wav_rate)
    sr = model.hp.wav_rate
    chunk_seconds = 30.0 if chunk_seconds is None else chunk_seconds
    overlap_seconds = 1.0 if overlap_seconds is None else overlap_seconds

    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length

    chunks = []
    for idx, start in enumerate(range(0, wav.shape[-1], hop_length)):
        chunk = wav[start : start + chunk_length]
        normed, peak = normalize_waveform(chunk)
        normed = np.pad(normed, (0, 441))
        out = model(normed, seed=seed + idx)[0][: len(chunk)] * peak
        chunks.append(out.astype(np.float32))

    hwav = merge_chunks(chunks, chunk_length, hop_length, sr=sr, length=wav.shape[-1])
    return hwav.astype(np.float32), sr

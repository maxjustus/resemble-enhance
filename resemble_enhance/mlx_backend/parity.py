from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def max_abs_diff(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.max(np.abs(a - b)))


def mean_abs_diff(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean(np.abs(a - b)))


def si_sdr(reference, estimate, eps: float = 1e-8) -> float:
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    if reference.shape != estimate.shape:
        n = min(len(reference), len(estimate))
        reference = reference[:n]
        estimate = estimate[:n]
    ref_energy = np.dot(reference, reference) + eps
    scale = np.dot(reference, estimate) / ref_energy
    target = scale * reference
    noise = estimate - target
    return float(10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def write_report(path: str | Path, report: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

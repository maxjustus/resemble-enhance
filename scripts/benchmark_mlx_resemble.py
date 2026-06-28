from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resemble_enhance.mlx_backend.audio import load_audio
from resemble_enhance.mlx_backend.denoiser import (
    denoise_audio_mlx,
    denoise_audio_with_model,
    load_denoiser,
)
from resemble_enhance.mlx_backend.enhancer import (
    enhance_audio_mlx,
    enhance_audio_with_model,
    load_enhancer,
)
from resemble_enhance.mlx_backend.parity import write_report


def detect_hardware_model() -> str | None:
    for command in (["sysctl", "-n", "hw.model"], ["sysctl", "-n", "machdep.cpu.brand_string"]):
        try:
            return subprocess.check_output(command, text=True).strip()
        except Exception:
            pass
    return None


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(rss)
    return int(rss * 1024)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="enhance", choices=["denoise", "enhance"])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--reload-each-run", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("artifacts/mlx_benchmark.json"))
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--hparams_path", type=Path, default=None)
    parser.add_argument("--solver", type=str, default="midpoint", choices=["midpoint", "rk4", "euler"])
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--lambd", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk_seconds", type=float, default=None)
    parser.add_argument("--overlap_seconds", type=float, default=None)
    parser.add_argument("--ui_responsive", type=str, default="unknown")
    args = parser.parse_args()

    wav, sr = load_audio(args.input)

    timings = []
    warmup_timings = []
    load_time = None
    out_sr = None
    out = None
    if args.reload_each_run:
        for run in range(args.runs):
            start = time.perf_counter()
            if args.mode == "denoise":
                out, out_sr = denoise_audio_mlx(
                    wav,
                    sr,
                    weights_path=args.weights_path,
                    hparams_path=args.hparams_path,
                    chunk_seconds=args.chunk_seconds,
                    overlap_seconds=args.overlap_seconds,
                )
            else:
                out, out_sr = enhance_audio_mlx(
                    wav,
                    sr,
                    weights_path=args.weights_path,
                    hparams_path=args.hparams_path,
                    solver=args.solver,
                    nfe=args.nfe,
                    lambd=args.lambd,
                    tau=args.tau,
                    seed=args.seed + run,
                    chunk_seconds=args.chunk_seconds,
                    overlap_seconds=args.overlap_seconds,
                )
            timings.append(time.perf_counter() - start)
    else:
        load_start = time.perf_counter()
        if args.mode == "denoise":
            model = load_denoiser(args.weights_path, hparams_path=args.hparams_path)
        else:
            model = load_enhancer(args.weights_path, hparams_path=args.hparams_path)
        load_time = time.perf_counter() - load_start

        def run_once(run_seed: int):
            if args.mode == "denoise":
                return denoise_audio_with_model(
                    model,
                    wav,
                    sr,
                    chunk_seconds=args.chunk_seconds,
                    overlap_seconds=args.overlap_seconds,
                )
            return enhance_audio_with_model(
                model,
                wav,
                sr,
                solver=args.solver,
                nfe=args.nfe,
                lambd=args.lambd,
                tau=args.tau,
                seed=run_seed,
                chunk_seconds=args.chunk_seconds,
                overlap_seconds=args.overlap_seconds,
            )

        for run in range(args.warmup_runs):
            start = time.perf_counter()
            out, out_sr = run_once(args.seed + 100_000 + run)
            warmup_timings.append(time.perf_counter() - start)

        for run in range(args.runs):
            start = time.perf_counter()
            out, out_sr = run_once(args.seed + run)
            timings.append(time.perf_counter() - start)

    duration_in = len(wav) / sr
    duration_out = len(out) / out_sr
    mean_wall_time = sum(timings) / len(timings)
    report = {
        "hardware_model": detect_hardware_model(),
        "macos_version": platform.mac_ver()[0],
        "mlx_version": version("mlx"),
        "input_duration_seconds": duration_in,
        "output_duration_seconds": duration_out,
        "timed_scope": "load_plus_inference" if args.reload_each_run else "warm_inference",
        "model_reused": not args.reload_each_run,
        "load_time_seconds": load_time,
        "warmup_runs": 0 if args.reload_each_run else args.warmup_runs,
        "warmup_wall_time_seconds": warmup_timings,
        "wall_time_seconds": timings,
        "mean_wall_time_seconds": mean_wall_time,
        "estimated_cold_wall_time_seconds": load_time + mean_wall_time if load_time is not None else None,
        "real_time_factor": mean_wall_time / duration_in if duration_in else None,
        "peak_memory_bytes": peak_rss_bytes(),
        "ui_responsive": args.ui_responsive,
        "chunk_size_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "mode": args.mode,
    }
    write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

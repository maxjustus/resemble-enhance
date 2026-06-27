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
from resemble_enhance.mlx_backend.denoiser import denoise_audio_mlx
from resemble_enhance.mlx_backend.enhancer import enhance_audio_mlx
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
    out_sr = None
    out = None
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

    duration_in = len(wav) / sr
    duration_out = len(out) / out_sr
    report = {
        "hardware_model": detect_hardware_model(),
        "macos_version": platform.mac_ver()[0],
        "mlx_version": version("mlx"),
        "input_duration_seconds": duration_in,
        "output_duration_seconds": duration_out,
        "wall_time_seconds": timings,
        "mean_wall_time_seconds": sum(timings) / len(timings),
        "real_time_factor": (sum(timings) / len(timings)) / duration_in if duration_in else None,
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

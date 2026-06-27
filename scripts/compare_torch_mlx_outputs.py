from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resemble_enhance.enhancer.inference import denoise as torch_denoise
from resemble_enhance.enhancer.inference import enhance as torch_enhance
from resemble_enhance.mlx_backend.audio import load_audio, save_audio
from resemble_enhance.mlx_backend.denoiser import denoise_audio_mlx
from resemble_enhance.mlx_backend.enhancer import enhance_audio_mlx
from resemble_enhance.mlx_backend.parity import max_abs_diff, mean_abs_diff, si_sdr, write_report


def duration_seconds(wav: np.ndarray, sr: int) -> float:
    return float(len(wav) / sr)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["denoise", "enhance"])
    parser.add_argument("--torch-out", type=Path, required=True)
    parser.add_argument("--mlx-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--run_dir", type=Path, default=None)
    parser.add_argument("--weights_path", type=Path, default=None)
    parser.add_argument("--hparams_path", type=Path, default=None)
    parser.add_argument("--solver", type=str, default="midpoint", choices=["midpoint", "rk4", "euler"])
    parser.add_argument("--nfe", type=int, default=64)
    parser.add_argument("--lambd", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wav, sr = load_audio(args.input)

    if args.mode == "denoise":
        with torch.no_grad():
            torch_out, torch_sr = torch_denoise(torch.from_numpy(wav), sr, device="cpu", run_dir=args.run_dir)
        mlx_out, mlx_sr = denoise_audio_mlx(
            wav,
            sr,
            weights_path=args.weights_path,
            hparams_path=args.hparams_path,
        )
    else:
        torch.manual_seed(args.seed)
        with torch.no_grad():
            torch_out, torch_sr = torch_enhance(
                torch.from_numpy(wav),
                sr,
                device="cpu",
                nfe=args.nfe,
                solver=args.solver,
                lambd=args.lambd,
                tau=args.tau,
                run_dir=args.run_dir,
            )
        mlx_out, mlx_sr = enhance_audio_mlx(
            wav,
            sr,
            weights_path=args.weights_path,
            hparams_path=args.hparams_path,
            nfe=args.nfe,
            solver=args.solver,
            lambd=args.lambd,
            tau=args.tau,
            seed=args.seed,
        )

    torch_np = np.asarray(torch_out, dtype=np.float32)
    mlx_np = np.asarray(mlx_out, dtype=np.float32)

    save_audio(args.torch_out, torch_np, torch_sr)
    save_audio(args.mlx_out, mlx_np, mlx_sr)

    n = min(len(torch_np), len(mlx_np))
    report = {
        "mode": args.mode,
        "input": str(args.input),
        "torch_output": str(args.torch_out),
        "mlx_output": str(args.mlx_out),
        "sample_rate_torch": torch_sr,
        "sample_rate_mlx": mlx_sr,
        "duration_torch": duration_seconds(torch_np, torch_sr),
        "duration_mlx": duration_seconds(mlx_np, mlx_sr),
        "max_abs_diff": max_abs_diff(torch_np[:n], mlx_np[:n]),
        "mean_abs_diff": mean_abs_diff(torch_np[:n], mlx_np[:n]),
        "si_sdr_torch_vs_mlx": si_sdr(torch_np[:n], mlx_np[:n]),
        "notes": [],
    }

    if args.mode == "enhance":
        report["notes"].append(
            "Enhance-mode parity includes RNG-stream differences between torch and MLX unless matched-noise diagnostics are used."
        )

    write_report(args.report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

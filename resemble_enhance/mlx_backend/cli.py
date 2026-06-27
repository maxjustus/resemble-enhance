from __future__ import annotations

import argparse
from pathlib import Path

from .audio import load_audio, save_audio
from .denoiser import denoise_audio_mlx
from .enhancer import enhance_audio_mlx


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared(subparser):
        subparser.add_argument("input", type=Path)
        subparser.add_argument("output", type=Path)
        subparser.add_argument("--weights_path", type=Path, default=None)
        subparser.add_argument("--hparams_path", type=Path, default=None)
        subparser.add_argument("--chunk_seconds", type=float, default=None)
        subparser.add_argument("--overlap_seconds", type=float, default=None)

    denoise_parser = subparsers.add_parser("denoise", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared(denoise_parser)

    enhance_parser = subparsers.add_parser("enhance", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_shared(enhance_parser)
    enhance_parser.add_argument("--solver", type=str, default=None, choices=["midpoint", "rk4", "euler"])
    enhance_parser.add_argument("--nfe", type=int, default=None)
    enhance_parser.add_argument("--lambd", type=float, default=0.5)
    enhance_parser.add_argument("--tau", type=float, default=0.5)
    enhance_parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    wav, sr = load_audio(args.input)

    if args.command == "denoise":
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
            seed=args.seed,
            chunk_seconds=args.chunk_seconds,
            overlap_seconds=args.overlap_seconds,
        )

    save_audio(args.output, out, out_sr)


if __name__ == "__main__":
    main()

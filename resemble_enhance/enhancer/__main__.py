import argparse
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(path, always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav, sr


def save_audio(path: Path, wav: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("in_dir", type=Path, help="Path to input audio folder")
    parser.add_argument("out_dir", type=Path, help="Output folder")
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Path to the enhancer run folder, if None, use the default torch model",
    )
    parser.add_argument("--weights_path", type=Path, default=None, help="Path to converted MLX safetensors")
    parser.add_argument("--hparams_path", type=Path, default=None, help="Optional hparams YAML for the MLX backend")
    parser.add_argument("--suffix", type=str, default=".wav", help="Audio file suffix")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for the torch backend")
    parser.add_argument("--backend", type=str, default="torch", choices=["torch", "mlx"], help="Inference backend")
    parser.add_argument("--denoise_only", action="store_true", help="Only apply denoising without enhancement")
    parser.add_argument("--lambd", type=float, default=1.0, help="Denoise strength for enhancement (0.0 to 1.0)")
    parser.add_argument("--tau", type=float, default=0.5, help="CFM prior temperature (0.0 to 1.0)")
    parser.add_argument("--solver", type=str, default="midpoint", choices=["midpoint", "rk4", "euler"], help="Numerical solver to use")
    parser.add_argument("--nfe", type=int, default=64, help="Number of function evaluations")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the MLX backend")
    parser.add_argument("--parallel_mode", action="store_true", help="Shuffle audio paths and skip existing outputs")
    args = parser.parse_args()

    start_time = time.perf_counter()
    paths = sorted(args.in_dir.glob(f"**/*{args.suffix}"))

    if args.parallel_mode:
        random.shuffle(paths)

    if len(paths) == 0:
        print(f"No {args.suffix} files found in the following path: {args.in_dir}")
        return

    if args.backend == "torch":
        import torch

        from .inference import denoise as torch_denoise
        from .inference import enhance as torch_enhance

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available but --device is set to cuda, using CPU instead")
            device = "cpu"

        def process(wav: np.ndarray, sr: int, index: int):
            dwav = torch.from_numpy(wav)
            if args.denoise_only:
                out, out_sr = torch_denoise(dwav=dwav, sr=sr, device=device, run_dir=args.run_dir)
            else:
                out, out_sr = torch_enhance(
                    dwav=dwav,
                    sr=sr,
                    device=device,
                    nfe=args.nfe,
                    solver=args.solver,
                    lambd=args.lambd,
                    tau=args.tau,
                    run_dir=args.run_dir,
                )
            return out.cpu().numpy(), out_sr

    else:
        from ..mlx_backend.denoiser import denoise_audio_mlx
        from ..mlx_backend.enhancer import enhance_audio_mlx

        def process(wav: np.ndarray, sr: int, index: int):
            if args.denoise_only:
                return denoise_audio_mlx(
                    wav,
                    sr,
                    weights_path=args.weights_path,
                    hparams_path=args.hparams_path,
                )
            return enhance_audio_mlx(
                wav,
                sr,
                weights_path=args.weights_path,
                hparams_path=args.hparams_path,
                nfe=args.nfe,
                solver=args.solver,
                lambd=args.lambd,
                tau=args.tau,
                seed=args.seed + index,
            )

    pbar = tqdm(paths)
    for index, path in enumerate(pbar):
        out_path = args.out_dir / path.relative_to(args.in_dir)
        if args.parallel_mode and out_path.exists():
            continue
        pbar.set_description(f"Processing {out_path}")
        wav, sr = load_audio(path)
        out, out_sr = process(wav, sr, index)
        save_audio(out_path, out, out_sr)

    elapsed_time = time.perf_counter() - start_time
    print(f"Enhancement done! {len(paths)} files processed in {elapsed_time:.2f}s")


if __name__ == "__main__":
    main()

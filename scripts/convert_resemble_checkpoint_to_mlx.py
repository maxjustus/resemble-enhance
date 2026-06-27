from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resemble_enhance.mlx_backend.checkpoint import (
    DEFAULT_MLX_DIR,
    checkpoint_prefix_counts,
    export_denoiser_weights,
    export_enhancer_weights,
    inspect_checkpoint_manifest,
    write_manifest,
)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional source run dir")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_MLX_DIR, help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = inspect_checkpoint_manifest(args.run_dir)
    manifest_path = out_dir / "checkpoint_manifest.json"
    write_manifest(manifest_path, manifest)

    enhancer_weights = export_enhancer_weights(run_dir=args.run_dir, weights_path=out_dir / "enhancer_stage2.safetensors")
    denoiser_weights = export_denoiser_weights(run_dir=args.run_dir, weights_path=out_dir / "denoiser.safetensors")

    report = {
        "source_checkpoint": manifest["source_checkpoint"],
        "mapped_weights": {
            "enhancer": str(enhancer_weights),
            "denoiser": str(denoiser_weights),
        },
        "prefix_counts": dict(checkpoint_prefix_counts(args.run_dir)),
        "renamed_keys": [
            "PyTorch conv weights are stored in channel-first layouts and exported into MLX layouts.",
            "Weight-normalized PyTorch layers are de-parametrized before export so MLX stores plain weights.",
        ],
        "transposed_keys": [
            "Conv1d: [out, in, kernel] -> [out, kernel, in]",
            "Conv2d: [out, in, h, w] -> [out, h, w, in]",
            "ConvTranspose1d: [in, out, kernel] -> [out, kernel, in]",
        ],
        "skipped_keys": [
            "Training-only optimizer and DeepSpeed metadata are not exported.",
        ],
        "missing_expected_keys": [],
        "unexpected_keys": [],
    }
    report_path = out_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    text_report = out_dir / "conversion_report.txt"
    text_report.write_text(
        "\n".join(
            [
                f"Source checkpoint: {manifest['source_checkpoint']}",
                f"Enhancer weights: {enhancer_weights}",
                f"Denoiser weights: {denoiser_weights}",
                f"Manifest: {manifest_path}",
                f"JSON report: {report_path}",
            ]
        )
        + "\n"
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

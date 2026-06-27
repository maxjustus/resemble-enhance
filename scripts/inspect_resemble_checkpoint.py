from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resemble_enhance.mlx_backend.checkpoint import inspect_checkpoint_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run_dir", type=Path, default=None, help="Optional source run dir")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/mlx/checkpoint_manifest.json"),
        help="Where to write the JSON manifest",
    )
    args = parser.parse_args()

    manifest = inspect_checkpoint_manifest(args.run_dir)
    write_manifest(args.manifest, manifest)

    print(json.dumps({"source_checkpoint": manifest["source_checkpoint"], "counts": manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()

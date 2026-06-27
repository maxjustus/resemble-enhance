from pathlib import Path

from resemble_enhance.mlx_backend.checkpoint import (
    export_denoiser_weights,
    export_enhancer_weights,
    inspect_checkpoint_manifest,
)


def test_checkpoint_manifest_has_expected_components():
    manifest = inspect_checkpoint_manifest()
    assert manifest["counts"]["denoiser"] > 0
    assert manifest["counts"]["enhancer"] > 0
    assert manifest["counts"]["vocoder"] > 0


def test_weight_exports_exist():
    denoiser = export_denoiser_weights()
    enhancer = export_enhancer_weights()
    assert Path(denoiser).exists()
    assert Path(enhancer).exists()

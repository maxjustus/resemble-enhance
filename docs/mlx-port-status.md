# MLX Port Status

Date: 2026-06-27

## Current State

- MLX denoiser and full enhancer inference run end-to-end from converted safetensors.
- The MLX backend is separate from the PyTorch implementation and does not silently fall back to torch.
- `python -m resemble_enhance.mlx_backend.cli denoise ...` and `enhance ...` work.
- `resemble-enhance ... --backend mlx` works for directory processing and now preloads one MLX model per invocation.
- Checkpoint conversion, parity comparison, and benchmark scripts are available under `scripts/`.
- MLX model `to_mel` paths now use an MLX FFT implementation with the librosa path retained as a reference helper.

## Milestone Tracking

- Stage 0: architecture and checkpoint reconnaissance
  - Status: done
- Stage 1: checkpoint inspection and conversion script
  - Status: done
- Stage 2: audio preprocessing parity
  - Status: done
- Stage 3: low-level MLX layers
  - Status: done
- Stage 4: denoiser-only MLX inference
  - Status: done
- Stage 5: full enhancer MLX inference
  - Status: done
- Stage 6: benchmarking and compositor-friendly chunking verification
  - Status: done
- Stage 7: optional optimization
  - Status: partially done

## Parity Status

Reports were generated from `artifacts/mlx/test_input.wav` against the PyTorch CPU reference.

| Component | Shape parity | Numeric parity | Notes |
|---|---:|---:|---|
| Denoise | exact | max `4.097819e-06`, mean `9.565443e-07`, SI-SDR `89.35 dB` | Report: `artifacts/optimization/after/compare_denoise_accepted.json` |
| Enhance, `nfe=4` | exact | max `0.0167171`, mean `0.00735228`, SI-SDR `15.04 dB` | Includes torch-vs-MLX RNG-stream differences. Report: `artifacts/optimization/after/compare_enhance_nfe4_accepted.json` |

## Benchmarks

Reports were generated on `Mac16,5`, macOS `26.4.1`, MLX `0.31.2`, using `artifacts/mlx/bench_input_5s.wav`.

| Mode | Scope | Input duration | Mean wall time | RTF | Peak RSS | Notes |
|---|---|---:|---:|---:|---:|---|
| Denoise | reload each run | 5.0s | `0.1888s` | `0.0378` | `364 MB` | Report: `artifacts/optimization/after/benchmark_denoise_reload_accepted.json` |
| Denoise | warm reused model | 5.0s | `0.0836s` | `0.0167` | `381 MB` | Report: `artifacts/optimization/after/benchmark_denoise_warm_accepted.json` |
| Enhance, `nfe=64` | reload each run | 5.0s | `1.3577s` | `0.2715` | `1.81 GB` | Report: `artifacts/optimization/after/benchmark_enhance_reload_nfe64_accepted.json` |
| Enhance, `nfe=64` | warm reused model | 5.0s | `1.1733s` | `0.2347` | `1.82 GB` | Report: `artifacts/optimization/after/benchmark_enhance_warm_nfe64_accepted.json` |

## Optimization Notes

- Accepted: model reuse for MLX CLIs and benchmarks. This removes repeated weight loading from batch workflows and makes warm inference measurable.
- Accepted: cached CFM solver time grids and the exponential mapping root. This is behavior-preserving and removes repeated schedule setup.
- Accepted: MLX FFT mel frontend for MLX model `to_mel`. Prototype parity was max `5.1e-6` vs the reference mel path on the 5s fixture.
- Rejected: default fp16. Denoise showed no speedup, and enhance `nfe=4` SI-SDR dropped to `0.44 dB`.
- Rejected: default `mx.compile` for CFM. It improved warm `nfe=64` enhance from about `1.177s` to `1.128s`, but peak RSS rose to about `19.3 GB`.
- Deferred: STFT/ISTFT denoiser rewrite. The denoiser still relies on the torch-like librosa path for reconstruction parity.
- Deferred: LVC rewrite. No safe source-level change was identified; MLX laziness makes naive Python section timings unreliable.

## Known Risks

- Enhance parity remains dominated by torch-vs-MLX RNG stream differences unless a matched-noise diagnostic is used.
- The full model is large: about `356M` parameters when instantiated.
- Current source has a mismatch between CLI defaults and Python helper defaults for `nfe` and `lambd`.
- Exact resample parity may require keeping the I/O boundary in NumPy/SciPy/librosa instead of pure MLX.

## Next Steps

1. Add matched-noise enhance diagnostics so CFM/vocoder parity can be separated from RNG differences.
2. Evaluate a torch-like MLX STFT/ISTFT pair for the denoiser only if it preserves the current `89 dB` denoise parity.
3. Revisit `mx.compile` only behind an opt-in flag or after confirming the peak-memory behavior is fixed upstream.

# MLX Port Status

Date: 2026-06-27

## Current State

- Upstream `resemble-ai/resemble-enhance` source has been cloned into this workspace.
- A local Python 3.12 environment has been created at `.venv`.
- Inference dependencies for inspection and parity work are installed:
  - `torch`
  - `torchaudio`
  - `mlx`
  - `numpy`
  - `scipy`
  - `soundfile`
  - `librosa`
  - `pytest`
- The default enhancer checkpoint has been downloaded and inspected.
- Import-time coupling that blocked inference-only work has been partially removed:
  - `denoiser/inference.py` no longer imports the training module
  - `enhancer/inference.py` no longer imports the training module
  - `enhancer/download.py` no longer uses `torch.hub` just to fetch files
  - `utils` package imports are now lazy enough for inference-only loading
  - `Enhancer` no longer imports the training loop at module import time

## Milestone Tracking

- Stage 0: architecture and checkpoint reconnaissance
  - Status: done
- Stage 1: checkpoint inspection and conversion script
  - Status: not started
- Stage 2: audio preprocessing parity
  - Status: not started
- Stage 3: low-level MLX layers
  - Status: not started
- Stage 4: denoiser-only MLX inference
  - Status: not started
- Stage 5: full enhancer MLX inference
  - Status: not started
- Stage 6: benchmarking and compositor-friendly chunking verification
  - Status: not started
- Stage 7: optional optimization
  - Status: not started

## Known Risks

- The shipped checkpoint stores major enhancer/vocoder weights in weight-norm form.
- The full model is large: about `356M` parameters when instantiated.
- Current source has an accidental mismatch between CLI defaults and Python helper defaults for `nfe` and `lambd`.
- The vocoder path depends on custom location-variable convolution that is not a direct MLX layer swap.
- Exact torchaudio resample parity may require keeping the I/O boundary in NumPy/SciPy/librosa instead of pure MLX.

## Next Execution Order

1. Add checkpoint inspection and conversion scripts that materialize an inference-only, de-weight-normalized MLX-friendly weight export.
2. Add `resemble_enhance/mlx_backend/` scaffolding and the shared audio helpers.
3. Port and parity-test primitive layers, starting with conv/norm/upsample blocks used by the denoiser.
4. Land denoiser-only MLX inference plus CLI.
5. Port the enhancer stack in this order:
   - `IRMAE`
   - `WN`
   - `CFM` sampler
   - `UnivNet`
   - `LVCBlock`
6. Add torch-vs-MLX comparison and benchmark scripts, then tighten parity or document residual drift.

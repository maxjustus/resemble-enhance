# MLX Port Architecture

Date: 2026-06-27

This document maps the current PyTorch inference architecture in this checkout before any MLX model code is added. It is based on the live source tree plus the default shipped checkpoint downloaded from Hugging Face at `resemble_enhance/model_repo/enhancer_stage2`.

## Repo Map

### PyTorch model components

- Denoiser
  - Source files:
    - `resemble_enhance/denoiser/denoiser.py`
    - `resemble_enhance/denoiser/unet.py`
    - `resemble_enhance/denoiser/inference.py`
    - shared chunking path in `resemble_enhance/inference.py`
  - Class names:
    - `Denoiser`
    - `UNet`
    - `UNetBlock`
    - `PreactResBlock`
  - Input shape:
    - public model input: waveform `(b, t)`
    - internal STFT tensors: `mag/cos/sin` `(b, f, t)`
    - UNet input: stacked `(b, 3, f, t)`
  - Output shape:
    - waveform `(b, t)`
    - UNet output: `(b, 3, f, t)` split into magnitude mask plus phase residual channels
  - Checkpoint keys:
    - standalone denoiser: `net.*`, `mel_fn.*`, `dummy`
    - stage2 enhancer checkpoint: `denoiser.net.*`, `denoiser.mel_fn.*`, `denoiser.dummy`

- Enhancer
  - Source files:
    - `resemble_enhance/enhancer/enhancer.py`
    - `resemble_enhance/enhancer/inference.py`
    - `resemble_enhance/enhancer/lcfm/irmae.py`
    - `resemble_enhance/enhancer/lcfm/cfm.py`
    - `resemble_enhance/enhancer/lcfm/lcfm.py`
    - `resemble_enhance/enhancer/lcfm/wn.py`
    - `resemble_enhance/enhancer/univnet/univnet.py`
    - `resemble_enhance/enhancer/univnet/lvcnet.py`
    - `resemble_enhance/enhancer/univnet/amp.py`
    - `resemble_enhance/enhancer/univnet/alias_free_torch/*.py`
  - Class names:
    - `Enhancer`
    - `LCFM`
    - `IRMAE`
    - `CFM`
    - `Solver`
    - `SinusodialTimeEmbedding`
    - `WN`
    - `WNLayer`
    - `UnivNet`
    - `LVCBlock`
    - `KernelPredictor`
    - `AMPBlock`
    - `SnakeBeta`
    - `UpSample1d`
    - `DownSample1d`
    - `LowPassFilter1d`
  - Input shape:
    - public model input: waveform `(b, t)`
    - mel condition: `(b, 128, frames)`
    - latent flow state: `(b, 64, frames)`
    - vocoder conditioning: `(b, 160, frames)` where `160 = 128 mel + 32 extra`
  - Output shape:
    - waveform `(b, t)`
    - intermediate decoded conditioning from `LCFM`: `(b, 160, frames)`
  - Checkpoint keys:
    - `lcfm.ae.*`
    - `lcfm.cfm.*`
    - `vocoder.*`
    - `denoiser.*`
    - `normalizer.*`
    - `mel_fn.*`
    - `dummy`

### Audio preprocessing

- Resampling:
  - `resemble_enhance/inference.py`
  - `torchaudio.functional.resample`
  - explicit settings:
    - `new_freq=44100`
    - `lowpass_filter_width=64`
    - `rolloff=0.9475937167399596`
    - `resampling_method="sinc_interp_kaiser"`
    - `beta=14.769656459379492`
- Normalization:
  - chunk input is divided by per-chunk `abs().max().clamp(min=1e-7)`
  - denoiser and enhancer also normalize waveform batches internally by per-example absolute max
  - mel normalization uses `Normalizer` running mean/std buffers in `resemble_enhance/common.py`
- Chunking:
  - `resemble_enhance/inference.py`
  - default chunk length `30.0s`
  - default overlap `1.0s`
  - overlap alignment uses mel-correlation-based `compute_offset`
- Padding:
  - inference chunk pads waveform tail by `npad=441`
  - denoiser UNet pads 2D feature maps to multiples of the encoder scale factor
  - STFT drops the last frame on analysis and replicates one frame on ISTFT synthesis
- STFT / mel / spectral transforms:
  - denoiser uses `torch.stft` and `torch.istft` with Hann window, `n_fft=1680`, `win_length=1680`, `hop_length=420`
  - mel frontend uses `torchaudio.transforms.MelSpectrogram` with:
    - `n_fft=2048`
    - `win_length=2048`
    - `hop_length=420`
    - `n_mels=128`
    - `mel_scale="slaney"`
    - `norm="slaney"`
    - `power=1`
    - preemphasis `0.97`

### Audio postprocessing

- Overlap/add:
  - `resemble_enhance/inference.py::merge_chunks`
  - linear fade in/out windows over the overlap region
  - mel-correlation-based chunk offset correction before summation
- Normalization:
  - chunk output is rescaled by the chunk input peak that was stripped before inference
- Output sample rate:
  - always `44100` in current inference code

### CLI entrypoints

- Console script:
  - `setup.py` -> `resemble-enhance=resemble_enhance.enhancer.__main__:main`
- Files:
  - `resemble_enhance/enhancer/__main__.py`
  - `resemble_enhance/denoiser/__main__.py`
  - `app.py`
- Functions:
  - `resemble_enhance.enhancer.inference.denoise`
  - `resemble_enhance.enhancer.inference.enhance`
  - `resemble_enhance.denoiser.inference.denoise`
  - shared chunk runner `resemble_enhance.inference.inference`

## Default Checkpoint Inventory

The default shipped checkpoint is downloaded from:

- `https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/hparams.yaml?download=true`
- `https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/latest?download=true`
- `https://huggingface.co/ResembleAI/resemble-enhance/resolve/main/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt?download=true`

Observed facts from the live checkpoint:

- File: `resemble_enhance/model_repo/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt`
- Format: DeepSpeed checkpoint with top-level `module` state dict
- Total `module` keys: `909`
- Dtype on stored tensors: `float16`
- Component breakdown:
  - `lcfm`: `420`
  - `vocoder`: `294`
  - `denoiser`: `189`
  - `mel_fn`: `3`
  - `normalizer`: `2`
  - `dummy`: `1`

This means the MLX converter must either:

- consume the raw DeepSpeed-style state dict directly, including weight-norm parameter pairs like `weight_g` and `weight_v`, or
- load the PyTorch modules, remove parametrizations, and export a flattened inference-only state dict.

The second path is lower risk.

## Explicit Answers

1. What exact model files define the denoiser?
   - `resemble_enhance/denoiser/denoiser.py`
   - `resemble_enhance/denoiser/unet.py`
   - shared runtime wrapper in `resemble_enhance/inference.py`

2. What exact model files define the enhancer?
   - `resemble_enhance/enhancer/enhancer.py`
   - `resemble_enhance/enhancer/lcfm/irmae.py`
   - `resemble_enhance/enhancer/lcfm/cfm.py`
   - `resemble_enhance/enhancer/lcfm/lcfm.py`
   - `resemble_enhance/enhancer/lcfm/wn.py`
   - `resemble_enhance/enhancer/univnet/univnet.py`
   - `resemble_enhance/enhancer/univnet/lvcnet.py`
   - `resemble_enhance/enhancer/univnet/amp.py`
   - `resemble_enhance/enhancer/univnet/alias_free_torch/*.py`

3. What exact checkpoints are loaded by default?
   - The top-level shipped model is `enhancer_stage2`.
   - Runtime file path: `resemble_enhance/model_repo/enhancer_stage2/ds/G/default/mp_rank_00_model_states.pt`
   - There is no automatic standalone denoiser download path in the current repo; denoiser-only top-level inference uses the denoiser nested inside the enhancer checkpoint.

4. Are checkpoints downloaded from Hugging Face, packaged, or URL-fetched?
   - URL-fetched from Hugging Face.
   - Current code resolves direct HF URLs in `resemble_enhance/enhancer/download.py`.

5. What sample rate does each stage expect?
   - Shared inference wrapper resamples all inputs to `44100`.
   - Denoiser expects `44100`.
   - Enhancer expects `44100`.
   - Mel and STFT settings are all configured around `44100`.

6. Does the denoiser operate in waveform space or spectral space?
   - Mixed.
   - Public input/output are waveform.
   - Core prediction operates on STFT magnitude plus phase channels through a 2D UNet.

7. Does the enhancer operate in waveform space, mel space, latent space, or mixed?
   - Mixed.
   - Input/output are waveform.
   - Conditioning is mel space.
   - The autoencoder and CFM operate in latent and mel-derived feature space.
   - The final decoder is a waveform vocoder.

8. What randomness exists in enhancer inference?
   - `CFM._sample_ψ0` samples Gaussian noise.
   - In eval mode it is seeded deterministically with `manual_seed(0)`.
   - `LCFM.forward` also mixes the encoded prior with Gaussian noise using evaluation `tau`.
   - `UnivNet.forward` samples a Gaussian noise tensor `z` for the vocoder front end.
   - There is no public seed argument in the current PyTorch runtime; deterministic behavior comes from fixed internal seeds and eval-mode paths.

9. What are the default enhance parameters?
   - CLI defaults in `resemble_enhance/enhancer/__main__.py`:
     - `nfe=64`
     - `solver="midpoint"`
     - `lambd=1.0`
     - `tau=0.5`
   - Function defaults in `resemble_enhance/enhancer/inference.py` differ:
     - `nfe=32`
     - `solver="midpoint"`
     - `lambd=0.5`
     - `tau=0.5`
   - This mismatch should be preserved deliberately or fixed explicitly during the MLX integration. It should not stay accidental.

10. Are there custom CUDA-only assumptions?
   - No custom CUDA kernels were found in the live source.
   - There are CUDA-biased runtime defaults:
     - CLI device default is `cuda`
     - `torch.cuda.synchronize()` is used in timing
     - training stack is built around DeepSpeed
   - MPS is explicitly avoided in some spectral code by moving tensors to CPU for STFT and mel extraction.

11. Are there PyTorch operations with no direct MLX equivalent?
   - Yes, or at least no drop-in equivalent that should be trusted without dedicated tests:
     - `torch.stft` / `torch.istft`
     - torchaudio resampling with the exact Kaiser sinc parameters
     - weight-normalized conv layers serialized as `weight_g` / `weight_v`
     - `padding="same"` behavior for 1D convs with dilation
     - `ConvTranspose1d` output shape parity
     - `InstanceNorm1d`
     - alias-free upsample/downsample filters
     - the location-variable convolution implemented with `unfold` + `einsum`
     - random generator semantics inside the iterative sampler

12. Is any part of the pipeline already exportable to ONNX?
   - No ONNX export path exists in the repo.
   - Inference from source only: the denoiser body looks more exportable than the full enhancer, but the spectral ops and the enhancer sampler/vocoder stack make ONNX a secondary path, not the minimal MLX path.

13. What is the minimal path to useful denoiser-only MLX?
   - Shared audio I/O and resample parity.
   - Denoiser waveform normalization and chunk merge parity.
   - STFT / ISTFT parity.
   - UNet port plus weight copy.
   - Magnitude-mask and phase-residual reconstruction.
   - Denoiser-only can ship before mel normalization, CFM, or UnivNet are done.

14. What is the hardest blocker for full enhance MLX?
   - The `UnivNet` vocoder path, specifically `LVCBlock.location_variable_convolution`, is the single heaviest implementation/parity risk.
   - The next hardest blocker is reproducing the iterative `CFM` sampling path with the same seeded noise, time mapping, and accumulation behavior.

## Hard Parts To Isolate Early

- Convolution weight layout conversion:
  - MLX 1D conv weights are not PyTorch-shaped.
  - Dedicated deterministic parity tests are required before loading any real weights.
- Spectral parity:
  - The denoiser depends on exact STFT/ISTFT frame accounting and phase handling.
- Weight norm:
  - The checkpoint stores weight-normalized tensors for major enhancer/vocoder blocks.
- LVCBlock:
  - uses unfold-heavy tensor indexing plus `einsum`
  - exact output-shape and memory-layout parity must be tested in isolation
- Chunk boundary handling:
  - current overlap/add path uses mel correlation to estimate offsets
  - that behavior must be matched or explicitly replaced with measured impact


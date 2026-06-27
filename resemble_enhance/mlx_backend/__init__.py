__all__ = [
    "MLXDenoiser",
    "MLXEnhancer",
    "denoise_audio_mlx",
    "enhance_audio_mlx",
    "load_denoiser",
    "load_enhancer",
]


def __getattr__(name):
    if name in {"MLXDenoiser", "denoise_audio_mlx", "load_denoiser"}:
        from .denoiser import MLXDenoiser, denoise_audio_mlx, load_denoiser

        return {
            "MLXDenoiser": MLXDenoiser,
            "denoise_audio_mlx": denoise_audio_mlx,
            "load_denoiser": load_denoiser,
        }[name]

    if name in {"MLXEnhancer", "enhance_audio_mlx", "load_enhancer"}:
        from .enhancer import MLXEnhancer, enhance_audio_mlx, load_enhancer

        return {
            "MLXEnhancer": MLXEnhancer,
            "enhance_audio_mlx": enhance_audio_mlx,
            "load_enhancer": load_enhancer,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

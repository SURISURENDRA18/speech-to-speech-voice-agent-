"""Accelerator selection shared by the PyTorch adapters."""

import torch

from voice_agent.config import settings


def resolve_torch_device(preference: str = "auto") -> str:
    """Pick cuda > mps > cpu for "auto", otherwise honour the explicit choice."""
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def keep_models_loaded(device: str) -> bool:
    """Whether a PyTorch model should stay resident between turns.

    On a CUDA GPU there is room for every model, and reloading would cost
    seconds per turn. On 8 GB Macs (MPS/CPU) the old swap-per-turn behaviour
    is kept unless KEEP_MODELS_LOADED overrides it.
    """
    if settings.keep_models_loaded is not None:
        return settings.keep_models_loaded
    return device == "cuda"


def empty_device_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()

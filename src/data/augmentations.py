from __future__ import annotations

"""Injectable audio/spectrogram augmentation utilities.

Where you can inject this (examples):
1) In Dataset.__getitem__ (after feature extraction):
   melspec = augmenter(melspec) if train else melspec

2) In the training loop (right before model(images)):
   images = augmenter(images)   # if augmenter supports batched input

This module focuses on spectrogram augmentation (SpecAugment), because your
current pipeline trains on Mel-spectrogram tensors.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol

import torch
import torchaudio


class TensorAugmenter(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        ...


@dataclass(frozen=True)
class SpecAugmentConfig:
    """Configuration for SpecAugment on Mel-spectrograms."""

    time_mask_param: int = 30
    freq_mask_param: int = 15
    p: float = 1.0  # probability of applying each mask


class Compose:
    """Compose multiple augmenters into one."""

    def __init__(self, augmenters: Iterable[TensorAugmenter]):
        self.augmenters = list(augmenters)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for aug in self.augmenters:
            x = aug(x)
        return x


class SpecAugment:
    """Applies frequency and time masking to a (Mel) spectrogram.

    Supports tensors shaped:
    - (C, F, T)  e.g. (1, 64, frames)
    - (B, C, F, T) batched

    Notes:
    - This matches your current dataset output: (1, n_mels, frames)
    - It is intentionally stateless and easy to inject.
    """

    def __init__(self, cfg: SpecAugmentConfig = SpecAugmentConfig()):
        if not (0.0 <= cfg.p <= 1.0):
            raise ValueError("SpecAugmentConfig.p must be in [0, 1]")

        self.cfg = cfg
        self._time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=cfg.time_mask_param)
        self._freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=cfg.freq_mask_param)

    def _maybe(self, transform: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
        if self.cfg.p >= 1.0:
            return transform(x)
        if self.cfg.p <= 0.0:
            return x
        if torch.rand(1).item() < self.cfg.p:
            return transform(x)
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            # (C, F, T)
            x = self._maybe(self._freq_mask, x)
            x = self._maybe(self._time_mask, x)
            return x

        if x.dim() == 4:
            # (B, C, F, T) - apply per-item for more variety
            out = []
            for i in range(x.shape[0]):
                xi = x[i]
                xi = self._maybe(self._freq_mask, xi)
                xi = self._maybe(self._time_mask, xi)
                out.append(xi)
            return torch.stack(out, dim=0)

        raise ValueError(
            "SpecAugment expects a 3D (C,F,T) or 4D (B,C,F,T) tensor")


def build_train_augmenter(
    *,
    time_mask_param: int = 30,
    freq_mask_param: int = 15,
    p: float = 1.0,
) -> TensorAugmenter:
    """Convenience factory for your project defaults."""
    return SpecAugment(SpecAugmentConfig(time_mask_param=time_mask_param, freq_mask_param=freq_mask_param, p=p))


def apply_augmenter(x: torch.Tensor, augmenter: Optional[TensorAugmenter]) -> torch.Tensor:
    """Apply augmenter only if provided (helps keep injection clean)."""
    return x if augmenter is None else augmenter(x)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio


@dataclass(frozen=True)
class AudioFeatureConfig:
    sample_rate: int = 16000
    n_mels: int = 64
    max_duration: float = 1.0
    n_fft: int = 1024
    hop_length: int = 512

    @property
    def max_length(self) -> int:
        return int(self.sample_rate * self.max_duration)


def standardize_waveform(
    waveform: torch.Tensor,
    source_sample_rate: int,
    *,
    target_sample_rate: int,
    max_length: int,
) -> torch.Tensor:
    """Resample, convert to mono, and pad/truncate to fixed length.

    Returns a tensor shaped [1, max_length].
    """
    if source_sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            source_sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # Mono (average channels)
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Pad / truncate
    length_adj = max_length - waveform.shape[1]
    if length_adj > 0:
        waveform = torch.nn.functional.pad(waveform, (0, length_adj))
    else:
        waveform = waveform[:, :max_length]

    return waveform


def compute_melspec_db(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> torch.Tensor:
    """Compute MelSpectrogram then convert to decibel scale.

    Expects waveform shaped [1, time]. Returns [1, n_mels, frames].
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()

    melspec = mel_transform(waveform)
    melspec = db_transform(melspec)
    return melspec


def extract_melspec_db_from_file(
    file_path: str | Path,
    *,
    cfg: AudioFeatureConfig = AudioFeatureConfig(),
) -> torch.Tensor:
    """Load an audio file and return standardized Mel-spectrogram (dB)."""
    file_path = Path(file_path)
    waveform, sr = torchaudio.load(str(file_path))
    waveform = standardize_waveform(
        waveform,
        sr,
        target_sample_rate=cfg.sample_rate,
        max_length=cfg.max_length,
    )
    return compute_melspec_db(
        waveform,
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

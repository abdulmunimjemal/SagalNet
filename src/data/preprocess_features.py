from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from src.data.audio_features import AudioFeatureConfig, extract_melspec_db_from_file


AUDIO_EXTENSIONS = (".wav", ".ogg")


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    cfg: AudioFeatureConfig = AudioFeatureConfig(),
    overwrite: bool = False,
) -> int:
    """Precompute Mel-spectrogram (dB) features for a labeled directory tree.

    Expects input_dir to contain subfolders named 0..9 (labels). Writes
    output_dir/<label>/*.pt tensors (mel-spectrograms in dB).

    Returns number of feature files written.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    written = 0
    for label in range(10):
        label_in = input_dir / str(label)
        if not label_in.is_dir():
            continue

        label_out = output_dir / str(label)
        label_out.mkdir(parents=True, exist_ok=True)

        for audio_path in sorted(label_in.iterdir()):
            if not audio_path.is_file():
                continue
            if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
                continue

            out_path = label_out / (audio_path.stem + ".pt")
            if out_path.exists() and not overwrite:
                continue

            melspec = extract_melspec_db_from_file(audio_path, cfg=cfg)
            torch.save(melspec, out_path)
            written += 1

    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute audio features (Mel-spectrogram in dB)")
    parser.add_argument("--input_dir", type=str,
                        default="data/processed", help="Input labeled audio directory")
    parser.add_argument("--output_dir", type=str, default="data/interim/features",
                        help="Output directory for .pt features")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing feature files")

    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--n_mels", type=int, default=64)
    parser.add_argument("--max_duration", type=float, default=1.0)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)

    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = AudioFeatureConfig(
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        max_duration=args.max_duration,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    count = preprocess_directory(
        args.input_dir, args.output_dir, cfg=cfg, overwrite=args.overwrite)
    print(f"Wrote {count} feature files to {args.output_dir}")


if __name__ == "__main__":
    main()

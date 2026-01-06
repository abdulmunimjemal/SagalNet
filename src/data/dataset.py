import os
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

from src.data.audio_features import standardize_waveform


class SpokenDigitDataset(Dataset):
    def __init__(self, data_path=None, file_list=None, sample_rate=16000, n_mels=64, max_duration=1.0, train=False, time_mask_param=30, freq_mask_param=15, input_format="audio"):
        """
        Args:
            data_path (str, optional): Path to the processed data directory. Used if file_list is None.
            file_list (list, optional): List of (path, label) tuples. Overrides data_path.
            sample_rate (int): Target sample rate.
            n_mels (int): Number of Mel bands.
            max_duration (float): Max duration in seconds.
            train (bool): If True, apply SpecAugment.
            time_mask_param (int): Time masking parameter.
            freq_mask_param (int): Frequency masking parameter.
            input_format (str): "audio" to load raw audio, "features" to load precomputed .pt tensors.
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
        self.train = train
        self.input_format = input_format

        self.file_list = []  # Stores (path, label)

        # Load files
        if file_list is not None:
            self.file_list = file_list
        elif data_path is not None:
            self._load_dataset()
        else:
            # This case should ideally not be reached if data_path is not optional,
            # but keeping for robustness if signature changes again.
            raise ValueError("Either data_path or file_list must be provided")

        # Audio Transforms (only needed for raw audio)
        if self.input_format == "audio":
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=1024,  # Kept original n_fft and hop_length
                hop_length=512
            )
            self.db_transform = torchaudio.transforms.AmplitudeToDB()
        elif self.input_format == "features":
            self.mel_transform = None
            self.db_transform = None
        else:
            raise ValueError(
                "input_format must be either 'audio' or 'features'")

        # Augmentation (SpecAugment) - Configurable
        self.time_mask = torchaudio.transforms.TimeMasking(
            time_mask_param=time_mask_param)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            freq_mask_param=freq_mask_param)

    def _load_dataset(self):
        # Traverse 0-9 folders
        for label in range(10):
            label_dir = os.path.join(self.data_path, str(label))
            if not os.path.isdir(label_dir):
                continue

            files = []
            if self.input_format == "audio":
                exts = ['*.ogg', '*.wav']
            else:
                exts = ['*.pt']

            for ext in exts:
                files.extend(glob.glob(os.path.join(label_dir, ext)))

            for f in files:
                self.file_list.append((f, label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]

        if self.input_format == "audio":
            # Load audio
            waveform, sr = torchaudio.load(file_path)

            waveform = standardize_waveform(
                waveform,
                sr,
                target_sample_rate=self.sample_rate,
                max_length=self.max_length,
            )

            # Features
            melspec = self.mel_transform(waveform)
            melspec = self.db_transform(melspec)
        else:
            loaded = torch.load(file_path)
            # Allow either a raw tensor or a dict that contains the tensor
            melspec = loaded.get("melspec") if isinstance(
                loaded, dict) else loaded
            if melspec.dim() == 2:
                melspec = melspec.unsqueeze(0)

        # Augmentation (only if training)
        if self.train:
            melspec = self.freq_mask(melspec)
            melspec = self.time_mask(melspec)

        return melspec, label


if __name__ == "__main__":
    # Test
    dataset = SpokenDigitDataset(data_path="data/processed", train=True)
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset) > 0:
        spec, label = dataset[0]
        print(f"Sample 0 shape: {spec.shape}, Label: {label}")

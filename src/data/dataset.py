import os
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

class SpokenDigitDataset(Dataset):
    def __init__(self, data_path=None, file_list=None, sample_rate=16000, n_mels=64, max_duration=1.0, train=False):
        """
        Args:
            data_path (str, optional): Path to the processed data directory. Used if file_list is None.
            file_list (list, optional): List of (path, label) tuples. Overrides data_path.
            sample_rate (int): Target sample rate.
            n_mels (int): Number of Mel bands.
            max_duration (float): Max duration in seconds.
            train (bool): If True, apply SpecAugment.
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
        self.train = train
        
        self.file_list = [] # Stores (path, label)
        
        # Load files
        if file_list is not None:
            self.file_list = file_list
        elif data_path is not None:
            self._load_dataset()
        else:
            raise ValueError("Either data_path or file_list must be provided")

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512
        )
        
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        # Augmentations
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=10)
        
    def _load_dataset(self):
        # Traverse 0-9 folders
        for label in range(10):
            label_dir = os.path.join(self.data_path, str(label))
            if not os.path.isdir(label_dir):
                continue
                
            files = []
            for ext in ['*.ogg', '*.wav']:
                files.extend(glob.glob(os.path.join(label_dir, ext)))
                
            for f in files:
                self.file_list.append((f, label))
                
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(file_path)
        
        # Resample
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad/Truncate
        length_adj = self.max_length - waveform.shape[1]
        if length_adj > 0:
            waveform = torch.nn.functional.pad(waveform, (0, length_adj))
        else:
            waveform = waveform[:, :self.max_length]
            
        # Features
        melspec = self.mel_transform(waveform)
        melspec = self.db_transform(melspec)
        
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


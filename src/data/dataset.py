import os
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset

class SpokenDigitDataset(Dataset):
    def __init__(self, data_path, sample_rate=16000, n_mels=64, max_duration=1.0):
        """
        Args:
            data_path (str): Path to the processed data directory (containing 0-9 folders).
            sample_rate (int): Target sample rate.
            n_mels (int): Number of Mel bands.
            max_duration (float): Max duration in seconds (files will be padded/truncated).
        """
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_duration)
        self.file_list = []
        self.labels = []
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512
        )
        
        self.db_transform = torchaudio.transforms.AmplitudeToDB()
        
        self._load_dataset()
        
    def _load_dataset(self):
        # Traverse 0-9 folders
        for label in range(10):
            label_dir = os.path.join(self.data_path, str(label))
            if not os.path.isdir(label_dir):
                continue
                
            # Find all audio files (ogg, wav)
            files = []
            for ext in ['*.ogg', '*.wav']:
                files.extend(glob.glob(os.path.join(label_dir, ext)))
                
            for f in files:
                self.file_list.append(f)
                self.labels.append(label)
                
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or Truncate
        length_adj = self.max_length - waveform.shape[1]
        if length_adj > 0:
            # Pad
            waveform = torch.nn.functional.pad(waveform, (0, length_adj))
        else:
            # Truncate
            waveform = waveform[:, :self.max_length]
            
        # Extract features (MelSpectrogram)
        melspec = self.mel_transform(waveform)
        melspec = self.db_transform(melspec)
        
        # melspec shape: (1, n_mels, time_steps)
        
        return melspec, label

if __name__ == "__main__":
    # Simple test
    dataset = SpokenDigitDataset("data/processed")
    print(f"Loaded {len(dataset)} samples.")
    if len(dataset) > 0:
        spec, label = dataset[0]
        print(f"Sample 0 shape: {spec.shape}, Label: {label}")

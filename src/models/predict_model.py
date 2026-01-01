import os
import torch
import torchaudio
import argparse
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.model import SimpleCNN

def predict(file_path, model_path='models/best_model.pth', device='cpu'):
    # Load Model
    model = SimpleCNN(num_classes=10)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model not found at {model_path}")
        return
        
    model.eval()
    model.to(device)
    
    # Process Audio (Copy logic from dataset)
    sample_rate = 16000
    n_mels = 64
    max_duration = 1.0
    max_length = int(sample_rate * max_duration)
    
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
        
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    length_adj = max_length - waveform.shape[1]
    if length_adj > 0:
        waveform = torch.nn.functional.pad(waveform, (0, length_adj))
    else:
        waveform = waveform[:, :max_length]
        
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mels, n_fft=1024, hop_length=512
    )
    db_transform = torchaudio.transforms.AmplitudeToDB()
    
    melspec = mel_transform(waveform)
    melspec = db_transform(melspec)
    
    # Add batch dimension
    melspec = melspec.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(melspec)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
    print(f"File: {file_path}")
    print(f"Predicted Digit: {predicted.item()}")
    print(f"Confidence: {probabilities[0][predicted.item()]:.4f}")
    
    return predicted.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="Path to audio file")
    parser.add_argument('--model_path', type=str, default='models/best_model.pth')
    args = parser.parse_args()
    
    predict(args.file_path, args.model_path)

import numpy as np
from src.data.audio_features import AudioFeatureConfig, extract_melspec_db_from_file
from src.models.model import SimpleCNN, DeeperCNN
import os
import torch
import torchaudio
import argparse
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


def load_model(model_path, device):
    # Try loading as DeeperCNN first (checking state dict keys or just try/except)
    # A robust way is to instantiate DeeperCNN and try loading.

    checkpoint = torch.load(model_path, map_location=device)

    # Check if 'features.0.weight' exists (indicative of DeeperCNN/Sequential names I used)
    # SimpleCNN used 'conv1.0.weight'
    keys = checkpoint.keys()
    if any('features' in k for k in keys):
        model = DeeperCNN(num_classes=10)
        print("Detected DeeperCNN.")
    else:
        model = SimpleCNN(num_classes=10)
        print("Detected SimpleCNN.")

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def predict(file_path, model_path='models/best_model.pth', device='cpu'):
    # Determine device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else (
            'mps' if torch.backends.mps.is_available() else 'cpu'))

    # Load Model
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

    # Process Audio (shared feature extraction)
    try:
        melspec = extract_melspec_db_from_file(
            file_path, cfg=AudioFeatureConfig())
    except Exception as e:
        print(f"Error loading/processing audio: {e}")
        return None, None

    # Add batch dimension
    melspec = melspec.unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(melspec)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    prob_list = probabilities[0].cpu().numpy()
    pred_idx = predicted.item()

    print(f"File: {file_path}")
    print(f"Predicted Digit: {pred_idx}")
    print(f"Confidence: {prob_list[pred_idx]:.4f}")

    return pred_idx, prob_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="Path to audio file")
    parser.add_argument('--model_path', type=str,
                        default='models/best_model.pth')
    args = parser.parse_args()

    predict(args.file_path, args.model_path)

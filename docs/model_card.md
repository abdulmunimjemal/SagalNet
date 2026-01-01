# Model Card: SimpleCNN for Spoken Digit Recognition

## Model Details
- **Architecture**: Simple Convolutional Neural Network (4 Conv Layers + FC Layers).
- **Input**: MelSpectrogram (1x64xDataset_Time_Steps).
- **Output**: 10 Classes (Digits 0-9).
- **Framework**: PyTorch.
- **Developer**: Antigravity (AI Assistant).

## Intended Use
- **Primary Use Case**: Classification of spoken digits in Afaan Oromoo.
- **Target Audience**: Educational and research purposes.

## Training Data
- **Source**: Custom Afaan Oromoo Spoken Digit Dataset.
- **Size**: ~2,300 Audio Samples.
- **Augmentation**: SpecAugment (TimeMasking, FrequencyMasking) used during training.

## Performance
- **Metric**: Accuracy.
- **Target**: >90%.
- **Validation Results**: Achieved >90% validation accuracy after 50 epochs.

## Limitations
- **Input Duration**: Fixed time duration (padded/truncated), may not handle very long pauses well.
- **Environment**: Performance may degrade in noisy environments as the model was trained on relatively clean data (with some augmentation).

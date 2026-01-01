# 4. Experiments & Results

This section documents the journey from a baseline prototype to a high-accuracy production model.

## 🧪 The "Laboratory" Setup

All experiments were tracked using **MLflow**. We varied:
1.  **Model Architecture**: Depth and Width of the CNN.
2.  **Learning Rate**: 0.001 vs 0.0005.
3.  **Augmentation**: Intensity of Time and Frequency Masking.

## 📈 Phase 1: The Baseline (`SimpleCNN`)
We started with a lightweight 4-layer CNN (`SimpleCNN`) to establish a baseline.
*   **Result**: ~87% Accuracy at 30 Epochs.
*   **Issue**: Underfitting. The model lacked the capacity to distinguish subtly similar phonemes in the relatively small dataset.

## 🚀 Phase 2: Going Deeper (`DeeperCNN`)
We introduced `DeeperCNN`, adding:
*   **More Channels**: 32 -> 64 -> 128 -> 256.
*   **Regularization**: Dropout (0.5) and Batch Normalization.
*   **Result**: 90% Accuracy.

## 🔧 Phase 3: Advanced Optimization
To bridge the gap to 95%, we implemented:
1.  **SpecAugment**: Randomly masking blocks of time and frequency. This prevents the model from relying on any single "cheat" feature.
2.  **LR Scheduling**: `ReduceLROnPlateau` to lower the learning rate when validation loss saturates.
3.  **Extended Training**: Increased epochs to 50.

## 🏆 Final Results

The best performing configuration was:
*   **Model**: `DeeperCNN`
*   **Augmentation**: TimeMask=30, FreqMask=15.
*   **Epochs**: 50.

### Metrics Table

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **91.94%** | 92 out of 100 digits are correctly identified. |
| **F1-Score** | 0.9194 | Balanced performance between Precision and Recall. |
| **Precision** | 0.9217 | When it predicts "5", it is almost always "5". |
| **Recall** | 0.9192 | It rarely misses a true "5". |

## 📉 Limitations & Future Work

While 92% is strong, the remaining 8% error rate is likely due to:
1.  **Data Size**: ~2.3k samples is small for Deep Learning.
2.  **Noise**: The dataset is relatively clean; real-world usage might be noisier.

**Next Steps**:
*   **Data Collection**: Crowdsourcing more samples.
*   **Transfer Learning**: Fine-tuning `Wav2Vec2` (Facebook AI) for potentially near-perfect accuracy.

---
[Back to Home](../README.md)

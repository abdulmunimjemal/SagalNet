# 2. System Architecture

This document details the technical design of the system, from data flow to the deep learning model.

## 🏗️ High-Level Design

The system follows a modular **Model-View-Controller (MVC)** inspired pattern, adapted for ML applications.

```mermaid
graph TD
    User((👤 User))
    UI[🖥️ Streamlit App (View)]
    Predict[⚙️ Inference Eng (Controller)]
    Model[🧠 DeeperCNN (Model)]
    Data[📂 Feature Extractor]

    User -->|Uploads/Records| UI
    UI -->|Raw Audio| Predict
    Predict -->|Waveform| Data
    Data -->|Mel-Spectrogram| Model
    Model -->|Logits| Predict
    Predict -->|Probability| UI
```

1.  **View (`app.py`)**: Handles user interaction (recording/uploading).
2.  **Controller (`predict_model.py`)**: Orchestrates the logic. It loads the model, preprocesses audio, and interprets results.
3.  **Model (`src/models/model.py`)**: The pure PyTorch neural network.
4.  **Data (`src/data`)**: Handles the physics of audio conversion.

---

## 🌊 Data Processing Pipeline

Raw audio is not fed directly into the neural network. It must be converted into a visual representation that captures frequency patterns.

```mermaid
graph LR
    A[🎤 Rough Audio] -->|Resample 16kHz| B[📉 Clean Waveform]
    B -->|STFT + Mel Scale| C[🖼️ Melspectrogram (64xN)]
    C -->|Log Scale| D[Decibel Spectrogram]
    D -->|Normalize| E[Tensor (1, 64, N)]
```

*   **Resampling**: Ensures all inputs match the training frequency (16kHz).
*   **Mel-Spectrogram**: A 2D image where Y-axis is Frequency (Mel scale) and X-axis is Time. Color represents Loudness (dB).

---

## 🧠 Model Architecture: `DeeperCNN`

We developed a custom Convolutional Neural Network (CNN) specifically for this task. It is deeper than standard tutorial models to capture the subtle nuances of Afaan Oromoo pronunciation.

### Layer Diagram

```mermaid
graph TD
    Input[Input Tensor (1, 64, Time)]
    
    subgraph Feature Extraction
    C1[Conv2d (1->32) + BN + ReLU] --> P1[MaxPool (2x2)]
    P1 --> C2[Conv2d (32->64) + BN + ReLU]
    C2 --> P2[MaxPool (2x2)]
    P2 --> C3[Conv2d (64->128) + BN + ReLU]
    C3 --> P3[MaxPool (2x2)]
    P3 --> C4[Conv2d (128->256) + BN + ReLU]
    C4 --> GP[AdaptiveAvgPool (4x4)]
    end
    
    subgraph Classification Head
    GP --> F[Flatten]
    F --> D1[Linear (4096->512) + ReLU]
    D1 --> DO1[Dropout (0.5)]
    DO1 --> D2[Linear (512->128) + ReLU]
    D2 --> DO2[Dropout (0.3)]
    DO2 --> Out[Linear (128->10)]
    end

    Input --> C1
    Out --> SM[Softmax Output]
```

### Design Keypoints
1.  **Batch Normalization (BN)**: Applied after every convolution. This stabilizes training and allows for higher learning rates.
2.  **AdaptiveAvgPool**: We force the output of the feature extractor to be `4x4` regardless of the input audio length. This allows the model to handle audio of varying durations (to an extent).
3.  **Dropout**: We use aggressive dropout (0.5 and 0.3) in the fully connected layers to prevent overfitting, forcing the model to learn redundant features.

---
[Next: Setup & Usage ➡️](03_setup_and_usage.md)

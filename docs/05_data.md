# 5. Data Dictionary

## Overview
This dataset consists of audio recordings of spoken digits in **Afaan Oromoo**, ranging from 0 to 9. It serves as the foundational data for training and evaluating our speech recognition model.

## 📂 Dataset Structure
The dataset follows a standard **ImageNet-style** directory structure, where the folder name indicates the class label.

```
data/processed/
├── 0/
│   ├── recording_001.ogg
│   ├── recording_002.wav
│   └── ...
├── 1/
├── ...
└── 9/
```

## 🔊 Audio Properties
Before being fed into the model, all audio is standardized by the `SpokenDigitDataset` class:

*   **Sample Rate**: `16,000 Hz` (Standard for Speech Recognition).
*   **Channels**: `Mono` (Stereo files are averaged).
*   **Duration**: Fixed to `1.0 Second`.
    *   *Shorter files* are padded with silence.
    *   *Longer files* are truncated.
*   **Formats**: Supports `.wav`, `.ogg`, and `.m4a`.

## 🗣️ Class Labels

| Directory / Label | Digit | Afaan Oromoo Word | Pronunciation Note |
| :--- | :--- | :--- | :--- |
| **0** | 0 | **Duwwaa** | /duːwːɑː/ |
| **1** | 1 | **Tokko** | /tokːo/ |
| **2** | 2 | **Lama** | /lɐmɐ/ |
| **3** | 3 | **Sadii** | /sɐdiː/ |
| **4** | 4 | **Afur** | /ɐfur/ |
| **5** | 5 | **Shan** | /ʃɐn/ |
| **6** | 6 | **Ja'a** | /dʒɐʔɐ/ |
| **7** | 7 | **Torba** | /torbɐ/ |
| **8** | 8 | **Saddeet** | /sɐdːeːt/ |
| **9** | 9 | **Sagal** | /sɐgɐl/ |

---
[Back to Home](../README.md)

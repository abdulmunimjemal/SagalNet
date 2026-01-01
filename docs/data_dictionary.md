# Data Dictionary - Spoken Digit Recognition (Afaan Oromoo)

## Overview
This dataset consists of audio recordings of spoken digits in Afaan Oromoo, ranging from 0 to 9. It is designed for training and evaluating speech recognition models.

## Dataset Structure
The dataset is organized into `data/raw/` (source archives) and `data/processed/` (extracted files).

```
data/processed/
├── 0/
│   ├── source_file_001.ogg
│   ├── source_file_002.m4a
│   └── ...
├── 1/
├── ...
└── 9/
```

## File Formats
- **.ogg**: Vorbis compressed audio (Majority of dataset).
- **.m4a**: MPEG-4 Audio (Added in version 1.1).

## Audio Properties
- **Sample Rate**: Resampled to 16,000 Hz.
- **Channels**: Mono (1 channel).
- **Duration**: Padded/Truncated to 1.0 second (16,000 samples).

## Labels
The directory name corresponds to the spoken digit.

| Label | Digit | Afaan Oromoo Word |
|-------|-------|-------------------|
| 0     | 0     | Duwwaa            |
| 1     | 1     | Tokko             |
| 2     | 2     | Lama              |
| 3     | 3     | Sadii             |
| 4     | 4     | Afur              |
| 5     | 5     | Shan              |
| 6     | 6     | Ja'a              |
| 7     | 7     | Torba             |
| 8     | 8     | Saddeet           |
| 9     | 9     | Sagal             |

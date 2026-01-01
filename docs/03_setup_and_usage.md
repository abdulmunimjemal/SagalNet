# 3. Setup & Usage Guide

This guide covers how to set up the development environment, train the model, and run the interactive application.

## 💻 Prerequisites

*   Python 3.9+
*   `ffmpeg` (Required for handling `.m4a` files)

### Installing FFmpeg
*   **Mac**: `brew install ffmpeg`
*   **Linux**: `sudo apt install ffmpeg`
*   **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/yourusername/spoken-digit-recognition-afaan-oromoo.git
    cd spoken-digit-recognition-afaan-oromoo
    ```

2.  **Create Virtual Environment**
    It's recommended to work in a clean environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🎮 Running the App (Inference)

The easiest way to use the model is via the Streamlit UI.

```bash
streamlit run app.py
```
This will open a browser window where you can:
1.  **Record Time**: Click the microphone icon to record a digit.
2.  **Upload**: Upload an existing file.
3.  **Visualise**: See the confidence bars update in real-time.

## 🏋️‍♀️ Training the Model

If you want to retrain the model on new data or experiment with parameters.

### Basic Training
```bash
python run.py train --epochs 30 --model_type deeper
```

### Advanced Options
You can tune hyperparameters via the CLI:
*   `--lr`: Learning Rate (default: 0.001)
*   `--time_mask`: SpecAugment Time Masking parameter (default: 30)
*   `--freq_mask`: SpecAugment Frequency Masking parameter (default: 15)

Example:
```bash
python run.py train --epochs 50 --lr 0.0005 --time_mask 40
```

## 🧪 Experiment Tracking (MLflow)

We use **MLflow** to track every training run. This allows you to compare different architectures and parameters.

1.  **Start the Server**:
    ```bash
    mlflow ui
    ```
2.  **View Results**:
    Open `http://127.0.0.1:5000` in your browser. You will see a table of all runs, with columns for Accuracy, F1-Score, and Loss.

---
[Next: Experiments & Results ➡️](04_experiments_and_results.md)

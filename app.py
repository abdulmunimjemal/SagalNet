import streamlit as st
import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from src.models.predict_model import predict

# Page Config
st.set_page_config(page_title="Spoken Digit Recognition", page_icon="🎙️")

st.title("🎙️ Afaan Oromoo Spoken Digit Recognizer")
st.write("Upload an audio file or record your voice to predict the Spoken Digit (0-9).")

# Sidebar for Model Selection
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path", "models/best_model.pth")

# Input Method
input_method = st.radio("Select Input Method:", ["Upload File", "Record Audio"])

audio_file = None

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Choose a WAV/OGG/M4A file", type=["wav", "ogg", "m4a"])
    if uploaded_file is not None:
        audio_file = uploaded_file

elif input_method == "Record Audio":
    audio_val = st.audio_input("Record a digit")
    if audio_val is not None:
        audio_file = audio_val

# Prediction Logic
if audio_file is not None:
    st.audio(audio_file, format='audio/wav')
    
    if st.button("Predict"):
        with st.spinner("Analyzing Audio..."):
            # Save temporary file because torchaudio.load expects a path or file-like object
            # Streamlit UploadedFile is file-like, but let's save to be safe/consistent with predict.py
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.getvalue())
            
            # Predict
            digit, probs = predict(temp_path, model_path=model_path, device='auto')
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if digit is not None:
                st.success(f"**Predicted Digit:** {digit}")
                
                # Probability Bar Chart
                fig, ax = plt.subplots()
                digits = list(range(10))
                bars = ax.bar(digits, probs, color='skyblue')
                
                # Highlight prediction
                bars[digit].set_color('green')
                
                ax.set_xlabel('Digit')
                ax.set_ylabel('Probability')
                ax.set_title('Prediction Confidence')
                ax.set_xticks(digits)
                ax.set_ylim(0, 1)
                
                st.pyplot(fig)
            else:
                st.error("Prediction failed. Check the logs.")

st.markdown("---")
st.markdown("Optimization & Training by [Your Name]")

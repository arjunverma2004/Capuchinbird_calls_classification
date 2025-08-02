import streamlit as st
import os
import tempfile
import tensorflow as tf
import numpy as np
import librosa
from itertools import groupby
import soundfile as sf

# --- Page and App Configuration ---
st.set_page_config(page_title="Capuchinbird Call Classifier", layout="centered")
MAX_DURATION_SECONDS = 120 

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the Keras model. Caches for performance."""
    try:
        model_path = 'models/calls_classifier.keras'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at '{model_path}'.")
            st.stop()
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

# --- Preprocessing Functions (Kept for regular use) ---
def create_spectrogram(wav_slice):
    """Converts a single audio slice to a normalized log-spectrogram."""
    wav_slice = tf.squeeze(wav_slice)
    slice_len = tf.shape(wav_slice)[0]
    padding_needed = 48000 - slice_len
    
    if padding_needed > 0:
        zero_padding = tf.zeros([padding_needed], dtype=tf.float32)
        wav = tf.concat([wav_slice, zero_padding], 0)
    else:
        wav = wav_slice
    wav = tf.slice(wav, [0], [48000])

    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-6)
    
    min_val = tf.reduce_min(spectrogram)
    max_val = tf.reduce_max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val + 1e-6)
    
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


# --- Main Application UI ---
st.title("🐦 Capuchinbird Call Classifier")
st.write("This app analyzes audio files to detect Capuchinbird calls.")

st.header("Step 1: Definitive Model Test")
st.warning("Before analyzing new audio, let's test the model with a known-good input from the training notebook.")

if st.button("Run Diagnostic Test on Known Spectrogram"):
    diagnostic_file = 'positive_spectrogram.npy'
    if not os.path.exists(diagnostic_file):
        st.error(f"Diagnostic file '{diagnostic_file}' not found! Please upload it to your repository.")
    else:
        with st.spinner("Running diagnostic..."):
            # Load the known-good spectrogram
            known_spectrogram = np.load(diagnostic_file)
            
            # The model expects a batch. Add a batch dimension.
            input_tensor = np.expand_dims(known_spectrogram, axis=0)
            
            # Predict
            diagnostic_pred = model.predict(input_tensor)
            prob = diagnostic_pred[0][0]

            st.subheader("Diagnostic Result")
            st.write(f"Prediction probability on known-good input: **{prob:.4f}**")
            
            if prob > 0.8:
                st.success("✅ **Test Passed:** The model file is working correctly! The issue must be in the audio preprocessing for uploaded files.")
            else:
                st.error("❌ **Test Failed:** The model produced a low score on a known-good input. This confirms the issue is with the **model file itself** (`.keras`). It may be corrupted or is not the correctly trained version.")

st.header("Step 2: Analyze New Audio")
st.info(f"If the diagnostic test passed, you can proceed to upload an audio file. Audio files are limited to **{MAX_DURATION_SECONDS} seconds**.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file:
    # (The rest of the file processing code remains the same as the previous version)
    # This part will only be relevant if the diagnostic test passes.
    st.write("---")
    # ... (all the file processing logic from the previous answer)

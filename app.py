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

# Set a maximum duration for audio files to prevent memory crashes (in seconds)
# A 2-minute limit is a safe starting point for a 1GB RAM environment.
MAX_DURATION_SECONDS = 120 

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the Keras model from disk. Caches the model for performance."""
    try:
        # Ensure the model path is correct.
        # If the model is in the root directory with app.py:
        model_path = 'models/calls_classifier.keras'
        # If it's in a subfolder like 'models':
        # model_path = 'models/Capuchinbird_calls_classifier.keras'
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at '{model_path}'. Please ensure the model is in the correct directory.")
            st.stop()
            
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

# --- Audio Processing Functions (Aligned with your notebook) ---

def check_audio_duration(filename):
    """Checks the duration of an audio file without loading it all into memory."""
    try:
        with sf.SoundFile(filename) as f:
            duration = len(f) / f.samplerate
        return duration
    except Exception as e:
        st.error(f"Could not read audio file properties: {e}")
        return None

def load_audio_16k_mono(filename):
    """Loads an audio file, resamples to 16kHz mono, and returns a TensorFlow tensor."""
    try:
        # Librosa loads the entire file, which is memory-intensive.
        wav_np, _ = librosa.load(filename, sr=16000, mono=True)
        return tf.convert_to_tensor(wav_np, dtype=tf.float32)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def preprocess_mp3(sample, index):
    """
    Takes a 3-second audio slice and converts it to a spectrogram.
    This function is mapped over the dataset of audio slices.
    """
    sample = sample[0]
    # Pad audio to exactly 3 seconds (48000 samples at 16kHz)
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    
    # Compute the Short-Time Fourier Transform (STFT)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    
    # Add a channel dimension for the CNN
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    
    # Ensure the shape is consistent for the model input
    spectrogram = tf.ensure_shape(spectrogram, (1491, 257, 1))
    return spectrogram


# --- Main Application UI ---

st.title("🐦 Capuchinbird Call Counter")
st.write("Upload a forest audio file (WAV or MP3). The model will analyze it in 3-second segments and count the number of distinct Capuchinbird calls.")
st.info(f"ℹ️ For stability on this free platform, audio files are limited to **{MAX_DURATION_SECONDS} seconds**.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Use a temporary file to save the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filename = tmp_file.name

    # --- Pre-computation Safeguard ---
    duration = check_audio_duration(tmp_filename)
    
    if duration is None:
        st.error("Could not process the uploaded file.")
    elif duration > MAX_DURATION_SECONDS:
        st.error(f"Audio file is too long ({duration:.1f}s). Please upload a file shorter than {MAX_DURATION_SECONDS}s.")
    else:
        st.success(f"Audio file duration: {duration:.1f} seconds. Processing...")
        
        # --- Processing and Prediction ---
        with st.spinner("Loading and preprocessing audio..."):
            wav = load_audio_16k_mono(tmp_filename)
        
        if wav is not None:
            # Create 3-second (48000 samples) windows from the audio
            audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1
            )
            
            # Map the preprocessing function to create spectrograms
            audio_slices = audio_slices.map(preprocess_mp3)
            # Batch the spectrograms for efficient prediction
            audio_slices = audio_slices.batch(64)

            with st.spinner("Running model prediction..."):
                # Get model predictions
                yhat = model.predict(audio_slices)
                
                # Apply the same threshold as in your notebook (0.99)
                predictions = [1 if pred > 0.9 else 0 for pred in yhat]
                
                # Group consecutive detections to count distinct call events
                post_processed = [key for key, group in groupby(predictions)]
                num_capuchin_calls = int(np.sum(post_processed))

            st.success(f"**Analysis Complete: Found {num_capuchin_calls} distinct Capuchinbird call(s).**")
            
            with st.expander("Show Prediction Details"):
                st.write(f"The audio was split into {len(predictions)} 3-second segments for analysis.")
                st.write("Raw model probabilities for each segment:")
                st.bar_chart(yhat.flatten())

    # Clean up the temporary file
    if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
        os.remove(tmp_filename)


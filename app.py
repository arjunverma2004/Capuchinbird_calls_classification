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

# IMPORTANT: These values must be calculated from your training dataset
GLOBAL_MIN = -5.43  # Replace with the real minimum from your notebook
GLOBAL_MAX = 12.87  # Replace with the real maximum from your notebook

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the Keras model. Caches for performance."""
    try:
        model_path = 'models/calls_classifier.keras' # Ensure this path is correct
        if not os.path.exists(model_path):
            st.error(f"Model file not found at '{model_path}'.")
            st.stop()
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

# --- Audio Processing Functions ---
def create_spectrogram(wav_slice):
    """Converts a single audio slice to a globally normalized log-spectrogram."""
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
    
    # Normalize using the global constants
    spectrogram = (spectrogram - GLOBAL_MIN) / (GLOBAL_MAX - GLOBAL_MIN + 1e-6)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


# --- Main Application UI ---
st.title(" Capuchinbird Call Classifier🦜")
st.write("Upload an audio file (WAV or MP3) to count the number of distinct Capuchinbird calls.")
st.info(f"ℹ️ For stability, audio files are limited to **{MAX_DURATION_SECONDS} seconds**.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

# ===================================================================
#  THE FIX: Check if uploaded_file is not None before using it.
# ===================================================================
if uploaded_file is not None:
    # This block of code will only run AFTER a user has uploaded a file.
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filename = tmp_file.name

    try:
        duration = librosa.get_duration(path=tmp_filename)
        
        if duration > MAX_DURATION_SECONDS:
            st.error(f"Audio file is too long ({duration:.1f}s). Please upload a file shorter than {MAX_DURATION_SECONDS}s.")
        else:
            st.success(f"Audio file duration: {duration:.1f} seconds. Processing...")
            wav, _ = librosa.load(tmp_filename, sr=16000, mono=True)
            wav = tf.convert_to_tensor(wav, dtype=tf.float32)
            
            audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1
            )
            
            spectrograms = []
            with st.spinner("Generating spectrograms..."):
                for audio_slice_tuple in audio_slices:
                    spec = create_spectrogram(audio_slice_tuple[0])
                    spectrograms.append(spec)
            
            spectrogram_batch = tf.stack(spectrograms)

            with st.spinner("Running model prediction..."):
                yhat = model.predict(spectrogram_batch)
            
            threshold = 0.8
            predictions = [1 if pred > threshold else 0 for pred in yhat]
            post_processed = [key for key, group in groupby(predictions)]
            num_capuchin_calls = int(np.sum(post_processed))

            st.header("Analysis Complete")
            st.success(f"**Found {num_capuchin_calls} distinct Capuchinbird call(s).**")
            st.write(f"(Using a prediction threshold of {threshold})")

            with st.expander("Show Detailed Prediction Scores"):
                st.bar_chart(yhat.flatten())
                st.write("Raw probabilities:", yhat.flatten())

    finally:
        # Ensure the temporary file is always deleted
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

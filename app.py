import streamlit as st
import os
import tempfile
import tensorflow as tf
import numpy as np
import librosa
from itertools import groupby

st.set_page_config(page_title="Capuchinbird Call Classifier", layout="centered")

# --- Model Loader ---
@st.cache_resource
def load_model():
    try:
        # IMPORTANT: Use correct path and filename as saved in training!
        model = tf.keras.models.load_model('models/calls_classifier.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- Preprocessing (MATCHES NOTEBOOK) ---
def load_audio_16k_mono(filename):
    """Loads audio file as mono, 16kHz float32 tensor."""
    wav_np, _ = librosa.load(filename, sr=16000, mono=True)
    return tf.convert_to_tensor(wav_np, dtype=tf.float32)

def preprocess_mp3(sample, index):
    """Pads/trims to 48000, then computes abs STFT, shape (1491, 257, 1)."""
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.ensure_shape(spectrogram, (1491, 257, 1))
    return spectrogram

# --- Main App Logic ---

st.title("Capuchinbird Call Counter 🐦")
st.write("Upload a forest audio file (MP3 or WAV). The app will detect and count distinct Capuchinbird calls.")

uploaded_file = st.file_uploader("Upload audio file", type=["mp3", "wav"])

if uploaded_file:
    with st.spinner(f"Preprocessing audio..."):
        # Save uploaded file to a temp file (needed for librosa)
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_filename = tmp_file.name

        wav = load_audio_16k_mono(tmp_filename)  # shape: (n_samples,)
        os.unlink(tmp_filename)  # clean up

        # Make 3-second (48000 samples) windows, no overlap for simplicity
        audio_slices = tf.keras.utils.timeseries_dataset_from_array(
            wav,
            wav,
            sequence_length=48000,
            sequence_stride=48000,
            batch_size=1
        )
        # Map to spectrograms
        audio_slices = audio_slices.map(preprocess_mp3)
        # Batch for model speed
        audio_slices = audio_slices.batch(64)

    with st.spinner("Predicting Capuchinbird calls..."):
        yhat = model.predict(audio_slices)
        st.write("Raw prediction outputs (probabilities):", yhat.flatten())

        # Threshold as in training
        yhat_classes = [1 if pred > 0.99 else 0 for pred in yhat.flatten()]

        # Collapse consecutive calls
        collapse = [key for key, group in groupby(yhat_classes)]
        num_calls = int(np.sum(collapse))

    st.success(f"Detected Capuchinbird calls: **{num_calls}**")
    st.write(f"({len(yhat_classes)} windows analyzed, using 3-second slices.)")

    st.bar_chart(yhat.flatten())  # Optional: show prediction strength per window

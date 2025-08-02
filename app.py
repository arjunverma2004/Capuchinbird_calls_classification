import streamlit as st
import os
import tempfile
import tensorflow as tf
import numpy as np
import librosa
from itertools import groupby

# --- IMPORTANT: This must be the very first Streamlit command. ---
st.set_page_config(page_title="Capuchinbird Call Classifier", layout="centered")

# --- App Configuration ---
MAX_DURATION_SECONDS = 15 * 60  # 15 minutes

# --- Model Loading ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model safely."""
    try:
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model('models/calls_classifier.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Fatal: Could not load the model. Please check the model file. Error: {e}")
        st.stop()

model = load_model()

# --- Audio Processing Functions ---
def load_audio_16k_mono(filename):
    """Loads audio, resamples to 16kHz mono, returns a TensorFlow tensor."""
    try:
        wav_np, _ = librosa.load(filename, sr=16000, mono=True)
        return tf.convert_to_tensor(wav_np, dtype=tf.float32)
    except Exception as e:
        st.error(f"Error loading audio file: {e}")
        return None

def preprocess_slice(wav_slice):
    """Converts a 3-second audio slice into a spectrogram for the model."""
    wav_slice = wav_slice[0]
    zero_padding = tf.zeros([48000] - tf.shape(wav_slice), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav_slice], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

# --- UI Styling ---
st.markdown("""
<style>
.main-header { font-size: 2.5em; font-weight: bold; color: #4CAF50; text-align: center; margin-bottom: 0.5em; }
.sub-header { font-size: 1.5em; color: #333; text-align: center; margin-bottom: 1em; }
.success-box { background-color: #e8f5e9; border-left: 5px solid #4CAF50; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
.warning-box { background-color: #fff3e0; border-left: 5px solid #ff9800; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- Main Application ---
st.markdown('<div class="main-header">Capuchinbird Call Classifier 🦜</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload audio (MP3, WAV) up to 15 minutes long.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose an audio file(s)", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Analyzing: {uploaded_file.name}")
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                tmp_path = tmpfile.name

            wav = load_audio_16k_mono(tmp_path)

            if wav is None:
                continue

            # **MEMORY SAFEGUARD**
            duration_seconds = len(wav) / 16000
            if duration_seconds > MAX_DURATION_SECONDS:
                st.error(f"Audio is too long! Please upload a file under {MAX_DURATION_SECONDS // 60} minutes.")
                continue # Skip to the next file

            # Create a dataset of 3-second, non-overlapping audio slices
            audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                wav, wav,
                sequence_length=48000,
                sequence_stride=48000,
                batch_size=1
            )

            # Map the preprocessing function to each slice
            audio_slices = audio_slices.map(lambda data, target: preprocess_slice(data))
            audio_slices = audio_slices.batch(64).prefetch(tf.data.AUTOTUNE)

            # Make predictions
            with st.spinner("Analyzing audio... This may take a moment. 🧐"):
                yhat = model.predict(audio_slices)

            class_preds = [1 if prediction > 0.99 else 0 for prediction in yhat]
            postprocessed = tf.math.reduce_sum([key for key, group in groupby(class_preds)]).numpy()

            # Display results
            with st.container(border=True):
                st.markdown(f"**Filename:** `{uploaded_file.name}`")
                st.audio(uploaded_file)
                if postprocessed > 0:
                    st.markdown(f'<div class="success-box">✅ **Result:** Found {int(postprocessed)} potential Capuchinbird call(s)!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">⚠️ **Result:** No Capuchinbird calls were detected.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An unexpected error occurred while processing `{uploaded_file.name}`: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

st.markdown("---")
st.info("This app analyzes audio in 3-second chunks and uses a >99% confidence threshold.")
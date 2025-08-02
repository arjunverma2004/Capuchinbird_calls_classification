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

# --- Audio Processing Functions ---

def check_audio_duration(filename):
    """Checks audio duration without loading the whole file."""
    try:
        with sf.SoundFile(filename) as f:
            return len(f) / f.samplerate
    except Exception as e:
        st.error(f"Could not read audio file properties: {e}")
        return None

def load_audio_16k_mono(filename):
    """Loads and preprocesses the audio file."""
    try:
        wav_np, _ = librosa.load(filename, sr=16000, mono=True)
        return tf.convert_to_tensor(wav_np, dtype=tf.float32)
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None

def create_spectrogram(wav_slice):
    """Converts a single audio slice to a spectrogram tensor."""
    # The slice must be padded/trimmed to exactly 3 seconds (48000 samples)
    zero_padding = tf.zeros([48000] - tf.shape(wav_slice), dtype=tf.float32)
    wav = tf.concat([wav_slice, zero_padding], 0)
    wav = tf.slice(wav, [0], [48000]) # Ensure it's exactly 48000 samples

    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram


# --- Main Application UI ---
st.title("🐦 Capuchinbird Call Classifier & Debugger")
st.write("Upload an audio file to count Capuchinbird calls. Use the tools below to debug predictions.")
st.info(f"ℹ️ Audio files are limited to **{MAX_DURATION_SECONDS} seconds**.")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filename = tmp_file.name

    duration = check_audio_duration(tmp_filename)
    
    if duration is None:
        st.error("Could not process the uploaded file.")
    elif duration > MAX_DURATION_SECONDS:
        st.error(f"Audio file is too long ({duration:.1f}s). Please upload a file shorter than {MAX_DURATION_SECONDS}s.")
    else:
        st.success(f"Audio file duration: {duration:.1f} seconds. Processing...")
        
        with st.spinner("Loading audio and preparing for analysis..."):
            wav = load_audio_16k_mono(tmp_filename)
            
            if wav is not None:
                # Create 3-second (48000 samples) windows from the audio
                audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                    wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1
                )
                
                # ===================================================================
                #  FIX: Replace .map() with a direct for-loop for reliability
                # ===================================================================
                spectrograms = []
                with st.spinner("Generating spectrograms for each segment..."):
                    for audio_slice in audio_slices:
                        # audio_slice is a tuple (data, label), we just need the data
                        spec = create_spectrogram(audio_slice[0])
                        spectrograms.append(spec)

                if not spectrograms:
                    st.warning("Could not generate any spectrograms from the audio file.")
                    st.stop()
                    
                # Stack all the individual spectrograms into a single batch tensor
                spectrogram_batch = tf.stack(spectrograms)

        with st.spinner("Running model prediction..."):
            yhat = model.predict(spectrogram_batch)
        
        st.header("🔬 Prediction Analysis")
        
        st.subheader("1. Raw Model Probabilities")
        st.write("These are the direct outputs from the model. They should now be different for each segment.")
        st.bar_chart(yhat.flatten())
        with st.expander("Show Raw Values as Text"):
            st.write(yhat.flatten())
            
        st.subheader("2. Adjust Prediction Threshold")
        st.write("Adjust the slider to find the right confidence level for counting calls.")
        threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.95, 0.01)

        predictions = [1 if pred > threshold else 0 for pred in yhat]
        post_processed = [key for key, group in groupby(predictions)]
        num_capuchin_calls = int(np.sum(post_processed))

        st.subheader("3. Final Result")
        st.success(f"**With a threshold of `{threshold}`,"
                   f" found {num_capuchin_calls} distinct Capuchinbird call(s).**")

    if 'tmp_filename' in locals() and os.path.exists(tmp_filename):
        os.remove(tmp_filename)

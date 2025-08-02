import streamlit as st
import os
import tempfile
import tensorflow as tf
import librosa
import numpy as np
from itertools import groupby

# --- IMPORTANT: This must be the very first Streamlit command. ---
st.set_page_config(page_title="Capuchinbird Call Classifier", layout="centered")

# Load the pre-trained model
@st.cache_resource
def load_model():
    """Loads the pre-trained Capuchinbird call classifier model."""
    try:
        # Suppress the optimizer-related warning which is harmless for inference
        with st.spinner("Loading model..."):
            model = tf.keras.models.load_model('models/calls_classifier.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

# Load the model
model = load_model()

# --- Audio Preprocessing Functions ---
def load_mp3_16k_mono(filename):
    """
    Load an audio file, convert it to a float tensor, resample to 16 kHz
    single-channel audio using librosa.
    """
    try:
        # Load audio using librosa, resampling to 16kHz and converting to mono
        wav_np, _ = librosa.load(filename, sr=16000, mono=True, res_type='soxr_hq')
        # Convert the numpy array to a TensorFlow tensor
        wav = tf.convert_to_tensor(wav_np, dtype=tf.float32)
        return wav
    except Exception as e:
        st.error(f"Error loading or processing audio file with librosa: {e}")
        return None

def preprocess_mp3(sample):
    """
    Convert a 3-second audio clip into a spectrogram for prediction.
    """
    # Pad the audio clip to a fixed length of 48000 samples
    # This padding is now redundant as we pad the entire audio file first,
    # but kept for safety.
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample], 0)
    
    # Convert the padded audio to a spectrogram
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2) # Add channel dimension
    
    # Ensure the shape is consistent for the model
    spectrogram = tf.ensure_shape(spectrogram, (1491, 257, 1))

    return spectrogram

# --- Main Streamlit App Logic ---

# --- UI Styling ---
st.markdown("""
<style>
.main-header {
    font-size: 2.5em;
    font-weight: bold;
    color: #4CAF50;
    text-align: center;
    margin-bottom: 0.5em;
}
.sub-header {
    font-size: 1.5em;
    color: #333;
    text-align: center;
    margin-bottom: 1em;
}
.stFileUploader > div > div > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 10px 20px;
    cursor: pointer;
}
.stFileUploader > div > div > button:hover {
    background-color: #45a049;
}
.result-card {
    border: 2px solid #ddd;
    border-radius: 10px;
    padding: 20px;
    margin-top: 20px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.success-box {
    background-color: #e8f5e9;
    border-left: 5px solid #4CAF50;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
}
.warning-box {
    background-color: #fff3e0;
    border-left: 5px solid #ff9800;
    padding: 10px;
    margin-bottom: 10px;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Capuchinbird Call Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload audio files (MP3, WAV) to detect Capuchinbird calls.</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Choose an audio file(s)", type=["wav", "mp3"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"Analyzing: {uploaded_file.name}")

        # Create a temporary file to save the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmp_path = tmpfile.name
        
        # Check if the file is a valid audio file
        try:
            wav = load_mp3_16k_mono(tmp_path)
            
            if wav is not None and len(wav) > 0:
                # --- FIX: Pad the entire audio file to be a multiple of sequence_length ---
                sequence_length = 48000 # 3 seconds
                total_samples = tf.shape(wav)[0]
                
                # Calculate how many zeros to pad to make it a multiple of sequence_length
                padding_needed = (sequence_length - (total_samples % sequence_length)) % sequence_length
                wav_padded = tf.pad(wav, [[0, padding_needed]])
                
                # Reshape into 3-second slices
                num_slices = tf.shape(wav_padded)[0] // sequence_length
                audio_slices = tf.reshape(wav_padded, [num_slices, sequence_length])
                
                # Create a TensorFlow dataset
                audio_dataset = tf.data.Dataset.from_tensor_slices(audio_slices)

                # Preprocess the audio slices to create spectrograms
                audio_dataset = audio_dataset.map(preprocess_mp3)
                
                # Process predictions in batches to avoid memory issues
                all_predictions = []
                batch_size = 64
                with st.spinner("Processing audio... This may take a moment."):
                    for batch in audio_dataset.batch(batch_size):
                        predictions = model.predict(batch, verbose=0)
                        all_predictions.extend(predictions.flatten())
                
                # Convert predictions to binary classes using the specified threshold
                class_preds = [1 if prediction > 0.99 else 0 for prediction in all_predictions]
                
                # Count the total number of detected segments
                detected_segments = sum(class_preds)

                # Group consecutive detections to count unique calls
                calls = tf.math.reduce_sum([key for key, group in groupby(class_preds)]).numpy()
                
                with st.container(border=True):
                    st.markdown(f"**Filename:** `{uploaded_file.name}`")
                    st.audio(uploaded_file, format='audio/wav')
                    
                    if calls > 0:
                        st.markdown(f'<div class="success-box">✅ **Result:** Capuchinbird calls detected! ({int(calls)} calls)</div>', unsafe_allow_html=True)
                        st.markdown(f"*(Note: The model detected {detected_segments} audio segments with a high probability.)*")
                    else:
                        st.markdown('<div class="warning-box">⚠️ **Result:** No Capuchinbird calls detected.</div>', unsafe_allow_html=True)

            else:
                st.warning(f"Could not process the file: `{uploaded_file.name}`. The audio might be empty or corrupt.")

        except Exception as e:
            st.error(f"An error occurred while processing `{uploaded_file.name}`: {e}")
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

st.markdown("---")
st.info("Note: The prediction is based on a `0.99` probability threshold for 3-second audio chunks.")


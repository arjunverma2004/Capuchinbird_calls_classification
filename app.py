import streamlit as st
import os
import tempfile
import tensorflow as tf
import librosa
import numpy as np
from itertools import groupby

# Load the pre-trained model
@st.cache_resource
def load_model():
    """Loads the pre-trained Capuchinbird call classifier model."""
    try:
        model = tf.keras.models.load_model('models/calls_classifier.keras')
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
        wav_np, _ = librosa.load(filename, sr=16000, mono=True)
        # Convert the numpy array to a TensorFlow tensor
        wav = tf.convert_to_tensor(wav_np, dtype=tf.float32)
        return wav
    except Exception as e:
        st.error(f"Error loading or processing audio file with librosa: {e}")
        return None

def preprocess_mp3(sample, index):
    """
    Convert longer audio clips into windowed spectrograms for prediction.
    """
    # Ensure the sample tensor has a batch dimension (e.g., shape [1, 48000])
    # timeseries_dataset_from_array can produce samples with a leading dimension of 1
    if len(tf.shape(sample)) > 1:
        sample = sample[0]
    
    # Pad the audio clip to a fixed length of 48000 samples
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

st.set_page_config(page_title="Capuchinbird Call Classifier", layout="centered")

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
            # The load_mp3_16k_mono function handles both .wav and .mp3
            wav = load_mp3_16k_mono(tmp_path)
            
            if wav is not None:
                # Create a TensorFlow dataset from the audio slices
                # The original code uses a sequence_length of 48000 and stride of 48000,
                # which is for the smaller clips. For longer forest recordings, the stride is often
                # shorter to capture more potential calls. We will use a smaller stride for better
                # detection, as seen in your section 9.3 example (16000). Let's use a conservative stride
                # of 16000 to improve detection of shorter calls.
                sequence_length = 48000 # 3 seconds
                sequence_stride = 16000 # 1 second stride
                
                audio_slices = tf.keras.utils.timeseries_dataset_from_array(
                    wav, 
                    wav, 
                    sequence_length=sequence_length, 
                    sequence_stride=sequence_stride, 
                    batch_size=1
                )
                
                # Preprocess the audio slices to create spectrograms
                audio_slices = audio_slices.map(preprocess_mp3)
                audio_slices = audio_slices.batch(64)
                
                # Make predictions
                yhat = model.predict(audio_slices)
                
                # Convert predictions to binary classes
                # A threshold of 0.99 is used in your code, which is a good starting point.
                class_preds = [1 if prediction > 0.99 else 0 for prediction in yhat]
                
                # Group consecutive detections to count unique calls
                calls = tf.math.reduce_sum([key for key, group in groupby(class_preds)]).numpy()
                
                with st.container(border=True):
                    st.markdown(f"**Filename:** `{uploaded_file.name}`")
                    st.audio(uploaded_file, format='audio/wav')
                    
                    if calls > 0:
                        st.markdown(f'<div class="success-box">✅ **Result:** Capuchinbird calls detected! ({int(calls)} calls)</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">⚠️ **Result:** No Capuchinbird calls detected.</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred while processing `{uploaded_file.name}`: {e}")
            # Ensure the temporary file is cleaned up even on error
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

st.markdown("---")
st.info("Note: This app uses a pre-trained model to classify audio. The model's performance may vary.")


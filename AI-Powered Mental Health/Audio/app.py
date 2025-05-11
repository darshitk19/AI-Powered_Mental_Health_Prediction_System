import streamlit as st
import tensorflow as tf
import librosa
import numpy as np
import soundfile as sf
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Mental Health Audio Analysis",
    page_icon="🧠",
    layout="wide"
)

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('mental_health_cnn_lstm.h5')
    return model

# Function to process audio file
def process_audio(audio_file):
    # Load audio file
    audio, sr = librosa.load(audio_file, sr=None)
    
    # Ensure audio is 3 seconds long (300 frames at 100Hz)
    target_length = 3 * sr
    if len(audio) > target_length:
        audio = audio[:target_length]
    else:
        # Pad with zeros if shorter
        audio = np.pad(audio, (0, target_length - len(audio)))
    
    # Parameters for feature extraction
    n_fft = 2048  # FFT window size
    hop_length = 512  # Hop length for STFT
    n_mels = 13  # Number of mel bands
    
    # Extract features
    # MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mels, n_fft=n_fft, hop_length=hop_length)
    
    # Spectral features
    spectral_center = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # Ensure all features have the same number of frames (300)
    n_frames = 300
    if mfccs.shape[1] > n_frames:
        mfccs = mfccs[:, :n_frames]
        spectral_center = spectral_center[:, :n_frames]
        spectral_rolloff = spectral_rolloff[:, :n_frames]
        spectral_bandwidth = spectral_bandwidth[:, :n_frames]
        spectral_flatness = spectral_flatness[:, :n_frames]
        chroma = chroma[:, :n_frames]
    else:
        # Pad with zeros if shorter
        pad_width = n_frames - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)))
        spectral_center = np.pad(spectral_center, ((0, 0), (0, pad_width)))
        spectral_rolloff = np.pad(spectral_rolloff, ((0, 0), (0, pad_width)))
        spectral_bandwidth = np.pad(spectral_bandwidth, ((0, 0), (0, pad_width)))
        spectral_flatness = np.pad(spectral_flatness, ((0, 0), (0, pad_width)))
        chroma = np.pad(chroma, ((0, 0), (0, pad_width)))
    
    # Combine all features
    features = np.concatenate([
        mfccs,                    # 13 features
        spectral_center,          # 1 feature
        spectral_rolloff,         # 1 feature
        spectral_bandwidth,       # 1 feature
        spectral_flatness,        # 1 feature
        chroma                    # 12 features
    ], axis=0)
    
    # Scale the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features.T).T
    
    # Ensure we have exactly 40 features
    if features_scaled.shape[0] != 40:
        # If we have fewer features, pad with zeros
        if features_scaled.shape[0] < 40:
            padding = np.zeros((40 - features_scaled.shape[0], features_scaled.shape[1]))
            features_scaled = np.concatenate([features_scaled, padding], axis=0)
        # If we have more features, truncate
        else:
            features_scaled = features_scaled[:40, :]
    
    # Ensure we have exactly 300 frames
    if features_scaled.shape[1] != 300:
        # If we have fewer frames, pad with zeros
        if features_scaled.shape[1] < 300:
            padding = np.zeros((features_scaled.shape[0], 300 - features_scaled.shape[1]))
            features_scaled = np.concatenate([features_scaled, padding], axis=1)
        # If we have more frames, truncate
        else:
            features_scaled = features_scaled[:, :300]
    
    # Reshape for CNN-LSTM model input (None, 1, 40, 300, 1)
    features_reshaped = features_scaled.reshape(1, 1, 40, 300, 1)
    
    return features_reshaped

def main():
    st.title("🧠 Mental Health Audio Analysis")
    st.write("Upload an audio file to analyze mental health indicators")
    
    # File uploader with support for both WAV and MP3
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_path = Path("temp_audio.wav")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Add analyze button
        if st.button("Analyze Audio"):
            try:
                with st.spinner("Processing audio and making predictions..."):
                    # Process the audio
                    features = process_audio(temp_path)
                    
                    # Load model and make prediction
                    model = load_model()
                    prediction = model.predict(features)
                    
                    # Apply softmax to get proper probabilities
                    prediction = tf.nn.softmax(prediction).numpy()
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    # Define all possible mental states (14 classes)
                    mental_states = [
                        "Normal", "Anxiety", "Depression", "Stress",
                        "Anger", "Fear", "Sadness", "Joy",
                        "Surprise", "Disgust", "Contempt", "Guilt",
                        "Shame", "Interest"
                    ]
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Display probabilities for all states
                        for state, prob in zip(mental_states, prediction[0]):
                            st.write(f"{state}: {prob:.2%}")
                    
                    with col2:
                        # Display progress bar for the highest probability
                        max_prob = max(prediction[0])
                        st.write("Confidence Level:")
                        st.progress(float(max_prob))
                        
                        # Display the predicted state
                        predicted_state = mental_states[np.argmax(prediction[0])]
                        st.write(f"Predicted State: {predicted_state}")
                        
                        # Display top 3 predictions
                        st.write("Top 3 Predictions:")
                        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
                        for idx in top_3_indices:
                            st.write(f"{mental_states[idx]}: {prediction[0][idx]:.2%}")
                    
                    # Add some styling
                    st.markdown("---")
                    st.info("Note: This analysis is for demonstration purposes only and should not be used as a substitute for professional medical advice.")
                    
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                st.error("Please make sure the audio file is valid and try again.")
                # Add debug information
                st.error(f"Prediction shape: {prediction.shape if 'prediction' in locals() else 'Not available'}")
            
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

if __name__ == "__main__":
    main() 
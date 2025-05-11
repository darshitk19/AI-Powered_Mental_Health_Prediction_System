import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pickle
import cv2

# Suppress TensorFlow warnings and optimize settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimization

# Set page configuration
st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="centered"
)

# Define confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 80.0,    # High confidence threshold
    'MEDIUM': 60.0,  # Medium confidence threshold
    'LOW': 40.0      # Low confidence threshold
}

def get_confidence_level(confidence):
    if confidence >= CONFIDENCE_THRESHOLDS['HIGH']:
        return 'HIGH', '#2ecc71'  # Green
    elif confidence >= CONFIDENCE_THRESHOLDS['MEDIUM']:
        return 'MEDIUM', '#f39c12'  # Orange
    elif confidence >= CONFIDENCE_THRESHOLDS['LOW']:
        return 'LOW', '#e74c3c'  # Red
    else:
        return 'VERY LOW', '#c0392b'  # Dark Red

# Load face detection model
@st.cache_resource
def load_face_detector():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face detector: {str(e)}")
        return None

# Check if image contains a face and assess quality
def analyze_face(image):
    try:
        # Convert PIL Image to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Load face detector
        face_cascade = load_face_detector()
        if face_cascade is None:
            return False, 0, "Face detector not available"
            
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return False, 0, "No face detected"
            
        # Get the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Calculate face quality metrics
        face_quality = 0
        quality_factors = []
        
        # Check face size
        if w > 100 and h > 100:
            face_quality += 20
            quality_factors.append("Good face size")
        else:
            quality_factors.append("Face too small")
            
        # Check image brightness
        brightness = np.mean(gray)
        if 50 <= brightness <= 200:
            face_quality += 20
            quality_factors.append("Good lighting")
        else:
            quality_factors.append("Poor lighting")
            
        # Check face position
        img_height, img_width = gray.shape
        face_center_x = x + w/2
        face_center_y = y + h/2
        
        if (0.3 * img_width <= face_center_x <= 0.7 * img_width and
            0.3 * img_height <= face_center_y <= 0.7 * img_height):
            face_quality += 20
            quality_factors.append("Good face position")
        else:
            quality_factors.append("Face not centered")
            
        # Check face angle
        if abs(face_center_x - img_width/2) < img_width/4:
            face_quality += 20
            quality_factors.append("Good face angle")
        else:
            quality_factors.append("Face at extreme angle")
            
        # Check image blur
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur > 100:
            face_quality += 20
            quality_factors.append("Clear image")
        else:
            quality_factors.append("Blurry image")
            
        return True, face_quality, quality_factors
        
    except Exception as e:
        st.error(f"Error analyzing face: {str(e)}")
        return False, 0, [str(e)]

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image):
    try:
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize to 56x56 pixels
        image = image.resize((56, 56))
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0
        
        # Reshape for model input (1, 56, 56, 1)
        img_array = img_array.reshape(1, 56, 56, 1)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

# Define emotion classes with emojis
MOOD_CLASSES = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']

def main():
    # Add custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            background-color: #f0f2f6;
            margin: 20px 0;
        }
        .stProgress > div > div > div > div {
            background-color: #2ecc71;
        }
        .warning-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #fff3cd;
            border: 1px solid #ffeeba;
            margin: 10px 0;
        }
        .quality-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #e8f5e9;
            border: 1px solid #c8e6c9;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with improved styling
    st.markdown("""
        <h1 style='text-align: center; color: #1f77b4;'>
            Mental Health Analysis from Images 🧠
        </h1>
        <p style='text-align: center; font-size: 1.2em; color: #666;'>
            Upload an image to analyze the emotional state
        </p>
    """, unsafe_allow_html=True)

    # Load model
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please try again later.")
        return

    # File uploader with improved UI
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a face with good lighting"
    )

    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        
        # Analyze face and quality
        has_face, face_quality, quality_factors = analyze_face(image)
        
        if not has_face:
            st.markdown("""
                <div class="warning-box">
                    <h4 style='color: #856404;'>⚠ No Face Detected</h4>
                    <p>Please upload an image containing a human face. The current image does not appear to contain a detectable face.</p>
                </div>
            """, unsafe_allow_html=True)
            return
            
        # Display image quality analysis
        st.markdown("""
            <div class="quality-box">
                <h4 style='color: #2c3e50;'>Image Quality Analysis</h4>
                <p>Quality Score: {face_quality}%</p>
                <ul>
        """.format(face_quality=face_quality), unsafe_allow_html=True)
        
        for factor in quality_factors:
            st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
            
        st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # Adjust confidence threshold based on face quality
        adjusted_confidence = face_quality * 0.8  # Scale quality to confidence
        
        analyze_and_display_results(image, model, adjusted_confidence)

def analyze_and_display_results(image, model, adjusted_confidence):
    try:
        # Display the image
        st.image(
            image, 
            caption='Uploaded Image'
        )
        
        # Add a prediction button with improved styling
        if st.button('Analyze Image', type='primary'):
            with st.spinner('Analyzing image... Please wait.'):
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                if processed_image is not None:
                    # Make prediction
                    try:
                        predictions = model.predict(processed_image)
                        predicted_class = np.argmax(predictions[0])
                        confidence = float(predictions[0][predicted_class] * 100)
                        
                        # Adjust confidence based on image quality
                        final_confidence = min(confidence, adjusted_confidence)
                        confidence_level, confidence_color = get_confidence_level(final_confidence)
                        
                        # Display results with improved styling
                        st.markdown("""
                            <h3 style='text-align: center; color: #2c3e50; margin-top: 2rem;'>
                                Analysis Results
                            </h3>
                        """, unsafe_allow_html=True)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                                <div class="prediction-box" style='text-align: center;'>
                                    <h4 style='color: #666;'>Detected Mood</h4>
                                    <h2 style='color: #1f77b4; margin: 0;'>{MOOD_CLASSES[predicted_class]}</h2>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown(f"""
                                <div class="prediction-box" style='text-align: center;'>
                                    <h4 style='color: #666;'>Confidence</h4>
                                    <h2 style='color: {confidence_color}; margin: 0;'>{final_confidence:.1f}%</h2>
                                </div>
                            """, unsafe_allow_html=True)
                            
                        with col3:
                            st.markdown(f"""
                                <div class="prediction-box" style='text-align: center;'>
                                    <h4 style='color: #666;'>Confidence Level</h4>
                                    <h2 style='color: {confidence_color}; margin: 0;'>{confidence_level}</h2>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Display confidence message
                        if confidence_level == 'HIGH':
                            st.success(f"✅ High confidence in detecting {MOOD_CLASSES[predicted_class].split()[0].lower()} expression. The analysis is reliable.")
                        elif confidence_level == 'MEDIUM':
                            st.warning(f"⚠ Medium confidence in detecting {MOOD_CLASSES[predicted_class].split()[0].lower()} expression. Consider taking another photo with better lighting.")
                        elif confidence_level == 'LOW':
                            st.error(f"⚠ Low confidence in detecting {MOOD_CLASSES[predicted_class].split()[0].lower()} expression. Please try again with a clearer image.")
                        else:
                            st.error("❌ Very low confidence in expression detection. The image may not be suitable for analysis. Please try again with a better quality image.")
                        
                        # Display emotion distribution with improved styling
                        st.markdown("""
                            <h3 style='text-align: center; color: #2c3e50; margin-top: 2rem;'>
                                Emotion Distribution
                            </h3>
                        """, unsafe_allow_html=True)
                        
                        # Create a container for the distribution
                        with st.container():
                            probabilities = predictions[0] * 100
                            for mood, prob in zip(MOOD_CLASSES, probabilities):
                                # Create three columns: label, progress bar, percentage
                                label_col, bar_col, pct_col = st.columns([2, 6, 1])
                                
                                # Display emotion label
                                with label_col:
                                    st.write(f"{mood}")
                                
                                # Display progress bar
                                with bar_col:
                                    st.progress(float(prob/100))
                                
                                # Display percentage
                                with pct_col:
                                    st.write(f"{prob:.1f}%")
                            
                    except Exception as e:
                        st.error("Error making prediction. Please ensure the image contains a clear face.")
                        st.write(f"Technical details: {str(e)}")
                        
    except Exception as e:
        st.error("Error processing image. Please try a different image.")
        st.write(f"Technical details: {str(e)}")

    # Add information about the app with improved styling
    with st.expander("ℹ About this app"):
        st.markdown("""
            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 10px;'>
                <h4 style='color: #2c3e50;'>About the Mental Health Analysis Tool</h4>
                <p>This app uses a deep learning model to analyze emotions from facial expressions in images.</p>
                
                <h5 style='color: #2c3e50; margin-top: 1rem;'>Detectable Emotions:</h5>
                <ul>
                    <li>😠 Anger</li>
                    <li>🤢 Disgust</li>
                    <li>😨 Fear</li>
                    <li>😊 Happiness</li>
                    <li>😐 Neutral</li>
                    <li>😢 Sadness</li>
                    <li>😲 Surprise</li>
                </ul>
                
                <h5 style='color: #2c3e50; margin-top: 1rem;'>Confidence Levels:</h5>
                <ul>
                    <li>🟢 High (≥80%): Reliable expression detection</li>
                    <li>🟠 Medium (≥60%): Consider retaking photo</li>
                    <li>🔴 Low (≥40%): Poor quality detection</li>
                    <li>⚫ Very Low (<40%): Unreliable detection</li>
                </ul>
                
                <h5 style='color: #2c3e50; margin-top: 1rem;'>For Best Results:</h5>
                <ul>
                    <li>Upload clear, well-lit images</li>
                    <li>Ensure the face is clearly visible and centered</li>
                    <li>Avoid multiple faces in one image</li>
                    <li>Use recent photographs for accurate analysis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
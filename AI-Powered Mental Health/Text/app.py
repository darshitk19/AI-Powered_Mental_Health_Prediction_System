import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page config
st.set_page_config(
    page_title="Mental Health Prediction",
    page_icon="🧠",
    layout="wide"
)

# Load the model and preprocessing tools
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('LSTM_mental_health_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def limit_sequence_length(sequences, tokenizer, maxlen=100):
    """Limit sequence length and handle out-of-vocabulary words"""
    # Get the vocabulary size from the model's embedding layer (5000)
    vocab_size = 5000
    
    # Replace out-of-vocabulary words with 0 (padding token)
    limited_sequences = []
    for seq in sequences:
        limited_seq = [min(token, vocab_size-1) for token in seq]
        limited_sequences.append(limited_seq)
    
    return tf.keras.preprocessing.sequence.pad_sequences(
        limited_sequences, maxlen=maxlen, padding='post', truncating='post'
    )

# Main app
def main():
    st.title("🧠 Mental Health Prediction")
    st.write("Enter your text below to predict mental health condition")

    # Load model and preprocessing tools
    try:
        model, tokenizer, label_encoder = load_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Text input
    user_input = st.text_area("Enter your text:", height=150)
    
    if st.button("Predict"):
        if user_input:
            try:
                # Preprocess the input text
                processed_text = preprocess_text(user_input)
                
                # Tokenize and pad the text
                sequences = tokenizer.texts_to_sequences([processed_text])
                padded_sequences = limit_sequence_length(sequences, tokenizer)
                
                # Make prediction
                prediction = model.predict(padded_sequences)
                predicted_label = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                # Get the predicted condition from the label encoder array
                predicted_condition = label_encoder[predicted_label]
                
                # Display results
                st.subheader("Prediction Results")
                st.write(f"**Predicted Condition:** {predicted_condition}")
                st.write(f"**Confidence:** {confidence:.2%}")
                
                # Display confidence bar
                st.progress(float(confidence))
                
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.info("Please try different text or rephrase your input.")
        else:
            st.warning("Please enter some text to predict.")

if __name__ == "__main__":
    main()

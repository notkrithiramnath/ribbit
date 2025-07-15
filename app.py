import streamlit as st
import numpy as np
import librosa
import pickle

# Load the trained model and labels
@st.cache_resource
def load_model_and_labels():
    with open('frog_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('labels.txt', 'r') as f:
        labels = [line.strip() for line in f]
    return model, labels

model, labels = load_model_and_labels()

def extract_features(audio, sr, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1).reshape(1, -1)

st.title("Frog Sound Classifier 🐸")
st.write("Upload a frog sound and we'll (try our best to) tell you which frog made it!")

uploaded_file = st.file_uploader("Upload a frog sound (WAV/MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    try:
        audio, sr = librosa.load(uploaded_file, sr=None)
        features = extract_features(audio, sr)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(features)[0]
        prediction = model.predict(features)
        predicted_idx = prediction[0]
        species = labels[predicted_idx]
        confidence = probabilities[predicted_idx] * 100
        
        # Display results
        st.success(f"Predicted Frog Species: {species}")
        st.info(f"Confidence: {confidence:.1f}%")
        
        # Show top 3 predictions
        st.subheader("Top Predictions:")
        top_indices = np.argsort(probabilities)[::-1][:3]
        for i, idx in enumerate(top_indices):
            species_name = labels[idx]
            prob = probabilities[idx] * 100
            st.write(f"{i+1}. {species_name}: {prob:.1f}%")
                
    except Exception as e:
        st.error(f"Error processing audio: {e}") 
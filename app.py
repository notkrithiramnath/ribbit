import streamlit as st
import numpy as np
import librosa
import pickle
import requests
from urllib.parse import quote

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

def format_species_name_for_wikipedia(species_name):
    """Convert snake_case species name to Wikipedia-friendly format"""
    # Convert snake_case to Title Case with spaces
    formatted = species_name.replace('_', ' ').title()
    # Handle special cases
    name_mapping = {
        'Emerald Tree Frog': 'Emerald tree frog',
        'Gray Tree Frog': 'Gray tree frog',
        'American Bull Frog': 'American bullfrog',
        'Northern Leopard Frog': 'Northern leopard frog',
        'Green Frog': 'Green frog',
        'Spring Peeper Frog': 'Spring peeper'
    }
    return name_mapping.get(formatted, formatted)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_frog_info(species_name):
    """Fetch frog information from Wikipedia API"""
    try:
        formatted_name = format_species_name_for_wikipedia(species_name)
        
        # Wikipedia API endpoint - use underscores as Wikipedia expects
        page_title = formatted_name.replace(' ', '_')
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_title}"
        
        # Wikipedia requires a User-Agent header
        headers = {
            'User-Agent': 'FrogSoundClassifier/1.0 (https://github.com/user/frog-classifier; contact@example.com)'
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'title': data.get('title', formatted_name),
                'extract': data.get('extract', 'No description available.'),
                'thumbnail': data.get('thumbnail', {}).get('source', None),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page', None)
            }
        else:
            return None
    except Exception:
        return None

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
        st.success(f"Predicted Frog Species: {species.replace('_', ' ').title()}")
        st.info(f"Confidence: {confidence:.1f}%")
        
        # Fetch and display frog information
        with st.spinner("Fetching frog information..."):
            frog_info = get_frog_info(species)
            
        if frog_info:
            st.markdown("---")
            st.subheader("📚 About This Frog")
            
            # Display thumbnail if available
            if frog_info.get('thumbnail'):
                st.image(frog_info['thumbnail'], width=300)
            
            # Display description
            st.write(frog_info['extract'])
            
            # Add link to Wikipedia page
            if frog_info.get('url'):
                st.markdown(f"[Learn more on Wikipedia →]({frog_info['url']})")
        else:
            st.info("Information about this frog species is not available at the moment.")
        
        # Show top 3 predictions
        st.markdown("---")
        st.subheader("Top Predictions:")
        top_indices = np.argsort(probabilities)[::-1][:3]
        for i, idx in enumerate(top_indices):
            species_name = labels[idx]
            prob = probabilities[idx] * 100
            st.write(f"{i+1}. {species_name.replace('_', ' ').title()}: {prob:.1f}%")
                
    except Exception as e:
        st.error(f"Error processing audio: {e}") 
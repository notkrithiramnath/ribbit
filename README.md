# Frog Sound Classifier 🐸

A simple web app to classify frog sounds by species.

## Features
- Upload a frog sound (WAV/MP3)
- Get a prediction of which frog made the sound

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Open the provided local URL in your browser.

## Machine Learning Model

I used Scikit Random forest classifier, though in the future I would want to move to TensorFlow.
I also could not find any dataset of frog noises, so I made my own in the program, but I need to upload many more frog noises and train. I also used Cursor AI for help in generating code.

---

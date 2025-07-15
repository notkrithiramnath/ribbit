import os
import librosa
import numpy as np
import glob

DATASET_DIR = 'frog_dataset'  # Change this if your dataset is elsewhere
N_MFCC = 13  # Number of MFCCs to extract

X = []
y = []
labels = []

# Build label list
for species in sorted(os.listdir(DATASET_DIR)):
    species_dir = os.path.join(DATASET_DIR, species)
    if os.path.isdir(species_dir):
        labels.append(species)

label_to_index = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    species_dir = os.path.join(DATASET_DIR, label)
    for file_path in glob.glob(os.path.join(species_dir, '*.wav')) + glob.glob(os.path.join(species_dir, '*.mp3')):
        try:
            audio, sr = librosa.load(file_path, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
            mfccs_mean = np.mean(mfccs, axis=1)  # shape: (N_MFCC,)
            X.append(mfccs_mean)
            y.append(label_to_index[label])
            print(f"Processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = np.array(y)

# Save features and labels
np.save('X_features.npy', X)
np.save('y_labels.npy', y)
with open('labels.txt', 'w') as f:
    for label in labels:
        f.write(f"{label}\n")

print("Feature extraction complete!")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("Labels saved to labels.txt")

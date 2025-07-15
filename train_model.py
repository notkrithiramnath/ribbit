import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load features and labels
X = np.load('X_features.npy')
y = np.load('y_labels.npy')

# Train a classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save the trained model
with open('frog_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved as frog_model.pkl")

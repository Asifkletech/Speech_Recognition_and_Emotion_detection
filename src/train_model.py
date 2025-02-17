import numpy as np
import pandas as pd
import librosa
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import classification_report, accuracy_score
from data_preprocessing import extract_features

# Define dataset paths (modify paths accordingly)
TESS_PATH = 'path_to/TESS'
SAVEE_PATH = 'path_to/SAVEE'
CREMA_PATH = 'path_to/CREMA-D'
RAVDESS_PATH = 'path_to/RAVDESS'

# Function to load datasets (modify this function for dataset loading)
def load_dataset(dataset_path, dataset_name):
    features, labels = [], []
    for file in os.listdir(dataset_path):
        if file.endswith(".wav"):
            file_path = os.path.join(dataset_path, file)
            emotion = "neutral"  # Modify logic for actual emotion extraction
            features.extend(extract_features(file_path))
            labels.extend([emotion] * 4)
    return features, labels

# Load all datasets
tess_features, tess_labels = load_dataset(TESS_PATH, "TESS")
savee_features, savee_labels = load_dataset(SAVEE_PATH, "SAVEE")
crema_features, crema_labels = load_dataset(CREMA_PATH, "CREMA-D")
ravdess_features, ravdess_labels = load_dataset(RAVDESS_PATH, "RAVDESS")

# Combine all datasets
features = np.vstack([tess_features, savee_features, crema_features, ravdess_features])
labels = np.array(tess_labels + savee_labels + crema_labels + ravdess_labels)

# Encode labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels = to_categorical(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Reshape input for LSTM
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Define LSTM model
model = Sequential([
    LSTM(256, return_sequences=False, input_shape=(40, 1)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)

# Save model
model.save('../models/speech_emotion_model.h5')

# Evaluate
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("Accuracy Score:", accuracy_score(y_test_classes, y_pred_classes))
print("Classification Report:\n", classification_report(y_test_classes, y_pred_classes, target_names=label_encoder.classes_))

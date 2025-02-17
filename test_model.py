import os
import numpy as np
import librosa
import tensorflow as tf
from pydub import AudioSegment
import argparse
import warnings

warnings.filterwarnings("ignore")

# Load trained model
MODEL_PATH = "models/speech_emotion_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define emotion classes (Ensure these match your training labels)
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def convert_to_wav(input_path):
    """
    Converts an audio file to WAV format if not already in WAV.
    Saves the converted file in the same directory with .wav extension.
    """
    ext = os.path.splitext(input_path)[-1].lower()

    if ext != ".wav":
        print(f"Converting {input_path} to WAV format...")
        audio = AudioSegment.from_file(input_path, format=ext.replace(".", ""))
        output_path = input_path.replace(ext, ".wav")
        audio.export(output_path, format="wav")
        return output_path
    return input_path  # Return same path if already WAV

def extract_mfcc(file_path):
    """
    Extracts MFCC features from the audio file.
    """
    audio, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, 40, 1)

def predict_emotion(audio_path):
    """
    Predicts the emotion from the given audio file.
    """
    # Convert to WAV if needed
    wav_path = convert_to_wav(audio_path)
    
    # Extract features
    features = extract_mfcc(wav_path)
    
    # Predict
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)

    # Print result
    print(f"Predicted Emotion: {EMOTIONS[predicted_class]}")

    return EMOTIONS[predicted_class]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Speech Emotion Recognition Model")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    args = parser.parse_args()

    # Run prediction
    predict_emotion(args.audio_file)

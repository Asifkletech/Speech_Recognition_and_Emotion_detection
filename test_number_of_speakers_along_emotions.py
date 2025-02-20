import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import torch
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Load Hugging Face Token (Set this in your environment variables)
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("Hugging Face token (HF_TOKEN) is not set. Set it as an environment variable.")

# Load trained emotion recognition model
MODEL_PATH = r"E:\git_speech_emotion\Speech_Recognition_and_Emotion_detection\models\speech_emotion_recognition_with_augmentation.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

# Load Label Encoder
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']
label_encoder = LabelEncoder()
label_encoder.fit(emotions)

# Load Pyannote speaker diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=HF_TOKEN)

# Function to convert audio to WAV format
def convert_to_wav(input_audio_path):
    if input_audio_path.lower().endswith('.wav'):
        return input_audio_path  # Already in WAV format

    output_wav_path = input_audio_path.rsplit('.', 1)[0] + ".wav"
    audio = AudioSegment.from_file(input_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)  # Convert to mono, 16kHz
    audio.export(output_wav_path, format="wav")

    return output_wav_path

# Function to extract MFCC features
def extract_features(audio_segment, sample_rate, max_len=216000):
    if len(audio_segment) < max_len:
        audio_segment = np.pad(audio_segment, (0, max_len - len(audio_segment)), mode='constant')
    else:
        audio_segment = audio_segment[:max_len]

    mfccs = librosa.feature.mfcc(y=audio_segment, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)

    return mfccs

# Function to predict emotion
def predict_emotion(audio_segment, sample_rate):
    features = extract_features(audio_segment, sample_rate)
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    prediction = model.predict(features)
    emotion_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return emotion_label

# Main processing function
def process_audio(input_audio_path):
    # Convert to WAV if needed
    wav_audio_path = convert_to_wav(input_audio_path)

    # Run diarization
    diarization = pipeline(wav_audio_path)

    # Load full audio
    audio, sr = librosa.load(wav_audio_path, sr=16000)

    # Store speaker emotions
    speaker_emotions = {}

    # Identify unique speakers
    speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)

        start, end = turn.start, turn.end
        start_frame = int(start * sr)
        end_frame = int(end * sr)

        # Extract segment
        speaker_audio = audio[start_frame:end_frame]

        # Predict emotion
        emotion = predict_emotion(speaker_audio, sr)

        # Store results
        if speaker not in speaker_emotions:
            speaker_emotions[speaker] = []
        speaker_emotions[speaker].append(emotion)

    # Print number of speakers
    num_speakers = len(speakers)
    print(f"Number of speakers: {num_speakers}")

    # Print speaker emotions
    for speaker, emotions in speaker_emotions.items():
        print(f"Speaker {speaker}: {emotions}")

if __name__ == "__main__":
    AUDIO_FILE = r"E:\git_speech_emotion\1001_DFA_ANG_XX.wav"  # Change to your file path

    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

    print("Processing audio file...")
    process_audio(AUDIO_FILE)
    print("Processing complete.")

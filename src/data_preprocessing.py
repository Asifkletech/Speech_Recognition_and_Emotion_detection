import os
import numpy as np
import librosa

# Data augmentation functions
def add_noise(audio):
    noise = np.random.randn(len(audio)) * 0.005
    return audio + noise

def change_pitch(audio, sr, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

def change_speed(audio, speed_factor=1.2):
    return librosa.effects.time_stretch(audio, rate=speed_factor)

# Function to extract MFCC features with augmentation
def extract_features(file_path, max_len=216000, augment=True):
    audio, sample_rate = librosa.load(file_path)
    if len(audio) < max_len:
        audio = np.pad(audio, (0, max_len - len(audio)), mode='constant')
    else:
        audio = audio[:max_len]

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    features = [mfccs]

    if augment:
        for augmented_audio in [add_noise(audio), change_pitch(audio, sample_rate), change_speed(audio)]:
            features.append(np.mean(librosa.feature.mfcc(y=augmented_audio, sr=sample_rate, n_mfcc=40).T, axis=0))

    return features

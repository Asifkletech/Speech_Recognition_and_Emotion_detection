import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import speech_recognition as sr
from pyannote.audio import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from keybert import KeyBERT
from transformers import pipeline

# Load Pyannote's pre-trained speaker diarization model
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Speaker Diarization Function
def diarize_speakers(audio_file):
    diarization = pipeline(audio_file)
    speaker_segments = {}
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))
    return len(speaker_segments), speaker_segments

# Speech-to-Text function
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Speech not recognized"

# Extract Key Topics
def extract_topics(text, top_n=5):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
    return [kw[0] for kw in keywords]

# TextRank Summarization
def text_rank_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])

def abstractive_summary(text, max_length=150):
    summarizer = pipeline("summarization", model="t5-small")

    # If text is too short, return it as is
    if len(text.split()) < 10:
        return text  # No need to summarize short sentences

    summary = summarizer(text, max_length=max_length, min_length=50, do_sample=False)
    return summary[0]['summary_text']

import nltk
nltk.download('punkt')

nltk.download('all')

test_audio = "/content/1001_DFA_DIS_XX.wav"                         #pass your audio file here
num_speakers, speaker_segments = diarize_speakers(test_audio)
print(f"🗣️ Number of Speakers Detected: {num_speakers}")

transcribed_text = transcribe_audio(test_audio)
print("🎤 Transcribed Speech:", transcribed_text)

topics = extract_topics(transcribed_text)
print("🔹 Key Topics:", topics)

'''extractive_summary = text_rank_summary(transcribed_text)
print("📝 Extractive Summary (TextRank):", extractive_summary)

abstractive_summary_text = abstractive_summary(transcribed_text)
print("📜 Abstractive Summary (T5):", abstractive_summary_text)'''

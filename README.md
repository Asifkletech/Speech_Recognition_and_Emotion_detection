
# 🎤 Speech Emotion Recognition (SER) with LSTM

This project classifies emotions from speech using an **LSTM model** trained on datasets like **RAVDESS, TESS, SAVEE, and CREMA-D**.The model takes **MFCC features** extracted from audio files and predicts emotions.

---

## 📌 Features
✅ Supports multiple **speech emotion datasets**  
✅ **Data augmentation** applied to improve performance  
✅ Uses **MFCC features** for audio processing  
✅ **LSTM-based deep learning model**  
✅ Can test **any audio format (MP3, FLAC, WAV, etc.)**  
✅ Automatically **converts audio** to WAV format before testing  

---

## 🚀 Installation
### 1️⃣ Clone the repository
```bash
git clone https://github.com/Asifkletech/Speech_Recognition_and_Emotion_detection.git
cd speech-emotion-recognition
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Used
- **RAVDESS**  
- **TESS**  
- **SAVEE**  
- **CREMA-D**  

### Total emotions classified:
🎭 **Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise,Calm**

---

## 🏗️ Model Architecture
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients)  
- **Deep Learning Model**: LSTM with Fully Connected Layers  
- **Loss Function**: Categorical Cross-Entropy  
- **Optimizer**: Adam  
- **Augmentation**: Noise addition, pitch shift, speed change  

---

## 🏃‍♂️ Training the Model
### 1️⃣ Ensure datasets are placed in the correct directories:
```bash
/dataset/TESS
/dataset/SAVEE
/dataset/CREMA-D
/dataset/RAVDESS
```

### 2️⃣ Run the training script
```bash
python train_model.py
```

### 3️⃣ Model is saved in the models/ directory
```bash
models/speech_emotion_model.h5
```

---

## 🎧 Testing the Model
You can test the trained model on any audio file using:
```bash
python test_model.py path/to/audio.mp3
```

✔ Supports **MP3, FLAC, WAV, OGG** formats  
✔ Converts audio to **WAV** automatically  
✔ Outputs predicted emotion  

### Example Output:
```bash
Converting path/to/audio.mp3 to WAV format...
Predicted Emotion: happy
```

---

## 📁 Project Structure
```
Speech-Emotion-Recognition/
│── dataset/               # (Optional) Store small samples of datasets if legal
│── src/
│   ├── data_preprocessing.py  # Script for feature extraction and augmentation
│   ├── train_model.py         # Script to train the model
│   ├── test_model.py           # Script for testing/inferencing
│── models/
│   ├── speech_emotion_recognition_with_augmentation.h5  # Saved trained model
│── notebooks/
│── requirements.txt           # Required Python libraries
│── app.py                     # Deployment script (Flask/FastAPI)
│── README.md                  # Project documentation
│── .gitignore                  # Ignore unnecessary files


---


### Dependencies:
- **TensorFlow** (for LSTM model)  
- **Librosa** (for audio feature extraction)  
- **Pandas, NumPy, Scikit-learn** (for data processing)  
- **Pydub** (for audio format conversion)  
- **Matplotlib & Seaborn** (for visualization)  

---

## 🔥 Future Enhancements
🔹 Improve model accuracy with more data  
🔹 Deploy as a web application using Flask or FastAPI  
🔹 Implement real-time speech emotion detection  

---

## 🤝 Contributing
Contributions are welcome!
1. Fork the repo
2. Create a new branch
3. Commit and push changes
4. Submit a pull request

---



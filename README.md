# Emotion Classification App 🎭

A deep learning NLP project that classifies text into 6 emotions — sadness, joy, love, anger, fear, and surprise — using a Bidirectional GRU model, deployed as an interactive Streamlit web app.

## 🚀 Live Demo
https://emotion-classifier-gs25.onrender.com

## 📊 Overview
This project compares plain RNN, LSTM, and GRU architectures before building an advanced **Bidirectional GRU (BiGRU)** model for 6-class emotion classification on the `dair-ai/emotion` dataset from Hugging Face.

## 🧠 Model
- Architecture: Bidirectional GRU with Embedding + Dropout layers
- Framework: TensorFlow / Keras
- Dataset: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- Classes: sadness, joy, love, anger, fear, surprise

## ⚙️ Tech Stack
- Python, TensorFlow/Keras
- Streamlit (web app)
- Pandas, NumPy
- Deployed on Render

## 🏃 Running Locally
```bash
pip install -r requirements.txt
streamlit run app.py

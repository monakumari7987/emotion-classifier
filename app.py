import re
import pickle
import numpy as np
import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 1. Constants
# -----------------------------
MODEL_PATH = "Artifacts/BiGRU_Model.keras"
TOKENIZER_PATH = "Artifacts/tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 50

EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

EMOTION_EMOJIS = {
    'sadness': '😢',
    'joy': '😄',
    'love': '❤️',
    'anger': '😠',
    'fear': '😨',
    'surprise': '😲'
}

# -----------------------------
# 2. Load model + tokenizer (cached so it loads only once)
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

model, tokenizer = load_artifacts()

# -----------------------------
# 3. Preprocessing (same as training)
# -----------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_emotion(text: str):
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    probabilities = model.predict(padded)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = EMOTION_LABELS[predicted_index]
    confidence = float(probabilities[predicted_index])
    all_probs = {EMOTION_LABELS[i]: float(probabilities[i]) for i in range(len(EMOTION_LABELS))}
    return predicted_label, confidence, all_probs

# -----------------------------
# 4. Streamlit UI
# -----------------------------
st.set_page_config(page_title="Emotion Classifier", page_icon="🎭")
st.title("🎭 Emotion Classification")
st.write("Enter a sentence and the model will predict its emotion.")

user_text = st.text_area("Your text:", placeholder="I feel so happy today!")

if st.button("Predict Emotion"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        label, confidence, all_probs = predict_emotion(user_text)
        emoji = EMOTION_EMOJIS[label]
        st.success(f"Predicted Emotion: **{label.upper()}** {emoji}  (confidence: {confidence:.2%})")

        st.subheader("All probabilities")
        st.bar_chart(all_probs)

        model_path = "BiGRU_Model.keras"
        tokenizer_path = "tokenizer.pkl"
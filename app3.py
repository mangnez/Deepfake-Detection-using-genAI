import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import librosa
import tempfile
import os
import requests
import json
import plotly.graph_objects as go

# Set Gemini API Key
GEMINI_API_KEY = "AIzaSyANFNZIyIzc_GXHjx6uF_ssS0fNHVb7ObE"
GEMINI_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPT = """
You are an AI chatbot  for media and news organizations. 
Your role is to provide insights on advancements, risks, countermeasures, legal aspects, 
and AI-based detection tools related to deepfake detection in media.

- If a user greets you, respond warmly and suggest relevant deepfake-related questions.
- If a user asks about deepfake detection, provide precise, well-informed answers.
- If a user asks something unrelated, politely redirect them to deepfake-related topics and suggest them fewer topic.
- You may discuss ethical concerns, AI detection solutions, and legal regulations.
"""

@st.cache_resource
def load_video_model():
    return tf.keras.models.load_model('C:/Users/sruti/Downloads/c708_deepfake_detection/deepfake_detection/deepfake_detection_model.keras')

@st.cache_resource
def load_audio_model():
    return tf.keras.models.load_model('C:/Users/sruti/Downloads/c708_deepfake_detection/deepfake_detection/audio_classifier.h5', compile=False)

def extract_frames(video_path, num_frames=10, target_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    frames = np.array(frames) / 255.0
    frames = np.expand_dims(frames, axis=0)
    return frames

def predict_video(video_file, model):
    frames = extract_frames(video_file, num_frames=10)
    if frames is None:
        return "Error: Could not extract frames. Please upload a valid video.", 0.0

    prediction = model.predict(frames)
    confidence = float(prediction[0][0])
    label = "Deepfake Detected" if confidence > 0.5 else "Real Video"
    return label, confidence

def predict_audio(audio_file, model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_file.read())
        temp_audio_path = temp_file.name

    y, sr = librosa.load(temp_audio_path, sr=16000)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_spec.shape[1] < 109:
        mel_spec = np.pad(mel_spec, ((0, 0), (0, 109 - mel_spec.shape[1])), mode='constant')
    else:
        mel_spec = mel_spec[:, :109]

    mel_spec = np.expand_dims(mel_spec, axis=[0, -1])
    prediction = model.predict(mel_spec)
    os.remove(temp_audio_path)

    confidence = float(prediction[0][0])
    label = "Deepfake Audio Detected" if confidence > 0.5 else "Real Audio"
    return label, confidence

def chat_with_gemini(user_input, chat_history):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    messages = [{
        "role": "user",
        "parts": [{"text": SYSTEM_PROMPT}]
    }]

    for msg in chat_history:
        messages.append({"role": "user", "parts": [{"text": msg["user"]}]})
        messages.append({"role": "model", "parts": [{"text": msg["bot"]}]})

    # Add latest input from user
    messages.append({"role": "user", "parts": [{"text": user_input}]})

    data = {"contents": messages}

    try:
        response = requests.post(url, headers=headers, json=data)
        response_data = response.json()

        if "candidates" in response_data and response_data["candidates"]:
            return response_data["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Sorry, I couldn't get a valid response from Gemini."
    except Exception as e:
        return f"Error communicating with Gemini API: {e}"


def show_confidence_gauge(confidence, label):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{label} Confidence", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred" if label.startswith("Deepfake") else "green"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 75], 'color': 'orange'},
                {'range': [75, 100], 'color': 'red'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

st.set_page_config(page_title="Deepfake Detection", layout="wide")
st.markdown("""
<style>
body {
    background-color: #f4f4f4;
    color: #333;
}
.stButton>button {
    background-color: #0047AB;
    color: white;
    font-size: 18px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("Deepfake Detection System")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Deepfake Detection", "Chatbot (Deepfake in Media)"])

if page == "Deepfake Detection":
    st.sidebar.title("Select Detection Type")
    detection_type = st.sidebar.radio("Choose:", ("Video Deepfake Detection", "Audio Deepfake Detection"))

    if detection_type == "Video Deepfake Detection":
        deepfake_model = load_video_model()
        st.subheader("Upload a Video File")
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_video_path = temp_file.name

            st.video(temp_video_path)

            if st.button("Detect Deepfake"):
                label, confidence = predict_video(temp_video_path, deepfake_model)
                st.subheader("Prediction:")
                st.success(label)
                show_confidence_gauge(confidence, label)
                os.remove(temp_video_path)

    elif detection_type == "Audio Deepfake Detection":
        deepfake_model = load_audio_model()
        st.subheader("Upload an Audio File")
        uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3"])

        if uploaded_file is not None:
            st.audio(uploaded_file)

            if st.button("Detect Deepfake"):
                label, confidence = predict_audio(uploaded_file, deepfake_model)
                st.subheader("Prediction:")
                st.success(label)
                show_confidence_gauge(confidence, label)

elif page == "Chatbot (Deepfake in Media)":
    st.subheader("Ask About Deepfake Detection in Media & News")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["bot"])

    user_input = st.text_input("Type your query...")

    if st.button("Ask"):
        if user_input:
            response = chat_with_gemini(user_input, st.session_state.chat_history)
            st.session_state.chat_history.append({"user": user_input, "bot": response})
            st.rerun()
        else:
            st.warning("Please enter a message.")

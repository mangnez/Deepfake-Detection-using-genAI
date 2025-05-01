# Deepfake Detection System

This project is a **Streamlit-based web application** that allows you to:

✅ Detect deepfake videos using a pre-trained TensorFlow video model  
✅ Detect deepfake audios using a pre-trained audio classifier  
✅ Chat with an AI-powered assistant (via Gemini API) to explore topics related to deepfake detection, risks, countermeasures, legal aspects, and ethical concerns  

---

## 🔧 Features

- **Video Deepfake Detection**  
  Upload a video file (`.mp4`, `.avi`, `.mov`, `.mkv`) and the system will:
  - Extract key frames
  - Run a TensorFlow model to predict whether the video is real or a deepfake
  - Display prediction confidence as an interactive gauge

- **Audio Deepfake Detection**  
  Upload an audio file (`.wav`, `.mp3`) and the system will:
  - Extract the Mel spectrogram
  - Run a TensorFlow model to classify if the audio is real or deepfake
  - Show a confidence gauge

- **Deepfake Chatbot**  
  Use the embedded chatbot (powered by Google Gemini API) to:
  - Ask about deepfake detection technologies, legal regulations, ethical concerns, and AI tools
  - Get warm, topic-specific responses tailored for media and news contexts

---

## 🗂 Project Structure

```
/deepfake_detection/
├── deepfake_detection_model.keras  # Pre-trained video model
├── audio_classifier.h5            # Pre-trained audio model
├── main_app.py                    # Main Streamlit app (provided above)
```

---

## 💻 Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/deepfake-detection-system.git
cd deepfake-detection-system
```

2️⃣ **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Configure Gemini API key**  
Replace the `GEMINI_API_KEY` in the script (`main_app.py`) with your own Gemini API key.

---

## 🚀 Run the App

```bash
streamlit run main_app.py
```

The app will open in your default browser at `http://localhost:8501/`.

---

## 📦 Requirements

- Python 3.8+
- TensorFlow
- OpenCV (`cv2`)
- NumPy
- Librosa
- Plotly
- Requests
- Streamlit

You can install all using:
```bash
pip install tensorflow opencv-python numpy librosa plotly requests streamlit
```

---

## 🛡 Disclaimer

This tool is for **research and educational purposes only**.  
The accuracy of deepfake detection models may vary depending on the dataset, and results should not be used for legal or forensic decisions.

---

## 📄 License

[MIT License](LICENSE)

---

## ✨ Acknowledgments

- TensorFlow team for model development tools  
- Google Gemini API for conversational capabilities  
- Open-source communities for providing foundational libraries

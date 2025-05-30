# Minimal Accent Classifier with SpeechBrain

This is a simple Streamlit web app that classifies English accents from YouTube video audio using the pretrained `Jzuluaga/accent-id-commonaccent_ecapa` model from SpeechBrain.

---

## Features

- Download audio from any public YouTube(or another) video URL
- Preprocess audio
- Predict English accent with confidence score
- Supports multiple accents including American, British, Indian


---

## Installation

1-) Clone the repository:

bash
git clone https://github.com/yourusername/accent-classifier.git
cd accent-classifier



2-) Create and activate a Python virtual environment:

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

3-) Install required dependencies:

pip install -r requirements.txt

Usage
Run the Streamlit app:

streamlit run app.py



.
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── pretrained_models/      # (Ignored by Git) Downloaded SpeechBrain models
└── README.md



Requirements
Python 3.8+

PyTorch

SpeechBrain

Torchaudio

yt-dlp

Streamlit





Acknowledgments
SpeechBrain for the pretrained accent classification model

yt-dlp for YouTube audio extraction

Streamlit for the web app framework






License
MIT License

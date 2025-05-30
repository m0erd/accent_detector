import streamlit as st
import torchaudio
from speechbrain.inference import EncoderClassifier
import yt_dlp
import os
import tempfile
import torch
import datetime

LABEL_MAP = {
    "england": "British English",
    "us": "American English",
    "australia": "Australian English",
    "indian": "Indian English",
    "scotland": "Scottish English",
    "canada": "Canadian English",
    "ireland": "Irish English",
}

if "usage" not in st.session_state:
    st.session_state.usage = {"date": None, "count": 0}

DAILY_LIMIT = 25


def check_limit():
    today = datetime.date.today()
    if st.session_state.usage["date"] != today:
        # Reset count for new day
        st.session_state.usage["date"] = today
        st.session_state.usage["count"] = 0

    if st.session_state.usage["count"] >= DAILY_LIMIT:
        return False
    else:
        st.session_state.usage["count"] += 1
        return True


@st.cache_resource
def load_model():
    model = EncoderClassifier.from_hparams(
        source="Jzuluaga/accent-id-commonaccent_ecapa",
        savedir="pretrained_models/accent-id-commonaccent_ecapa"
    )
    return model


def download_audio_from_youtube(url: str) -> str:
    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, "audio.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for f in os.listdir(temp_dir):
        if f.endswith(".wav"):
            return os.path.join(temp_dir, f), temp_dir
    raise Exception("Failed to extract audio")


def preprocess_audio(audio_path):
    signal, sr = torchaudio.load(audio_path, backend="soundfile")
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        signal = resampler(signal)
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def predict_accent(model, signal):
    try:
        if signal.ndim == 2:
            signal = signal.squeeze(0)
        signal = signal.unsqueeze(0)
        signal = signal.to(torch.float32).cpu()
        scores_all, max_score, class_idx, labels_list = model.classify_batch(signal)

        label = labels_list[0].lower()
        confidence = max_score.item()
        return LABEL_MAP.get(label, f"Unknown English Accent ({label})"), confidence

    except Exception as e:
        return f"Prediction error: {str(e)}", 0.0


def cleanup_temp_dir(temp_dir):
    try:
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
        os.rmdir(temp_dir)
    except Exception as e:
        print(f"Cleanup error: {e}")


def main():
    st.title("Minimal Accent Classifier with SpeechBrain")

    if not check_limit():
        st.error(f"Daily limit of {DAILY_LIMIT} uses reached. Please try again tomorrow.")
        return

    youtube_url = st.text_input("Enter YouTube URL or another video URL")

    if st.button("Classify Accent") and youtube_url:
        with st.spinner("Downloading audio..."):
            try:
                wav_path, temp_dir = download_audio_from_youtube(youtube_url)
            except Exception as e:
                st.error(f"Error downloading audio: {e}")
                return

        st.audio(wav_path, format="audio/wav")

        with st.spinner("Preprocessing audio..."):
            signal = preprocess_audio(wav_path)

        model = load_model()

        with st.spinner("Predicting accent..."):
            accent, confidence = predict_accent(model, signal)

        st.success(f"Predicted Accent: {accent}")
        st.info(f"Confidence: {confidence:.2f}")

        # Clean up temp files
        cleanup_temp_dir(temp_dir)


if __name__ == "__main__":
    main()

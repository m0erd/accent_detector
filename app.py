import streamlit as st
import torchaudio
from speechbrain.inference import EncoderClassifier
import yt_dlp
import os
import tempfile
import torch


LABEL_MAP = {
    "england": "British English",
    "us": "American English",
    "australia": "Australian English",
    "indian": "Indian English",
    "scotland": "Scottish English",
    "canada": "Canadian English",
    "ireland": "Irish English",
}


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
            return os.path.join(temp_dir, f)
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

        print("DEBUG labels_list:", labels_list)

        label = labels_list[0].lower()
        confidence = max_score.item()

        return LABEL_MAP.get(label, f"Unknown English Accent ({label})"), confidence

    except Exception as e:
        return f"Prediction error: {str(e)}", 0.0


def main():
    st.title("Minimal Accent Classifier with SpeechBrain")
    youtube_url = st.text_input("Enter YouTube URL or another video url")

    if st.button("Classify Accent") and youtube_url:
        with st.spinner("Downloading audio..."):
            wav_path = download_audio_from_youtube(youtube_url)
        st.audio(wav_path, format="audio/wav")

        with st.spinner("Preprocessing audio..."):
            signal = preprocess_audio(wav_path)

        model = load_model()

        with st.spinner("Predicting accent..."):
            accent, confidence = predict_accent(model, signal)

        st.success(f"Predicted Accent: {accent}")
        st.info(f"Confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()

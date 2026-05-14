"""
Streamlit Web App — Speech Emotion Recognition
Run AFTER the notebook: streamlit run app.py
"""

import os, io, pickle, tempfile, warnings
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import librosa.effects

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-title {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem;
  }
  .subtitle { text-align: center; color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem; }
  .emotion-card {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #6366f1; border-radius: 16px;
    padding: 2rem; text-align: center; margin: 1rem 0;
  }
  .emotion-label { font-size: 3rem; font-weight: 900; color: #a5b4fc; }
  .emotion-emoji { font-size: 4rem; margin-bottom: 0.5rem; }
  .confidence { font-size: 1.4rem; color: #6ee7b7; font-weight: 600; }
  .metric-card {
    background: #0f172a; border: 1px solid #334155;
    border-radius: 12px; padding: 1rem; margin: 0.3rem 0;
  }
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.abspath(".")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PROCESSED_DIR = os.path.join(BASE_DIR, "artifacts", "processed")

SAMPLE_RATE  = 22050
DURATION     = 4
N_MFCC       = 40
N_MELS       = 128
N_CHROMA     = 12
HOP_LENGTH   = 512
N_FFT        = 2048
FIXED_FRAMES = 128

EMOTION_EMOJI = {
    "angry": "😡", "calm": "😌", "disgust": "🤢", "fearful": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprised": "😲",
}
EMOTION_COLOR = {
    "angry": "#ef4444", "calm": "#6ee7b7", "disgust": "#84cc16",
    "fearful": "#f59e0b", "happy": "#facc15", "neutral": "#94a3b8",
    "sad": "#60a5fa", "surprised": "#f0abfc",
}


# ── Load label map ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_label_map():
    # Try dedicated label_map.pkl first (written by notebook Step 20)
    lm_path = os.path.join(PROCESSED_DIR, "label_map.pkl")
    if os.path.exists(lm_path):
        with open(lm_path, "rb") as f:
            d = pickle.load(f)
        return d["label_map"], d["inv_label_map"]

    # Fallback: extract from features.pkl
    feat_path = os.path.join(PROCESSED_DIR, "features.pkl")
    if os.path.exists(feat_path):
        with open(feat_path, "rb") as f:
            d = pickle.load(f)
        lm = d["label_map"]
        return lm, {v: k for k, v in lm.items()}

    return None, None


# ── Audio helpers ──────────────────────────────────────────────────────────────
def load_audio(path):
    y, sr = librosa.load(path, sr=SAMPLE_RATE, duration=DURATION, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    S = np.abs(librosa.stft(y))
    nf = np.mean(S[:, :10], axis=1, keepdims=True)
    y = librosa.istft(S * (S > nf * 1.5))
    return y, sr


def pad_or_truncate(arr, target):
    if arr.shape[-1] < target:
        pad = [(0, 0)] * (arr.ndim - 1) + [(0, target - arr.shape[-1])]
        return np.pad(arr, pad, mode="constant")
    return arr[..., :target]


def extract_flat(path):
    y, sr = load_audio(path)
    feats = []
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta  = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feats += [np.mean(mfcc,1), np.std(mfcc,1), np.mean(delta,1), np.mean(delta2,1)]
    mel    = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.append(np.mean(librosa.power_to_db(mel, ref=np.max), axis=1))
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.append(np.mean(chroma, axis=1))
    feats.append(np.array([np.mean(librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH))]))
    feats.append(np.array([np.mean(librosa.feature.rms(y=y, hop_length=HOP_LENGTH))]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.append(np.mean(contrast, axis=1))
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    feats.append(np.mean(tonnetz, axis=1))
    return np.concatenate(feats).astype(np.float32)


def extract_mfcc_2d(path):
    y, sr = load_audio(path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return pad_or_truncate(mfcc, FIXED_FRAMES).astype(np.float32)


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(name):
    keras_path = os.path.join(MODELS_DIR, f"{name}.keras")
    pkl_path   = os.path.join(MODELS_DIR, f"{name}.pkl")
    if os.path.exists(keras_path):
        import tensorflow as tf
        return tf.keras.models.load_model(keras_path), "keras"
    elif os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f), "sklearn"
    return None, None


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(audio_bytes, model_key, inv_label_map):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        model, mtype = load_model(model_key)
        if model is None:
            return None

        if mtype == "keras":
            mfcc_2d = extract_mfcc_2d(tmp_path)
            if "lstm" in model_key and "cnn" not in model_key:
                X = np.transpose(mfcc_2d, (1, 0))[np.newaxis]
            else:
                X = mfcc_2d[np.newaxis, ..., np.newaxis]
            proba = model.predict(X, verbose=0)[0]
        else:
            flat = extract_flat(tmp_path)
            proba = model.predict_proba(flat[np.newaxis])[0]

        idx       = int(np.argmax(proba))
        emotion   = inv_label_map[idx]
        confidence = float(proba[idx])
        proba_dict = {inv_label_map[i]: round(float(p), 4) for i, p in enumerate(proba)}
        return {"emotion": emotion, "confidence": confidence, "probabilities": proba_dict}
    finally:
        os.unlink(tmp_path)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎙️ SER")
    st.markdown("---")
    model_options = {
        "CNN (Best - 89.6% F1)":  "cnn",
        "SVM (Best AUC - 0.993)": "svm",
        "Bi-LSTM":                "lstm",
        "XGBoost":                "xgboost",
        "Random Forest":          "random_forest",
        "CNN-LSTM":               "cnn_lstm",
    }
    selected_label = st.selectbox("Select Model", list(model_options.keys()))
    selected_key   = model_options[selected_label]

    # Show available models
    st.markdown("---")
    st.markdown("**Available Models:**")
    for label, key in model_options.items():
        keras_p = os.path.join(MODELS_DIR, f"{key}.keras")
        pkl_p   = os.path.join(MODELS_DIR, f"{key}.pkl")
        exists  = os.path.exists(keras_p) or os.path.exists(pkl_p)
        icon    = "✅" if exists else "⬜"
        st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.markdown("""
**8 Emotions:**
😡 Angry · 😌 Calm · 🤢 Disgust
😨 Fearful · 😄 Happy · 😐 Neutral
😢 Sad · 😲 Surprised

**Dataset:** RAVDESS (1,440 clips)
""")


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎙️ Speech Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload speech audio — AI detects the emotion in real time</div>',
    unsafe_allow_html=True,
)

label_map, inv_label_map = load_label_map()

if label_map is None:
    st.error(
        "⚠️ No trained models found. "
        "Please run **`notebooks/SER_Training.ipynb`** first to train and save the models, "
        "then relaunch this app."
    )
    st.stop()

uploaded = st.file_uploader(
    "Upload a speech audio file",
    type=["wav", "mp3", "ogg", "flac"],
    help="Works best with clear 3–5 second speech clips",
)

if uploaded:
    audio_bytes = uploaded.read()
    st.audio(audio_bytes, format="audio/wav")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    # ── Visualizations ──────────────────────────────────────────────────────────
    with col1:
        st.subheader("📊 Audio Visualizations")
        try:
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)

            tab1, tab2, tab3 = st.tabs(["🌊 Waveform", "🎨 Mel Spectrogram", "🎵 MFCC"])

            def dark_ax(ax, title):
                ax.set_facecolor("#1e293b")
                ax.set_title(title, color="#e2e8f0")
                ax.tick_params(colors="#94a3b8")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#334155")

            with tab1:
                fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0f172a")
                librosa.display.waveshow(y, sr=sr, ax=ax, color="#6366f1", alpha=0.85)
                dark_ax(ax, "Waveform")
                ax.set_xlabel("Time (s)", color="#94a3b8")
                ax.set_ylabel("Amplitude", color="#94a3b8")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with tab2:
                fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0f172a")
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                img = librosa.display.specshow(mel_db, sr=sr, x_axis="time",
                                               y_axis="mel", ax=ax, cmap="magma")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                dark_ax(ax, "Mel Spectrogram")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with tab3:
                fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0f172a")
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
                img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax, cmap="coolwarm")
                fig.colorbar(img, ax=ax)
                dark_ax(ax, "MFCC (40 coefficients)")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # Stats
            st.markdown("### 📈 Statistics")
            duration = len(y) / sr
            rms = float(np.mean(librosa.feature.rms(y=y)))
            zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
            c1, c2, c3 = st.columns(3)
            c1.metric("Duration", f"{duration:.2f}s")
            c2.metric("RMS Energy", f"{rms:.4f}")
            c3.metric("ZCR", f"{zcr:.4f}")

        except Exception as e:
            st.error(f"Visualization error: {e}")

    # ── Prediction ──────────────────────────────────────────────────────────────
    with col2:
        st.subheader("🤖 Emotion Prediction")

        model, mtype = load_model(selected_key)
        if model is None:
            st.warning(
                f"⚠️ Model **{selected_label}** not found. "
                "Run the notebook to train it, then choose another model."
            )
        else:
            with st.spinner("Analyzing emotion…"):
                try:
                    result = predict(audio_bytes, selected_key, inv_label_map)
                    if result:
                        emotion    = result["emotion"]
                        confidence = result["confidence"]
                        proba_dict = result["probabilities"]
                        emoji = EMOTION_EMOJI.get(emotion, "🎭")
                        color = EMOTION_COLOR.get(emotion, "#6366f1")

                        st.markdown(f"""
                        <div class="emotion-card">
                            <div class="emotion-emoji">{emoji}</div>
                            <div class="emotion-label" style="color:{color}">{emotion.upper()}</div>
                            <div class="confidence">Confidence: {confidence:.1%}</div>
                            <div style="color:#94a3b8;margin-top:0.5rem;font-size:0.85rem">
                                Model: {selected_label}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Probability bars
                        st.markdown("### Probability Distribution")
                        for emo, prob in sorted(proba_dict.items(), key=lambda x: -x[1]):
                            c_l, c_r = st.columns([3, 1])
                            c_l.markdown(f"{EMOTION_EMOJI.get(emo,'🎭')} **{emo.capitalize()}**")
                            c_r.markdown(f"**{prob:.1%}**")
                            st.progress(float(prob))

                        # Pie chart
                        st.markdown("### Breakdown")
                        sorted_p = sorted(proba_dict.items(), key=lambda x: -x[1])
                        emos_   = [e for e, _ in sorted_p]
                        probs_  = [p for _, p in sorted_p]
                        colors_ = [EMOTION_COLOR.get(e, "#6366f1") for e in emos_]

                        fig, ax = plt.subplots(figsize=(5, 4.5), facecolor="#0f172a")
                        ax.set_facecolor("#0f172a")
                        wedges, texts, autotexts = ax.pie(
                            probs_,
                            labels=[f"{EMOTION_EMOJI.get(e,'🎭')} {e}" for e in emos_],
                            colors=colors_, autopct="%1.1f%%", startangle=140,
                        )
                        for t in texts:
                            t.set_color("#e2e8f0"); t.set_fontsize(8)
                        for at in autotexts:
                            at.set_color("#000"); at.set_fontsize(7); at.set_fontweight("bold")
                        plt.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close()

                except Exception as e:
                    st.error(f"Prediction error: {e}")

else:
    # Landing
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem">
        <div style="font-size:5rem;margin-bottom:1rem">🎙️</div>
        <h3 style="color:#94a3b8">Upload a speech recording to get started</h3>
        <p style="color:#64748b">Supports WAV · MP3 · OGG · FLAC<br>Best with 3–5 second clear speech clips</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (c1, "🧠", "6 ML Models", "RF · SVM · XGBoost\nCNN · Bi-LSTM · CNN-LSTM"),
        (c2, "🎵", "Rich Features", "MFCC · Mel Spectrogram\nChroma · Tonnetz · ZCR"),
        (c3, "😊", "8 Emotions", "Happy · Sad · Angry · Calm\nFearful · Disgust · Surprised · Neutral"),
    ]:
        col.markdown(f"""
        <div class="metric-card" style="text-align:center">
            <div style="font-size:2.5rem">{icon}</div>
            <h4 style="color:#a5b4fc">{title}</h4>
            <p style="color:#64748b;font-size:0.85rem;white-space:pre-line">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

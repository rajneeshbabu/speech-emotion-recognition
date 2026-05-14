"""
Feature Extraction — extracts audio features from RAVDESS .wav files.

Features extracted per file:
  • MFCC (40 coeffs) + delta + delta2  → 120 values
  • Mel Spectrogram (128 bins)         → 128 values
  • Chroma (12 bins)                   → 12 values
  • Zero Crossing Rate                 → 1 value
  • RMS Energy                         → 1 value
  • Spectral Contrast (7 bands)        → 7 values
  • Tonnetz (6 dims)                   → 6 values
  Total flat vector                    → ~275 values (mean of time-axis)

2-D tensors (for CNN / LSTM):
  • MFCC image  shape (40, T)
  • Mel image   shape (128, T)
"""

import os
import pickle
import numpy as np
import librosa
import librosa.effects
from tqdm import tqdm
from typing import Tuple, Optional

from src.logger import get_logger
from src.exception import SERException
from src.config import ProjectConfig, load_config

logger = get_logger(__name__)

# Fixed time dimension for 2-D feature tensors (pad / truncate)
FIXED_TIME_FRAMES = 128


# ── Low-level helpers ──────────────────────────────────────────────────────────

def _load_audio(filepath: str, cfg: ProjectConfig) -> Tuple[np.ndarray, int]:
    """Load a wav file, optionally trim silence and reduce noise."""
    try:
        y, sr = librosa.load(filepath, sr=cfg.audio.sample_rate,
                             duration=cfg.audio.duration, mono=True)

        if cfg.audio.silence_trim:
            y, _ = librosa.effects.trim(y, top_db=cfg.audio.trim_top_db)

        if cfg.audio.noise_reduction:
            # Simple spectral gating: suppress bins below mean noise floor
            S = np.abs(librosa.stft(y))
            noise_floor = np.mean(S[:, :10], axis=1, keepdims=True)  # first 10 frames
            mask = S > noise_floor * 1.5
            S_clean = S * mask
            y = librosa.istft(S_clean)

        return y, sr
    except Exception as e:
        raise SERException(e)


def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad (zero) or truncate along last axis to target_len."""
    if arr.shape[-1] < target_len:
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_len - arr.shape[-1])]
        return np.pad(arr, pad_width, mode="constant")
    return arr[..., :target_len]


# ── Flat feature vector ────────────────────────────────────────────────────────

def extract_flat_features(filepath: str, cfg: ProjectConfig) -> np.ndarray:
    """
    Extract a 1-D flat feature vector from a single audio file.
    Returns np.ndarray of shape (n_features,).
    """
    y, sr = _load_audio(filepath, cfg)

    features = []

    # MFCC + delta + delta2
    mfcc = librosa.feature.mfcc(y=y, sr=sr,
                                 n_mfcc=cfg.audio.n_mfcc,
                                 n_fft=cfg.audio.n_fft,
                                 hop_length=cfg.audio.hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.append(np.mean(mfcc, axis=1))
    features.append(np.std(mfcc, axis=1))
    features.append(np.mean(mfcc_delta, axis=1))
    features.append(np.mean(mfcc_delta2, axis=1))

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                          n_mels=cfg.audio.n_mels,
                                          n_fft=cfg.audio.n_fft,
                                          hop_length=cfg.audio.hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    features.append(np.mean(mel_db, axis=1))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_chroma=cfg.audio.n_chroma,
                                          n_fft=cfg.audio.n_fft,
                                          hop_length=cfg.audio.hop_length)
    features.append(np.mean(chroma, axis=1))

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=cfg.audio.hop_length)
    features.append(np.array([np.mean(zcr)]))

    # RMS Energy
    rms = librosa.feature.rms(y=y, hop_length=cfg.audio.hop_length)
    features.append(np.array([np.mean(rms)]))

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                   n_fft=cfg.audio.n_fft,
                                                   hop_length=cfg.audio.hop_length)
    features.append(np.mean(contrast, axis=1))

    # Tonnetz
    harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    features.append(np.mean(tonnetz, axis=1))

    return np.concatenate(features).astype(np.float32)


# ── 2-D feature tensors ────────────────────────────────────────────────────────

def extract_2d_features(filepath: str, cfg: ProjectConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract 2-D MFCC and Mel Spectrogram tensors (for CNN / LSTM).
    Returns (mfcc_2d, mel_2d), each shape (n_coeffs, FIXED_TIME_FRAMES).
    """
    y, sr = _load_audio(filepath, cfg)

    mfcc = librosa.feature.mfcc(y=y, sr=sr,
                                 n_mfcc=cfg.audio.n_mfcc,
                                 n_fft=cfg.audio.n_fft,
                                 hop_length=cfg.audio.hop_length)
    mfcc_2d = _pad_or_truncate(mfcc, FIXED_TIME_FRAMES)

    mel = librosa.feature.melspectrogram(y=y, sr=sr,
                                          n_mels=cfg.audio.n_mels,
                                          n_fft=cfg.audio.n_fft,
                                          hop_length=cfg.audio.hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_2d = _pad_or_truncate(mel_db, FIXED_TIME_FRAMES)

    return mfcc_2d.astype(np.float32), mel_2d.astype(np.float32)


# ── Batch extraction ───────────────────────────────────────────────────────────

def extract_features_for_dataset(
    catalogue: list[dict],
    cfg: ProjectConfig,
    extract_2d: bool = True,
) -> dict:
    """
    Batch-extract features for every entry in catalogue.

    Returns dict with keys:
      'X_flat'   : np.ndarray (N, n_features)
      'X_mfcc'   : np.ndarray (N, n_mfcc, T)       — if extract_2d
      'X_mel'    : np.ndarray (N, n_mels, T)        — if extract_2d
      'y'        : list[str]  emotion labels
      'label_map': dict  label_str → int
    """
    cache_path = os.path.join(cfg.dataset["processed_dir"], "features.pkl")

    if os.path.exists(cache_path):
        logger.info("Loading features from cache.")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    logger.info(f"Extracting features for {len(catalogue)} files…")

    X_flat, X_mfcc, X_mel, labels = [], [], [], []
    errors = 0

    for record in tqdm(catalogue, desc="Feature extraction"):
        try:
            flat = extract_flat_features(record["path"], cfg)
            X_flat.append(flat)
            labels.append(record["emotion"])

            if extract_2d:
                mfcc_2d, mel_2d = extract_2d_features(record["path"], cfg)
                X_mfcc.append(mfcc_2d)
                X_mel.append(mel_2d)

        except Exception as e:
            logger.warning(f"Skipping {record['path']}: {e}")
            errors += 1

    logger.info(f"Extraction complete. {len(X_flat)} succeeded, {errors} failed.")

    # Build label encoding
    unique_labels = sorted(set(labels))
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}

    result = {
        "X_flat": np.array(X_flat, dtype=np.float32),
        "y": labels,
        "label_map": label_map,
    }

    if extract_2d:
        result["X_mfcc"] = np.array(X_mfcc, dtype=np.float32)
        result["X_mel"] = np.array(X_mel, dtype=np.float32)

    os.makedirs(cfg.dataset["processed_dir"], exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    logger.info(f"Features cached to {cache_path}")

    return result


# ── Entry point ────────────────────────────────────────────────────────────────

def run_feature_extraction(catalogue: list[dict], cfg: ProjectConfig = None) -> dict:
    if cfg is None:
        cfg = load_config()
    logger.info("═══ Feature Extraction ═══")
    return extract_features_for_dataset(catalogue, cfg)


if __name__ == "__main__":
    from src.data_ingestion import run_ingestion
    cfg = load_config()
    catalogue = run_ingestion(cfg)
    run_feature_extraction(catalogue, cfg)

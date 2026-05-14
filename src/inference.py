"""
Inference — load the best saved model and predict emotion from a single audio file.
"""

import os
import pickle
import numpy as np
from typing import Optional

from src.logger import get_logger
from src.exception import SERException
from src.config import ProjectConfig, load_config
from src.feature_extraction import extract_flat_features, extract_2d_features

logger = get_logger(__name__)

# Default best model name (update after training to whichever performed best)
DEFAULT_MODEL = "cnn_lstm"


def _load_classical(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_keras(path: str):
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except ImportError:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")


def _load_label_map(cfg: ProjectConfig) -> dict:
    cache_path = os.path.join(cfg.dataset["processed_dir"], "features.pkl")
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Features cache not found at {cache_path}. "
            "Run the training pipeline first."
        )
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    return data["label_map"]   # label_str → int


def load_model(model_name: str, cfg: ProjectConfig):
    """
    Load a saved model by name.
    Tries .keras first, then .pkl.
    """
    keras_path = os.path.join(cfg.paths["models_dir"], f"{model_name}.keras")
    pkl_path = os.path.join(cfg.paths["models_dir"], f"{model_name}.pkl")

    if os.path.exists(keras_path):
        logger.info(f"Loading Keras model from {keras_path}")
        return _load_keras(keras_path), "keras"
    elif os.path.exists(pkl_path):
        logger.info(f"Loading sklearn model from {pkl_path}")
        return _load_classical(pkl_path), "sklearn"
    else:
        raise FileNotFoundError(
            f"No model found for '{model_name}' in {cfg.paths['models_dir']}. "
            "Available: random_forest, svm, xgboost, cnn, lstm, cnn_lstm"
        )


def predict(
    audio_path: str,
    cfg: ProjectConfig,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Predict the emotion of an audio file.

    Returns dict:
        {
          'emotion': str,
          'confidence': float,
          'probabilities': {emotion_str: probability, ...}
        }
    """
    try:
        model, model_type = load_model(model_name, cfg)
        label_map = _load_label_map(cfg)
        inv_map = {v: k for k, v in label_map.items()}  # int → label_str
        n_classes = len(label_map)

        if model_type == "keras":
            mfcc_2d, mel_2d = extract_2d_features(audio_path, cfg)
            model_name_lower = model_name.lower()

            if "cnn" in model_name_lower and "lstm" not in model_name_lower:
                X = mfcc_2d[np.newaxis, ..., np.newaxis]  # (1, n_mfcc, T, 1)
            elif "lstm" in model_name_lower and "cnn" not in model_name_lower:
                X = np.transpose(mfcc_2d, (1, 0))[np.newaxis]  # (1, T, n_mfcc)
            else:
                X = mfcc_2d[np.newaxis, ..., np.newaxis]  # (1, n_mfcc, T, 1) for CNN-LSTM

            proba = model.predict(X, verbose=0)[0]

        else:  # sklearn
            flat = extract_flat_features(audio_path, cfg)
            X = flat[np.newaxis, :]  # (1, n_features)
            proba = model.predict_proba(X)[0]

        predicted_idx = int(np.argmax(proba))
        predicted_emotion = inv_map[predicted_idx]
        confidence = float(proba[predicted_idx])

        proba_dict = {inv_map[i]: round(float(p), 4) for i, p in enumerate(proba)}

        logger.info(f"Prediction: {predicted_emotion} (confidence={confidence:.4f})")
        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "probabilities": proba_dict,
        }

    except Exception as e:
        raise SERException(e)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.inference <audio_file.wav> [model_name]")
        sys.exit(1)

    audio = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    cfg = load_config()
    result = predict(audio, cfg, model_name=model)
    print(f"\nPredicted Emotion : {result['emotion']}")
    print(f"Confidence        : {result['confidence']:.2%}")
    print(f"All probabilities :")
    for emo, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {emo:<12} {prob:.4f}  {bar}")

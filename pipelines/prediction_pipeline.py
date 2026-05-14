"""
Prediction Pipeline — load best model and predict from a file path or bytes.
"""

import os
import io
import numpy as np
import tempfile
from typing import Union

from src.logger import get_logger
from src.config import load_config, ProjectConfig
from src.inference import predict, load_model, DEFAULT_MODEL

logger = get_logger(__name__)


def run_prediction(
    audio_input: Union[str, bytes],
    cfg: ProjectConfig = None,
    model_name: str = DEFAULT_MODEL,
) -> dict:
    """
    Predict emotion from:
      - audio_input: file path (str) OR raw audio bytes (bytes from Streamlit uploader)
      - model_name: which saved model to use

    Returns dict: {emotion, confidence, probabilities}
    """
    if cfg is None:
        cfg = load_config()

    if isinstance(audio_input, bytes):
        # Write bytes to a temp file so librosa can read it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_input)
            tmp_path = tmp.name
        try:
            result = predict(tmp_path, cfg, model_name=model_name)
        finally:
            os.unlink(tmp_path)
    else:
        result = predict(audio_input, cfg, model_name=model_name)

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipelines.prediction_pipeline <audio.wav> [model_name]")
        sys.exit(1)
    audio = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_MODEL
    cfg = load_config()
    result = run_prediction(audio, cfg, model_name=model)
    print(f"\nEmotion    : {result['emotion']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print("\nProbabilities:")
    for emo, p in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"  {emo:<12} {p:.4f}  {bar}")

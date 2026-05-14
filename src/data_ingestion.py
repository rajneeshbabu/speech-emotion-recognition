"""
Data Ingestion — downloads RAVDESS dataset from Kaggle and organises it.

Usage:
    python -m src.data_ingestion

Requires:
    ~/.kaggle/kaggle.json  OR  env vars KAGGLE_USERNAME + KAGGLE_KEY
"""
import os
import glob
import pickle
import zipfile
import shutil
from pathlib import Path

from src.logger import get_logger
from src.exception import SERException
from src.config import load_config, ProjectConfig

logger = get_logger(__name__)


# ── RAVDESS filename encoding ──────────────────────────────────────────────────
# Format: 03-01-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
# Emotion codes: 01=neutral,02=calm,03=happy,04=sad,05=angry,06=fearful,07=disgust,08=surprised

def _parse_ravdess_label(filepath: str, emotion_map: dict) -> str | None:
    """Return emotion string from a RAVDESS .wav filename, or None if not parseable."""
    fname = os.path.basename(filepath)
    parts = fname.replace(".wav", "").split("-")
    if len(parts) < 7:
        return None
    modality = parts[0]   # 03 = speech-only
    emotion_code = parts[2]
    if modality != "03":  # skip song tracks
        return None
    return emotion_map.get(emotion_code)


# ── Kaggle download ────────────────────────────────────────────────────────────

def _setup_kaggle_credentials() -> None:
    """
    Check for credentials in env vars; if present, write ~/.kaggle/kaggle.json.
    """
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    if os.path.exists(kaggle_json):
        return

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        os.makedirs(kaggle_dir, exist_ok=True)
        import json
        with open(kaggle_json, "w") as f:
            json.dump({"username": username, "key": key}, f)
        os.chmod(kaggle_json, 0o600)
        logger.info("Kaggle credentials written from environment variables.")
    else:
        raise SERException(
            FileNotFoundError(
                "Kaggle credentials not found.\n"
                "Either place ~/.kaggle/kaggle.json OR set env vars:\n"
                "  export KAGGLE_USERNAME=your_username\n"
                "  export KAGGLE_KEY=your_api_key"
            )
        )


def download_dataset(cfg: ProjectConfig) -> str:
    """
    Download RAVDESS from Kaggle into artifacts/raw/.
    Returns path to the extracted directory.
    """
    raw_dir = cfg.dataset["raw_dir"]
    dataset_slug = cfg.dataset["kaggle_dataset"]
    extract_dir = os.path.join(raw_dir, "ravdess")

    # Check cache
    if os.path.exists(extract_dir) and len(glob.glob(f"{extract_dir}/**/*.wav", recursive=True)) > 100:
        logger.info(f"✅ Dataset already cached at {extract_dir}")
        return extract_dir

    logger.info(f"Downloading dataset: {dataset_slug}")
    _setup_kaggle_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()

        os.makedirs(raw_dir, exist_ok=True)
        api.dataset_download_files(
            dataset_slug,
            path=raw_dir,
            unzip=True,
            quiet=False,
        )
        logger.info(f"✅ Downloaded and extracted to {raw_dir}")

    except Exception as e:
        raise SERException(e)

    # Locate extracted wav files
    wav_files = glob.glob(f"{raw_dir}/**/*.wav", recursive=True)
    logger.info(f"Found {len(wav_files)} .wav files")
    return raw_dir


# ── File catalogue ────────────────────────────────────────────────────────────

def build_file_catalogue(raw_dir: str, cfg: ProjectConfig) -> list[dict]:
    """
    Walk raw_dir, parse every RAVDESS .wav filename, return list of
    {'path': ..., 'emotion': ...} dicts.
    """
    catalogue_path = os.path.join(cfg.dataset["processed_dir"], "catalogue.pkl")

    if os.path.exists(catalogue_path):
        logger.info("Loading catalogue from cache.")
        with open(catalogue_path, "rb") as f:
            return pickle.load(f)

    wav_files = glob.glob(f"{raw_dir}/**/*.wav", recursive=True)
    logger.info(f"Building catalogue from {len(wav_files)} files…")

    records = []
    skipped = 0
    for fp in wav_files:
        label = _parse_ravdess_label(fp, cfg.emotions)
        if label is None:
            skipped += 1
            continue
        records.append({"path": fp, "emotion": label})

    logger.info(f"Catalogue built: {len(records)} samples, {skipped} skipped.")

    # Emotion distribution
    from collections import Counter
    dist = Counter(r["emotion"] for r in records)
    logger.info(f"Emotion distribution: {dict(dist)}")

    os.makedirs(cfg.dataset["processed_dir"], exist_ok=True)
    with open(catalogue_path, "wb") as f:
        pickle.dump(records, f)

    return records


# ── Entry point ───────────────────────────────────────────────────────────────

def run_ingestion(cfg: ProjectConfig = None) -> list[dict]:
    if cfg is None:
        cfg = load_config()

    logger.info("═══ Data Ingestion ═══")
    raw_dir = download_dataset(cfg)
    catalogue = build_file_catalogue(raw_dir, cfg)
    logger.info(f"Ingestion complete. {len(catalogue)} usable samples.")
    return catalogue


if __name__ == "__main__":
    run_ingestion()

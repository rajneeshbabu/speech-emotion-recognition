"""
Load and expose project configuration from config/config.yaml.
"""
import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


def _load_yaml(path: str) -> dict:
    with open(os.path.abspath(path), "r") as f:
        return yaml.safe_load(f)


@dataclass
class AudioConfig:
    sample_rate: int
    duration: int
    n_mfcc: int
    n_mels: int
    n_chroma: int
    hop_length: int
    n_fft: int
    noise_reduction: bool
    silence_trim: bool
    trim_top_db: int


@dataclass
class SplitConfig:
    test_size: float
    val_size: float
    random_state: int
    stratify: bool


@dataclass
class DeepLearningConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    dropout: float
    cnn: Dict[str, Any]
    lstm: Dict[str, Any]
    cnn_lstm: Dict[str, Any]


@dataclass
class ProjectConfig:
    # raw dicts
    raw: dict = field(default_factory=dict)

    # typed sub-configs
    audio: AudioConfig = None
    split: SplitConfig = None
    deep_learning: DeepLearningConfig = None

    # convenience accessors
    emotions: Dict[str, str] = field(default_factory=dict)
    dataset: Dict[str, str] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    pca: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    classical_models: Dict[str, Any] = field(default_factory=dict)


def load_config(path: str = _CONFIG_PATH) -> ProjectConfig:
    raw = _load_yaml(path)
    cfg = ProjectConfig(raw=raw)

    # Audio
    a = raw["audio"]
    cfg.audio = AudioConfig(**a)

    # Split
    cfg.split = SplitConfig(**raw["split"])

    # Deep learning
    dl = raw["deep_learning"]
    cfg.deep_learning = DeepLearningConfig(**dl)

    # Simple dicts
    cfg.emotions = raw["emotions"]
    cfg.dataset = raw["dataset"]
    cfg.features = raw["features"]
    cfg.paths = raw["paths"]
    cfg.pca = raw["pca"]
    cfg.evaluation = raw["evaluation"]
    cfg.classical_models = raw["classical_models"]

    # Ensure directories exist
    for key in ("models_dir", "logs_dir", "artifacts_dir"):
        os.makedirs(cfg.paths[key], exist_ok=True)
    os.makedirs(raw["dataset"]["raw_dir"], exist_ok=True)
    os.makedirs(raw["dataset"]["processed_dir"], exist_ok=True)
    os.makedirs(raw["evaluation"]["plots_dir"], exist_ok=True)

    return cfg

"""
Training Pipeline — orchestrates the full ML workflow:
  1. Data Ingestion  (download + catalogue)
  2. Feature Extraction  (flat + 2D)
  3. Data Split  (train / val / test)
  4. Model Training  (RF, SVM, XGBoost, CNN, LSTM, CNN-LSTM)
  5. Evaluation  (metrics + plots + SHAP)
"""

import numpy as np
from sklearn.model_selection import train_test_split

from src.logger import get_logger
from src.config import load_config, ProjectConfig
from src.data_ingestion import run_ingestion
from src.feature_extraction import run_feature_extraction
from src.model_trainer import encode_labels, train_all_models
from src.model_evaluation import evaluate_all_models

logger = get_logger(__name__)


def run_training_pipeline(cfg: ProjectConfig = None) -> dict:
    """
    End-to-end training pipeline.
    Returns the evaluation results dict.
    """
    if cfg is None:
        cfg = load_config()

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║   Speech Emotion Recognition — Train  ║")
    logger.info("╚══════════════════════════════════════╝")

    # ── 1. Data Ingestion ──────────────────────────────────────────────────────
    catalogue = run_ingestion(cfg)

    # ── 2. Feature Extraction ─────────────────────────────────────────────────
    feature_data = run_feature_extraction(catalogue, cfg)

    X_flat = feature_data["X_flat"]
    X_mfcc = feature_data.get("X_mfcc")
    X_mel = feature_data.get("X_mel")
    y_str = feature_data["y"]
    label_map = feature_data["label_map"]
    label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    n_classes = len(label_map)

    y_enc = encode_labels(y_str, label_map)

    logger.info(f"Dataset: {len(y_enc)} samples | {n_classes} classes")
    logger.info(f"Labels: {label_names}")
    logger.info(f"Flat feature shape: {X_flat.shape}")
    if X_mfcc is not None:
        logger.info(f"MFCC 2D shape: {X_mfcc.shape}")

    # ── 3. Train / Val / Test Split ───────────────────────────────────────────
    split_cfg = cfg.split

    X_flat_tr, X_flat_tmp, X_mfcc_tr, X_mfcc_tmp, y_tr, y_tmp = train_test_split(
        X_flat, X_mfcc if X_mfcc is not None else X_flat,
        y_enc,
        test_size=split_cfg.test_size + split_cfg.val_size,
        random_state=split_cfg.random_state,
        stratify=y_enc if split_cfg.stratify else None,
    )

    val_ratio = split_cfg.val_size / (split_cfg.test_size + split_cfg.val_size)
    X_flat_val, X_flat_test, X_mfcc_val, X_mfcc_test, y_val, y_test = train_test_split(
        X_flat_tmp,
        X_mfcc_tmp,
        y_tmp,
        test_size=1.0 - val_ratio,
        random_state=split_cfg.random_state,
        stratify=y_tmp if split_cfg.stratify else None,
    )

    logger.info(
        f"Split — train: {len(y_tr)} | val: {len(y_val)} | test: {len(y_test)}"
    )

    # ── 4. Model Training ─────────────────────────────────────────────────────
    models = train_all_models(
        X_flat_train=X_flat_tr,
        X_flat_val=X_flat_val,
        X_mfcc_train=X_mfcc_tr,
        X_mfcc_val=X_mfcc_val,
        y_train_enc=y_tr,
        y_val_enc=y_val,
        n_classes=n_classes,
        cfg=cfg,
    )

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    results = evaluate_all_models(
        models=models,
        X_flat_test=X_flat_test,
        X_mfcc_test=X_mfcc_test,
        y_test=y_test,
        label_names=label_names,
        cfg=cfg,
    )

    logger.info("╔══════════════════════════════════════╗")
    logger.info("║         Training Complete!            ║")
    logger.info("╚══════════════════════════════════════╝")
    logger.info("Performance Summary (sorted by F1-Weighted):")
    for r in results:
        logger.info(
            f"  {r['name']:<20} Acc={r['accuracy']:.4f} "
            f"F1-W={r['f1_weighted']:.4f} "
            f"AUC={r.get('roc_auc', 'N/A')}"
        )

    return results


if __name__ == "__main__":
    run_training_pipeline()

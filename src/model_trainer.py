"""
Model Trainer — trains 6 models on SER features.

Classical (sklearn):
  1. Random Forest
  2. SVM (RBF kernel)
  3. XGBoost

Deep Learning (TensorFlow/Keras):
  4. CNN
  5. LSTM
  6. CNN-LSTM Hybrid

Each model is saved to artifacts/models/ after training.
Hyperparameter tuning: RandomizedSearchCV (classical) / manual grid (deep).
"""

import os
import pickle
import json
import numpy as np
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — XGBoost model will be skipped.")

from src.logger import get_logger
from src.exception import SERException
from src.config import ProjectConfig, load_config

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

def _model_path(name: str, cfg: ProjectConfig) -> str:
    return os.path.join(cfg.paths["models_dir"], f"{name}.pkl")


def _dl_model_path(name: str, cfg: ProjectConfig) -> str:
    return os.path.join(cfg.paths["models_dir"], f"{name}.keras")


# ── Label encoding ─────────────────────────────────────────────────────────────

def encode_labels(y_str: list[str], label_map: dict) -> np.ndarray:
    return np.array([label_map[lbl] for lbl in y_str], dtype=np.int32)


# ── Classical models ───────────────────────────────────────────────────────────

def _build_classical_pipeline(model, use_pca: bool, cfg: ProjectConfig) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=cfg.pca["n_components"],
                                  random_state=cfg.pca["random_state"])))
    steps.append(("model", model))
    return Pipeline(steps)


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                         cfg: ProjectConfig) -> Pipeline:
    logger.info("Training Random Forest…")
    save_path = _model_path("random_forest", cfg)

    if os.path.exists(save_path):
        logger.info("RF model found in cache, loading.")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    param_grid = cfg.classical_models["random_forest"]["param_grid"]
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    pipeline = _build_classical_pipeline(base_model, cfg.pca["use_pca"], cfg)

    # Prefix pipeline params
    prefixed = {f"model__{k}": v for k, v in param_grid.items()}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, prefixed, n_iter=10, scoring="f1_weighted",
        cv=cv, n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    logger.info(f"RF best params: {search.best_params_}  CV f1={search.best_score_:.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(best, f)
    return best


def train_svm(X_train: np.ndarray, y_train: np.ndarray,
               cfg: ProjectConfig) -> Pipeline:
    logger.info("Training SVM…")
    save_path = _model_path("svm", cfg)

    if os.path.exists(save_path):
        logger.info("SVM model found in cache, loading.")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    param_grid = cfg.classical_models["svm"]["param_grid"]
    base_model = SVC(kernel="rbf", probability=True, random_state=42)
    pipeline = _build_classical_pipeline(base_model, cfg.pca["use_pca"], cfg)
    prefixed = {f"model__{k}": v for k, v in param_grid.items()}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, prefixed, n_iter=8, scoring="f1_weighted",
        cv=cv, n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    logger.info(f"SVM best params: {search.best_params_}  CV f1={search.best_score_:.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(best, f)
    return best


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                   cfg: ProjectConfig) -> Optional[object]:
    if not HAS_XGB:
        logger.warning("XGBoost not installed, skipping.")
        return None

    logger.info("Training XGBoost…")
    save_path = _model_path("xgboost", cfg)

    if os.path.exists(save_path):
        logger.info("XGBoost model found in cache, loading.")
        with open(save_path, "rb") as f:
            return pickle.load(f)

    n_classes = len(np.unique(y_train))
    param_grid = cfg.classical_models["xgboost"]["param_grid"]
    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = _build_classical_pipeline(base_model, cfg.pca["use_pca"], cfg)
    prefixed = {f"model__{k}": v for k, v in param_grid.items()}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline, prefixed, n_iter=8, scoring="f1_weighted",
        cv=cv, n_jobs=-1, random_state=42, verbose=1
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_
    logger.info(f"XGB best params: {search.best_params_}  CV f1={search.best_score_:.4f}")

    with open(save_path, "wb") as f:
        pickle.dump(best, f)
    return best


# ── Deep Learning models ───────────────────────────────────────────────────────

def _get_tf():
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        raise ImportError("TensorFlow is not installed. Run: pip install tensorflow")


def train_cnn(X_mfcc: np.ndarray, y_train: np.ndarray,
               X_val_mfcc: np.ndarray, y_val: np.ndarray,
               n_classes: int, cfg: ProjectConfig):
    """CNN on MFCC 2-D feature maps. Input shape: (n_mfcc, T, 1)."""
    tf = _get_tf()
    save_path = _dl_model_path("cnn", cfg)

    if os.path.exists(save_path):
        logger.info("CNN model found in cache, loading.")
        return tf.keras.models.load_model(save_path)

    logger.info("Training CNN…")
    X_tr = X_mfcc[..., np.newaxis]
    X_v = X_val_mfcc[..., np.newaxis]

    dl = cfg.deep_learning
    cnn_cfg = dl.cnn

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_tr.shape[1:]),
        tf.keras.layers.Conv2D(cnn_cfg["filters"][0], (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dl.dropout),

        tf.keras.layers.Conv2D(cnn_cfg["filters"][1], (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dl.dropout),

        tf.keras.layers.Conv2D(cnn_cfg["filters"][2], (3, 3), activation="relu", padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dl.dropout),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(dl.dropout),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="CNN")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(dl.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=dl.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=dl.patience // 2),
    ]
    model.fit(X_tr, y_train, validation_data=(X_v, y_val),
              epochs=dl.epochs, batch_size=dl.batch_size,
              callbacks=callbacks, verbose=1)

    model.save(save_path)
    logger.info(f"CNN saved to {save_path}")
    return model


def train_lstm(X_mfcc: np.ndarray, y_train: np.ndarray,
                X_val_mfcc: np.ndarray, y_val: np.ndarray,
                n_classes: int, cfg: ProjectConfig):
    """Bi-LSTM on MFCC sequences. Input shape: (T, n_mfcc)."""
    tf = _get_tf()
    save_path = _dl_model_path("lstm", cfg)

    if os.path.exists(save_path):
        logger.info("LSTM model found in cache, loading.")
        return tf.keras.models.load_model(save_path)

    logger.info("Training Bi-LSTM…")
    # Transpose: (N, n_mfcc, T) → (N, T, n_mfcc)
    X_tr = np.transpose(X_mfcc, (0, 2, 1))
    X_v = np.transpose(X_val_mfcc, (0, 2, 1))

    dl = cfg.deep_learning
    lstm_cfg = dl.lstm

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=X_tr.shape[1:]),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_cfg["units"][0], return_sequences=True)
        ),
        tf.keras.layers.Dropout(dl.dropout),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_cfg["units"][1], return_sequences=False)
        ),
        tf.keras.layers.Dropout(dl.dropout),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(dl.dropout),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ], name="BiLSTM")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(dl.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=dl.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=dl.patience // 2),
    ]
    model.fit(X_tr, y_train, validation_data=(X_v, y_val),
              epochs=dl.epochs, batch_size=dl.batch_size,
              callbacks=callbacks, verbose=1)

    model.save(save_path)
    logger.info(f"LSTM saved to {save_path}")
    return model


def train_cnn_lstm(X_mfcc: np.ndarray, y_train: np.ndarray,
                    X_val_mfcc: np.ndarray, y_val: np.ndarray,
                    n_classes: int, cfg: ProjectConfig):
    """CNN-LSTM hybrid: CNN extracts local patterns, LSTM captures temporal context."""
    tf = _get_tf()
    save_path = _dl_model_path("cnn_lstm", cfg)

    if os.path.exists(save_path):
        logger.info("CNN-LSTM model found in cache, loading.")
        return tf.keras.models.load_model(save_path)

    logger.info("Training CNN-LSTM Hybrid…")
    # Input: (N, n_mfcc, T) → add channel dim → (N, n_mfcc, T, 1)
    X_tr = X_mfcc[..., np.newaxis]
    X_v = X_val_mfcc[..., np.newaxis]

    dl = cfg.deep_learning
    cnn_lstm_cfg = dl.cnn_lstm

    inp = tf.keras.layers.Input(shape=X_tr.shape[1:])

    # CNN block
    x = tf.keras.layers.Conv2D(cnn_lstm_cfg["cnn_filters"][0], (3, 3), activation="relu", padding="same")(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)
    x = tf.keras.layers.Conv2D(cnn_lstm_cfg["cnn_filters"][1], (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 1))(x)

    # Reshape for LSTM: (N, time, features)
    sh = x.shape
    x = tf.keras.layers.Reshape((sh[2], sh[1] * sh[3]))(x)

    # LSTM block
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(cnn_lstm_cfg["lstm_units"], return_sequences=False)
    )(x)
    x = tf.keras.layers.Dropout(dl.dropout)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(dl.dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out, name="CNN_LSTM_Hybrid")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(dl.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=dl.patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=dl.patience // 2),
    ]
    model.fit(X_tr, y_train, validation_data=(X_v, y_val),
              epochs=dl.epochs, batch_size=dl.batch_size,
              callbacks=callbacks, verbose=1)

    model.save(save_path)
    logger.info(f"CNN-LSTM saved to {save_path}")
    return model


# ── Train all ──────────────────────────────────────────────────────────────────

def train_all_models(
    X_flat_train, X_flat_val,
    X_mfcc_train, X_mfcc_val,
    y_train_enc, y_val_enc,
    n_classes: int,
    cfg: ProjectConfig,
) -> dict:
    """
    Train all 6 models and return a dict {name: fitted_model}.
    """
    models = {}

    # Classical
    models["Random Forest"] = train_random_forest(X_flat_train, y_train_enc, cfg)
    models["SVM"] = train_svm(X_flat_train, y_train_enc, cfg)
    xgb = train_xgboost(X_flat_train, y_train_enc, cfg)
    if xgb is not None:
        models["XGBoost"] = xgb

    # Deep learning
    models["CNN"] = train_cnn(X_mfcc_train, y_train_enc, X_mfcc_val, y_val_enc, n_classes, cfg)
    models["LSTM"] = train_lstm(X_mfcc_train, y_train_enc, X_mfcc_val, y_val_enc, n_classes, cfg)
    models["CNN-LSTM"] = train_cnn_lstm(X_mfcc_train, y_train_enc, X_mfcc_val, y_val_enc, n_classes, cfg)

    return models

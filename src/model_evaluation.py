"""
Model Evaluation — computes metrics and generates plots.

Metrics per model:
  • Accuracy
  • F1 Weighted
  • F1 Macro
  • ROC-AUC (OVR, macro)
  • Confusion matrix (saved as PNG)
  • Classification report

SHAP explainability for classical models.
Performance comparison table saved to JSON.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from src.logger import get_logger
from src.exception import SERException
from src.config import ProjectConfig

logger = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ensure_plots_dir(cfg: ProjectConfig) -> str:
    plots_dir = cfg.evaluation["plots_dir"]
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _get_predictions(model, X, is_keras_model: bool) -> tuple:
    """Return (y_pred_int, y_proba) for any model type."""
    try:
        if is_keras_model:
            y_proba = model.predict(X, verbose=0)
            y_pred = np.argmax(y_proba, axis=1)
        else:
            y_pred = model.predict(X)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)
            else:
                y_proba = None
    except Exception as e:
        raise SERException(e)
    return y_pred, y_proba


def _is_keras(model) -> bool:
    try:
        import tensorflow as tf
        return isinstance(model, tf.keras.Model)
    except ImportError:
        return False


# ── Per-model evaluation ───────────────────────────────────────────────────────

def evaluate_model(
    name: str,
    model,
    X_test,
    y_test: np.ndarray,
    label_names: list[str],
    cfg: ProjectConfig,
) -> dict:
    """
    Evaluate a single model. Returns metrics dict.
    """
    plots_dir = _ensure_plots_dir(cfg)
    is_keras = _is_keras(model)

    # Prepare input
    if is_keras and "CNN" in name and "LSTM" not in name:
        X_input = X_test[..., np.newaxis]
    elif is_keras and "LSTM" in name and "CNN" not in name:
        X_input = np.transpose(X_test, (0, 2, 1))
    elif is_keras and "CNN-LSTM" in name:
        X_input = X_test[..., np.newaxis]
    else:
        X_input = X_test

    y_pred, y_proba = _get_predictions(model, X_input, is_keras)

    acc = accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # ROC-AUC
    roc_auc = None
    if y_proba is not None:
        try:
            n_classes = len(label_names)
            y_bin = label_binarize(y_test, classes=list(range(n_classes)))
            roc_auc = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
        except Exception as e:
            logger.warning(f"ROC-AUC failed for {name}: {e}")

    logger.info(
        f"{name}: Acc={acc:.4f} | F1-W={f1_w:.4f} | F1-M={f1_m:.4f}"
        + (f" | AUC={roc_auc:.4f}" if roc_auc else "")
    )
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=label_names, zero_division=0)}")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, label_names, name, plots_dir)

    # ROC curves
    if y_proba is not None and roc_auc is not None:
        _plot_roc_curves(y_test, y_proba, label_names, name, plots_dir)

    return {
        "name": name,
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1_w, 4),
        "f1_macro": round(f1_m, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc else None,
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def _plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    model_name: str,
    plots_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14)
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"cm_{model_name.replace(' ', '_').replace('-', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved: {fname}")


def _plot_roc_curves(
    y_test: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
    model_name: str,
    plots_dir: str,
) -> None:
    n_classes = len(labels)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (label, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_i = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc_i:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name}", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fname = os.path.join(plots_dir, f"roc_{model_name.replace(' ', '_').replace('-', '_')}.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    logger.info(f"ROC curve saved: {fname}")


def plot_performance_comparison(results: list[dict], plots_dir: str) -> None:
    """Bar chart comparing all models on key metrics."""
    names = [r["name"] for r in results]
    accs = [r["accuracy"] for r in results]
    f1ws = [r["f1_weighted"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, accs, width, label="Accuracy", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, f1ws, width, label="F1 Weighted", color="#DD8452")

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    fname = os.path.join(plots_dir, "performance_comparison.png")
    plt.savefig(fname, dpi=150)
    plt.close()
    logger.info(f"Performance comparison chart saved: {fname}")


# ── SHAP explainability ────────────────────────────────────────────────────────

def shap_explain(model, X_sample: np.ndarray, label_names: list[str],
                  model_name: str, plots_dir: str) -> None:
    """Generate SHAP summary plot for a classical sklearn model."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping explainability. pip install shap")
        return

    try:
        logger.info(f"Computing SHAP values for {model_name}…")
        # Get the actual estimator from pipeline if needed
        estimator = model
        if hasattr(model, "named_steps"):
            X_transformed = model[:-1].transform(X_sample)
            estimator = model.named_steps["model"]
        else:
            X_transformed = X_sample

        explainer = shap.TreeExplainer(estimator) if hasattr(estimator, "estimators_") \
            else shap.KernelExplainer(estimator.predict_proba, shap.sample(X_transformed, 50))

        shap_values = explainer.shap_values(X_transformed[:100])

        fig = plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_transformed[:100], plot_type="bar",
                          class_names=label_names, show=False)
        plt.title(f"SHAP Feature Importance — {model_name}")
        plt.tight_layout()
        fname = os.path.join(plots_dir, f"shap_{model_name.replace(' ', '_')}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP plot saved: {fname}")
    except Exception as e:
        logger.warning(f"SHAP failed for {model_name}: {e}")


# ── Evaluate all ───────────────────────────────────────────────────────────────

def evaluate_all_models(
    models: dict,
    X_flat_test: np.ndarray,
    X_mfcc_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list[str],
    cfg: ProjectConfig,
) -> list[dict]:
    """
    Evaluate every model and return list of metric dicts.
    Saves comparison chart + JSON summary.
    """
    logger.info("═══ Model Evaluation ═══")
    results = []

    for name, model in models.items():
        is_keras = _is_keras(model)
        X = X_mfcc_test if is_keras else X_flat_test
        metrics = evaluate_model(name, model, X, y_test, label_names, cfg)
        results.append(metrics)

    # Sort by F1 weighted descending
    results.sort(key=lambda r: r["f1_weighted"], reverse=True)

    plots_dir = _ensure_plots_dir(cfg)
    plot_performance_comparison(results, plots_dir)

    # Save JSON
    json_path = os.path.join(cfg.paths["artifacts_dir"], "model_comparison.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # SHAP for RF and XGBoost
    for name in ["Random Forest", "XGBoost"]:
        if name in models:
            shap_explain(models[name], X_flat_test, label_names, name, plots_dir)

    return results

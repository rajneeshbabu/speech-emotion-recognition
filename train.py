"""
CLI entry point for training the SER models.

Usage:
    python train.py
    python train.py --config config/config.yaml
"""

import argparse
from src.config import load_config
from pipelines.training_pipeline import run_training_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition — Training")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config YAML (default: config/config.yaml)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    results = run_training_pipeline(cfg)

    print("\n" + "═" * 60)
    print("  FINAL RESULTS")
    print("═" * 60)
    print(f"{'Model':<22} {'Accuracy':>9} {'F1-W':>8} {'F1-M':>8} {'AUC':>8}")
    print("─" * 60)
    for r in results:
        auc_str = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "  N/A  "
        print(
            f"{r['name']:<22} {r['accuracy']:>9.4f} {r['f1_weighted']:>8.4f} "
            f"{r['f1_macro']:>8.4f} {auc_str:>8}"
        )
    print("═" * 60)

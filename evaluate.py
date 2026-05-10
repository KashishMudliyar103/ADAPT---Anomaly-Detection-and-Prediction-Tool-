"""
evaluate.py
-----------
Offline evaluation script:
    python evaluate.py --data data/cybercrime_forensic_dataset.csv

Computes classification metrics vs. ground-truth Label column,
prints confusion matrix, and saves the trained model.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from utils.preprocessor import preprocess_data
from utils.feature_engineer import engineer_features
from models.detector import AnomalyDetector


def evaluate(data_path: str, save_model: bool = True):
    print(f"\n{'='*60}")
    print("  ADAPT — Offline Evaluation")
    print(f"{'='*60}")

    # ── Load & process ────────────────────────────────────────────────
    print(f"\n[1/4] Loading data: {data_path}")
    df_raw    = pd.read_csv(data_path)
    df_clean  = preprocess_data(df_raw)
    df_feat   = engineer_features(df_clean)
    print(f"      Rows: {len(df_feat):,} | Columns: {len(df_feat.columns)}")

    # ── Fit models ────────────────────────────────────────────────────
    print("\n[2/4] Fitting K-Means + DBSCAN + Isolation Forest …")
    detector  = AnomalyDetector(n_clusters=3, eps=0.5, min_samples=5, contamination=0.10)
    df_result = detector.fit_predict(df_feat)

    # ── Ground-truth comparison ───────────────────────────────────────
    print("\n[3/4] Comparing to ground-truth labels …")
    y_true_raw = df_result["Label"].str.strip().str.title()
    # Map ground truth: Normal → Normal, Suspicious → Suspicious/Risky
    y_pred     = df_result["Risk_Level"]

    # Binary: Normal vs Anomalous
    y_true_bin = (y_true_raw != "Normal").astype(int)
    y_pred_bin = (y_pred != "Normal").astype(int)

    acc   = accuracy_score(y_true_bin, y_pred_bin)
    try:
        auc = roc_auc_score(y_true_bin, df_result["anomaly_score"])
    except Exception:
        auc = float("nan")

    print(f"\n  Binary Detection (Normal vs Anomalous)")
    print(f"  {'─'*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_true_bin, y_pred_bin,
                                  target_names=["Normal","Anomalous"]))

    # Confusion matrix
    cm = confusion_matrix(y_true_bin, y_pred_bin)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred Normal","Pred Anomalous"],
                yticklabels=["True Normal","True Anomalous"])
    plt.title("ADAPT — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150)
    print("  Confusion matrix saved → assets/confusion_matrix.png")

    # Risk level distribution
    print(f"\n  Risk Level Breakdown:")
    print(df_result["Risk_Level"].value_counts().to_string())

    # ── Save model ────────────────────────────────────────────────────
    if save_model:
        print("\n[4/4] Saving model …")
        detector.save("models/adapt_model.pkl")

    print(f"\n{'='*60}")
    print("  Evaluation complete.")
    print(f"{'='*60}\n")
    return df_result, detector


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAPT offline evaluation")
    parser.add_argument("--data",  default="data/cybercrime_forensic_dataset.csv")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    evaluate(args.data, save_model=not args.no_save)

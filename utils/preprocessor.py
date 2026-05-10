"""
preprocessor.py
---------------
Cleans and prepares raw log data for feature engineering.
"""
import pandas as pd
import numpy as np


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw log DataFrame:
    - Parse timestamps
    - Fill nulls intelligently
    - Normalize categorical fields
    Returns a clean copy.
    """
    df = df.copy()

    # ── Timestamps ──────────────────────────────────────────────────
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ── Numeric fills ────────────────────────────────────────────────
    df["Login_Attempts"] = df["Login_Attempts"].fillna(0).clip(lower=0)
    df["File_Size"]      = df["File_Size"].fillna(0).clip(lower=0)

    # ── Categorical fills ────────────────────────────────────────────
    df["File_Name"]    = df["File_Name"].fillna("none")
    df["Action"]       = df["Action"].fillna("Unknown")
    df["Anomaly_Type"] = df["Anomaly_Type"].fillna("None")

    # ── String normalisation ─────────────────────────────────────────
    for col in ["Activity_Type", "Action", "Anomaly_Type", "Resource_Accessed"]:
        df[col] = df[col].str.strip().str.title()

    # ── Label passthrough (ground truth kept for evaluation) ─────────
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.strip().str.title()

    # ── IP validity sanity-flag ──────────────────────────────────────
    df["is_internal_ip"] = df["IP_Address"].apply(_is_internal)

    return df


def _is_internal(ip: str) -> int:
    """Return 1 if IP looks like an RFC-1918 private address."""
    try:
        parts = str(ip).split(".")
        if len(parts) != 4:
            return 0
        first, second = int(parts[0]), int(parts[1])
        if first == 10:
            return 1
        if first == 172 and 16 <= second <= 31:
            return 1
        if first == 192 and second == 168:
            return 1
        return 0
    except Exception:
        return 0

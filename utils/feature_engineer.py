"""
feature_engineer.py
--------------------
Transforms clean log data into ML-ready behavioral features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "login_attempts_norm",
    "file_size_norm",
    "hour_sin",
    "hour_cos",
    "is_failed_action",
    "is_remote",
    "is_file_op",
    "is_network_op",
    "is_usb",
    "is_external_ip",
    "user_event_rate",
    "user_fail_rate",
    "user_unique_ips",
    "anomaly_score",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature-enriched DataFrame ready for clustering.
    Raw columns are preserved; new feature columns are appended.
    """
    df = df.copy()

    # ── Temporal features ─────────────────────────────────────────────
    df["hour"]     = df["Timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow"]      = df["Timestamp"].dt.dayofweek          # 0=Mon
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # ── Scaled numerics ───────────────────────────────────────────────
    scaler = StandardScaler()
    num_cols = ["Login_Attempts", "File_Size"]
    scaled   = scaler.fit_transform(df[num_cols].fillna(0))
    df["login_attempts_norm"] = scaled[:, 0]
    df["file_size_norm"]      = scaled[:, 1]

    # ── Activity flags ────────────────────────────────────────────────
    df["is_failed_action"] = (df["Action"].str.lower() == "failed").astype(int)
    df["is_remote"]        = df["Activity_Type"].str.contains("Remote", case=False, na=False).astype(int)
    df["is_file_op"]       = df["Activity_Type"].str.contains("File", case=False, na=False).astype(int)
    df["is_network_op"]    = df["Activity_Type"].str.contains("Network", case=False, na=False).astype(int)
    df["is_usb"]           = df["Activity_Type"].str.contains("Usb", case=False, na=False).astype(int)
    df["is_external_ip"]   = (1 - df["is_internal_ip"])

    # ── Per-user behavioral aggregates ───────────────────────────────
    user_stats = df.groupby("User_ID").agg(
        user_event_count  = ("User_ID", "count"),
        user_fail_count   = ("is_failed_action", "sum"),
        user_unique_ips   = ("IP_Address", "nunique"),
        user_avg_filesize = ("File_Size", "mean"),
    ).reset_index()
    user_stats["user_event_rate"] = (
        user_stats["user_event_count"] / user_stats["user_event_count"].max()
    )
    user_stats["user_fail_rate"]  = (
        user_stats["user_fail_count"] / user_stats["user_event_count"].clip(lower=1)
    )
    user_stats["user_unique_ips"] = (
        user_stats["user_unique_ips"] / user_stats["user_unique_ips"].max()
    )
    df = df.merge(
        user_stats[["User_ID","user_event_rate","user_fail_rate","user_unique_ips"]],
        on="User_ID", how="left",
    )

    # ── Composite anomaly score (heuristic, pre-model) ────────────────
    df["anomaly_score"] = (
        0.30 * df["login_attempts_norm"].clip(lower=0)
        + 0.20 * df["is_failed_action"]
        + 0.15 * df["is_remote"]
        + 0.15 * df["user_fail_rate"]
        + 0.10 * df["is_external_ip"]
        + 0.10 * df["user_unique_ips"]
    )

    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return the numeric feature matrix for model training."""
    return df[FEATURE_COLS].fillna(0).values

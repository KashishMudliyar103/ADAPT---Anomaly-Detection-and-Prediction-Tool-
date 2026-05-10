"""
detector.py
-----------
Core anomaly detection pipeline combining K-Means + DBSCAN +
Isolation Forest for multi-layer behavioral profiling.

Risk tier assignment:
    Risky      → DBSCAN noise point  OR  high anomaly score  (top contamination %)
    Suspicious → K-Means outlier cluster OR medium score
    Normal     → everything else
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

from utils.feature_engineer import get_feature_matrix, FEATURE_COLS


class AnomalyDetector:
    """
    Unsupervised anomaly detection using K-Means + DBSCAN + Isolation Forest.

    Parameters
    ----------
    n_clusters   : int   – K-Means cluster count
    eps          : float – DBSCAN epsilon
    min_samples  : int   – DBSCAN min_samples
    contamination: float – Expected anomaly fraction (Isolation Forest)
    """

    def __init__(
        self,
        n_clusters:    int   = 3,
        eps:           float = 0.5,
        min_samples:   int   = 5,
        contamination: float = 0.10,
        random_state:  int   = 42,
    ):
        self.n_clusters    = n_clusters
        self.eps           = eps
        self.min_samples   = min_samples
        self.contamination = contamination
        self.random_state  = random_state

        self.scaler  = StandardScaler()
        self.pca     = PCA(n_components=2, random_state=random_state)
        self.kmeans  = KMeans(n_clusters=n_clusters, random_state=random_state,
                               n_init="auto")
        self.dbscan  = DBSCAN(eps=eps, min_samples=min_samples)
        self.iso_forest = IsolationForest(
            contamination=contamination, random_state=random_state, n_estimators=150
        )

        self.feature_importances_ : dict = {}
        self.risky_cluster_        : int  = -1

    # ─────────────────────────────────────────────────────────────────
    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all models on df and return df enriched with:
            kmeans_cluster, dbscan_label, iso_flag,
            pca_x, pca_y, Risk_Level
        """
        X_raw = get_feature_matrix(df)
        X     = self.scaler.fit_transform(X_raw)

        # ── PCA (2-D for visualisation) ───────────────────────────────
        X_2d  = self.pca.fit_transform(X)

        # ── K-Means ───────────────────────────────────────────────────
        km_labels = self.kmeans.fit_predict(X)

        # Identify the "riskiest" cluster: highest mean anomaly_score
        score_idx = FEATURE_COLS.index("anomaly_score")
        cluster_risk = {}
        for c in range(self.n_clusters):
            mask = km_labels == c
            cluster_risk[c] = X_raw[mask, score_idx].mean() if mask.sum() else 0
        self.risky_cluster_ = max(cluster_risk, key=cluster_risk.get)

        # ── DBSCAN ────────────────────────────────────────────────────
        db_labels = self.dbscan.fit_predict(X)

        # ── Isolation Forest ──────────────────────────────────────────
        iso_pred  = self.iso_forest.fit_predict(X)   # -1 = anomaly, 1 = normal
        iso_score = self.iso_forest.score_samples(X) # lower = more anomalous

        # ── Feature importances (mean abs contribution to iso score) ──
        self.feature_importances_ = {
            feat: float(np.abs(X[:, i]).mean())
            for i, feat in enumerate(FEATURE_COLS)
        }

        # ── Risk tier assignment ──────────────────────────────────────
        score_threshold_high = np.percentile(df["anomaly_score"].values,
                                              100 * (1 - self.contamination))
        score_threshold_mid  = np.percentile(df["anomaly_score"].values,
                                              100 * (1 - self.contamination * 2.5))

        risk_levels = []
        for i in range(len(df)):
            s  = df["anomaly_score"].iat[i]
            db = db_labels[i]
            km = km_labels[i]
            iso = iso_pred[i]

            if db == -1 or (iso == -1 and s >= score_threshold_high):
                risk_levels.append("Risky")
            elif km == self.risky_cluster_ or s >= score_threshold_mid:
                risk_levels.append("Suspicious")
            else:
                risk_levels.append("Normal")

        result = df.copy()
        result["kmeans_cluster"] = km_labels
        result["dbscan_label"]   = db_labels
        result["iso_flag"]       = iso_pred
        result["iso_score"]      = iso_score
        result["pca_x"]          = X_2d[:, 0]
        result["pca_y"]          = X_2d[:, 1]
        result["Risk_Level"]     = risk_levels

        return result

    # ─────────────────────────────────────────────────────────────────
    def save(self, path: str = "models/adapt_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/adapt_model.pkl") -> "AnomalyDetector":
        return joblib.load(path)

"""
ADAPT - Anomaly Detection and Prediction Tool
==============================================
Unsupervised ML system for detecting abnormal behavioral patterns
in cybersecurity user log data using K-Means and DBSCAN.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

from utils.preprocessor import preprocess_data
from utils.feature_engineer import engineer_features
from models.detector import AnomalyDetector
from utils.visualizer import (
    plot_risk_distribution,
    plot_time_series,
    plot_cluster_scatter,
    plot_anomaly_heatmap,
    plot_user_risk_profile,
    plot_dbscan_results,
    plot_feature_importance,
    plot_activity_breakdown,
)

# ─────────────────────────── Page Config ────────────────────────────
st.set_page_config(
    page_title="ADAPT | Anomaly Detection & Prediction Tool",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Custom CSS ─────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2130; border-radius: 10px; padding: 10px; }
    .risk-normal   { color: #00d084; font-weight: bold; }
    .risk-suspicious { color: #ffa500; font-weight: bold; }
    .risk-risky    { color: #ff4444; font-weight: bold; }
    .section-header {
        font-size: 1.4rem; font-weight: 700;
        border-left: 4px solid #5c6bc0;
        padding-left: 12px; margin: 20px 0 10px 0;
    }
    div[data-testid="metric-container"] {
        background-color: #1e2130;
        border: 1px solid #2d3250;
        border-radius: 10px;
        padding: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── Sidebar ────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/ADAPT-v1.0-5c6bc0?style=for-the-badge", use_column_width=True)
    st.markdown("## ⚙️ Configuration")

    uploaded = st.file_uploader("📁 Upload Log CSV", type=["csv"])
    st.markdown("---")

    st.markdown("### 🔬 Model Parameters")
    n_clusters = st.slider("K-Means Clusters", 2, 10, 3, help="Number of behavioral clusters")
    eps_value   = st.slider("DBSCAN ε (epsilon)", 0.1, 2.0, 0.5, 0.1,
                            help="Maximum distance between two samples in a neighborhood")
    min_samples = st.slider("DBSCAN min_samples", 2, 20, 5,
                            help="Minimum samples to form a core point")
    contamination = st.slider("Expected Anomaly Rate", 0.01, 0.30, 0.10, 0.01,
                              help="Fraction of data expected to be anomalous")
    st.markdown("---")

    st.markdown("### 🎯 Filters")
    time_window = st.selectbox("Time Window", ["All Time", "Last 7 Days", "Last 30 Days"])
    risk_filter = st.multiselect("Show Risk Levels", ["Normal", "Suspicious", "Risky"],
                                  default=["Normal", "Suspicious", "Risky"])
    st.markdown("---")
    st.markdown("""
    <small>
    **ADAPT** uses K-Means + DBSCAN to profile entity behavior
    and surface anomalies in user log data.<br><br>
    Built with ❤️ using Scikit-Learn · Plotly · Streamlit
    </small>
    """, unsafe_allow_html=True)

# ─────────────────────────── Load Data ──────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_process(file_source, n_clusters, eps, min_s, cont):
    if file_source == "default":
        df_raw = pd.read_csv("data/cybercrime_forensic_dataset.csv")
    else:
        df_raw = pd.read_csv(file_source)
    df_clean   = preprocess_data(df_raw)
    df_feat    = engineer_features(df_clean)
    detector   = AnomalyDetector(n_clusters=n_clusters, eps=eps,
                                  min_samples=min_s, contamination=cont)
    df_result  = detector.fit_predict(df_feat)
    return df_raw, df_clean, df_feat, df_result, detector

file_src = uploaded if uploaded else "default"

with st.spinner("🔍 Running anomaly detection pipeline..."):
    try:
        df_raw, df_clean, df_feat, df_result, detector = load_and_process(
            file_src, n_clusters, eps_value, min_samples, contamination
        )
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# Apply time filter
df_display = df_result.copy()
if time_window == "Last 7 Days":
    cutoff = df_display["Timestamp"].max() - pd.Timedelta(days=7)
    df_display = df_display[df_display["Timestamp"] >= cutoff]
elif time_window == "Last 30 Days":
    cutoff = df_display["Timestamp"].max() - pd.Timedelta(days=30)
    df_display = df_display[df_display["Timestamp"] >= cutoff]

if risk_filter:
    df_display = df_display[df_display["Risk_Level"].isin(risk_filter)]

# ─────────────────────────── Header ─────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:2.5rem; margin:0;'>🛡️ ADAPT</h1>
    <p style='color:#8892b0; font-size:1.1rem; margin:0;'>
        Anomaly Detection and Prediction Tool · Cybersecurity Log Analytics
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────── KPI Row ────────────────────────────────
total      = len(df_display)
risky_n    = (df_display["Risk_Level"] == "Risky").sum()
susp_n     = (df_display["Risk_Level"] == "Suspicious").sum()
normal_n   = (df_display["Risk_Level"] == "Normal").sum()
unique_ips  = df_display["IP_Address"].nunique()
alert_rate  = round(100 * (risky_n + susp_n) / max(total, 1), 1)

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("📋 Total Events",    f"{total:,}")
col2.metric("🔴 Risky",           f"{risky_n:,}",   delta=f"{100*risky_n/max(total,1):.1f}%")
col3.metric("🟡 Suspicious",      f"{susp_n:,}",    delta=f"{100*susp_n/max(total,1):.1f}%")
col4.metric("🟢 Normal",          f"{normal_n:,}")
col5.metric("🌐 Unique IPs",      f"{unique_ips:,}")
col6.metric("🚨 Alert Rate",      f"{alert_rate}%",  delta_color="inverse")

st.markdown("---")

# ─────────────────────────── Tab Layout ─────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🔬 Cluster Analysis",
    "⏱️ Time Series",
    "👤 Entity Profiles",
    "📡 DBSCAN Deep Dive",
    "🗂️ Raw Events",
])

# ══════════════════ TAB 1 : OVERVIEW ════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Threat Landscape Overview</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        fig_donut = plot_risk_distribution(df_display)
        st.plotly_chart(fig_donut, use_container_width=True)
    with c2:
        fig_heat = plot_anomaly_heatmap(df_display)
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-header">Activity Breakdown by Risk Level</div>',
                unsafe_allow_html=True)
    fig_act = plot_activity_breakdown(df_display)
    st.plotly_chart(fig_act, use_container_width=True)

    st.markdown('<div class="section-header">Feature Contribution to Anomaly Score</div>',
                unsafe_allow_html=True)
    fig_feat = plot_feature_importance(df_feat, detector)
    st.plotly_chart(fig_feat, use_container_width=True)

# ══════════════════ TAB 2 : CLUSTER ANALYSIS ════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">K-Means Behavioral Clustering</div>',
                unsafe_allow_html=True)

    st.info(f"""
    **Model:** K-Means with **{n_clusters} clusters** trained on engineered behavioral features.
    Each point represents a user session. Clusters reveal distinct behavioral archetypes —
    abnormal sessions are flagged based on distance from cluster centroids.
    """)

    fig_scatter = plot_cluster_scatter(df_feat, df_result)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="section-header">Cluster Centroid Profiles (Radar)</div>',
                unsafe_allow_html=True)
    cluster_means = df_result.groupby("kmeans_cluster")[
    ["login_attempts_norm","file_size_norm","hour_sin","is_failed_action",
     "is_remote","anomaly_score"]
].mean().reset_index()
    cluster_means["kmeans_cluster"] = cluster_means["kmeans_cluster"].astype(str)
    fig_radar = px.line_polar(
        cluster_means.melt(id_vars="kmeans_cluster"),
        r="value", theta="variable",
        color="kmeans_cluster",
        line_close=True,
        template="plotly_dark",
        title="Cluster Centroid Radar — Behavioral Fingerprints",
    )
    fig_radar.update_traces(fill="toself", opacity=0.5)
    st.plotly_chart(fig_radar, use_container_width=True)

# ══════════════════ TAB 3 : TIME SERIES ═════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Anomaly Trends Over Time</div>',
                unsafe_allow_html=True)

    fig_ts = plot_time_series(df_display)
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="section-header">Hourly Activity Heatmap</div>',
                unsafe_allow_html=True)
    df_hour = df_display.copy()
    df_hour["Hour"]  = df_hour["Timestamp"].dt.hour
    df_hour["Day"]   = df_hour["Timestamp"].dt.day_name()
    heat_data = df_hour.groupby(["Day","Hour"])["Risk_Level"].apply(
        lambda x: (x.isin(["Suspicious","Risky"])).sum()
    ).reset_index(name="Alerts")
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    heat_data["Day"] = pd.Categorical(heat_data["Day"], categories=day_order, ordered=True)
    heat_pivot = heat_data.pivot(index="Day", columns="Hour", values="Alerts").fillna(0)
    fig_h = px.imshow(
        heat_pivot,
        color_continuous_scale="Reds",
        title="Alert Activity — Day × Hour Heatmap",
        labels=dict(x="Hour of Day", y="Day of Week", color="# Alerts"),
        template="plotly_dark",
    )
    st.plotly_chart(fig_h, use_container_width=True)

# ══════════════════ TAB 4 : ENTITY PROFILES ═════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">User Risk Profiler</div>',
                unsafe_allow_html=True)

    top_risky = (
        df_display[df_display["Risk_Level"] == "Risky"]["User_ID"]
        .value_counts().head(20).index.tolist()
    )
    all_users = sorted(df_display["User_ID"].unique().tolist())
    default_u = top_risky[:1] if top_risky else all_users[:1]

    selected_users = st.multiselect(
        "Select User IDs to profile (top risky users pre-selected)",
        options=all_users,
        default=default_u,
        format_func=lambda x: f"User {x}",
    )

    if selected_users:
        fig_profile = plot_user_risk_profile(df_display, selected_users)
        st.plotly_chart(fig_profile, use_container_width=True)

        st.markdown('<div class="section-header">User Activity Table</div>',
                    unsafe_allow_html=True)
        user_df = df_display[df_display["User_ID"].isin(selected_users)].sort_values(
            "Timestamp", ascending=False
        )
        st.dataframe(
            user_df[["Timestamp","User_ID","IP_Address","Activity_Type","Action",
                      "Risk_Level","Anomaly_Type","anomaly_score"]].head(200),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Select at least one user to profile.")

# ══════════════════ TAB 5 : DBSCAN ══════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">DBSCAN Density-Based Outlier Detection</div>',
                unsafe_allow_html=True)

    st.info(f"""
    **DBSCAN** (ε={eps_value}, min_samples={min_samples}) detects outliers as **noise points**
    (label = -1) that don't belong to any dense cluster. Unlike K-Means, DBSCAN
    finds clusters of arbitrary shape and explicitly marks outliers — ideal for
    rare, high-severity attack patterns.
    """)

    fig_db = plot_dbscan_results(df_feat, df_result)
    st.plotly_chart(fig_db, use_container_width=True)

    n_clusters_found = df_result["dbscan_label"].nunique() - (1 if -1 in df_result["dbscan_label"].values else 0)
    n_outliers       = (df_result["dbscan_label"] == -1).sum()

    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("DBSCAN Clusters Found", n_clusters_found)
    cc2.metric("Noise Points (Outliers)", f"{n_outliers:,}")
    cc3.metric("Outlier Rate", f"{100*n_outliers/len(df_result):.1f}%")

    st.markdown('<div class="section-header">DBSCAN Outlier Events</div>',
                unsafe_allow_html=True)
    outlier_events = df_display[df_display["User_ID"].isin(
        df_result[df_result["dbscan_label"] == -1]["User_ID"].unique()
    )].sort_values("anomaly_score", ascending=False)
    st.dataframe(
        outlier_events[["Timestamp","User_ID","IP_Address","Activity_Type",
                         "Action","Risk_Level","Anomaly_Type","anomaly_score"]].head(100),
        use_container_width=True, hide_index=True
    )

# ══════════════════ TAB 6 : RAW EVENTS ══════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">Raw Event Log</div>', unsafe_allow_html=True)

    col_s, col_r, col_a = st.columns(3)
    search_ip  = col_s.text_input("🔍 Filter by IP Address", "")
    risk_sel   = col_r.selectbox("Risk Level", ["All","Normal","Suspicious","Risky"])
    act_sel    = col_a.selectbox("Activity Type", ["All"] + sorted(df_display["Activity_Type"].unique().tolist()))

    filtered = df_display.copy()
    if search_ip:
        filtered = filtered[filtered["IP_Address"].str.contains(search_ip, na=False)]
    if risk_sel != "All":
        filtered = filtered[filtered["Risk_Level"] == risk_sel]
    if act_sel != "All":
        filtered = filtered[filtered["Activity_Type"] == act_sel]

    st.caption(f"Showing {len(filtered):,} events")
    st.dataframe(
        filtered.sort_values("Timestamp", ascending=False).head(1000),
        use_container_width=True, hide_index=True
    )

    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button("⬇️ Export Filtered Events (CSV)", csv_bytes,
                        "adapt_filtered_events.csv", "text/csv")

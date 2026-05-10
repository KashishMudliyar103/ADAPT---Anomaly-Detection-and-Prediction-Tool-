"""
visualizer.py
-------------
All Plotly chart builders for ADAPT dashboard.
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RISK_COLORS = {
    "Normal":     "#00d084",
    "Suspicious": "#ffa500",
    "Risky":      "#ff4444",
}
TEMPLATE = "plotly_dark"


# ─────────────────────────────────────────────────────────────────────
def plot_risk_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["Risk_Level"].value_counts().reset_index()
    counts.columns = ["Risk_Level", "Count"]
    colors = [RISK_COLORS.get(r, "#888") for r in counts["Risk_Level"]]
    fig = go.Figure(go.Pie(
        labels=counts["Risk_Level"],
        values=counts["Count"],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0e1117", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} events<extra></extra>",
    ))
    fig.update_layout(
        title="Risk Level Distribution",
        template=TEMPLATE,
        showlegend=True,
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text="RISK", x=0.5, y=0.5, font_size=16,
                          showarrow=False, font_color="#8892b0")]
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
def plot_anomaly_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["Activity_Type", "Risk_Level"])
        .size()
        .reset_index(name="Count")
        .pivot(index="Activity_Type", columns="Risk_Level", values="Count")
        .fillna(0)
    )
    fig = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        template=TEMPLATE,
        title="Activity Type × Risk Level Heatmap",
        labels=dict(color="# Events"),
        text_auto=True,
    )
    fig.update_layout(
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
def plot_time_series(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["Date"] = df2["Timestamp"].dt.date
    ts = (
        df2.groupby(["Date", "Risk_Level"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.line(
        ts,
        x="Date", y="Count", color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        template=TEMPLATE,
        title="Daily Event Counts by Risk Level",
        markers=True,
        labels={"Count": "# Events", "Date": "Date"},
    )
    fig.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(title="Risk Level"),
    )
    # Highlight the date with the most Risky events
    risky_ts = ts[ts["Risk_Level"] == "Risky"]
    if not risky_ts.empty:
        peak_date = risky_ts.loc[risky_ts["Count"].idxmax(), "Date"]
        fig.add_vrect(
            x0=str(peak_date), x1=str(ts["Date"].max()),
            line_width=0, fillcolor="red", opacity=0.04,
            annotation_text="High-Risk Period", annotation_position="top left",
        )
    return fig


# ─────────────────────────────────────────────────────────────────────
def plot_cluster_scatter(df_feat: pd.DataFrame, df_result: pd.DataFrame) -> go.Figure:
    merged = df_result.copy()
    merged["kmeans_cluster"] = merged["kmeans_cluster"].astype(str)
    fig = px.scatter(
        merged,
        x="pca_x", y="pca_y",
        color="Risk_Level",
        symbol="kmeans_cluster",
        color_discrete_map=RISK_COLORS,
        hover_data=["User_ID", "Activity_Type", "anomaly_score"],
        template=TEMPLATE,
        title="K-Means Clusters in PCA-2D Space",
        labels={"pca_x": "PC-1 (Behavioral Axis)", "pca_y": "PC-2"},
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=5, line=dict(width=0)))
    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
def plot_dbscan_results(df_feat: pd.DataFrame, df_result: pd.DataFrame) -> go.Figure:
    df2 = df_result.copy()
    df2["DBSCAN_Label"] = df2["dbscan_label"].apply(
        lambda x: "Outlier (Noise)" if x == -1 else f"Cluster {x}"
    )
    color_map = {"Outlier (Noise)": "#ff4444"}
    unique_clusters = [c for c in df2["DBSCAN_Label"].unique() if c != "Outlier (Noise)"]
    palette = px.colors.qualitative.Pastel
    for i, c in enumerate(sorted(unique_clusters)):
        color_map[c] = palette[i % len(palette)]

    fig = px.scatter(
        df2,
        x="pca_x", y="pca_y",
        color="DBSCAN_Label",
        color_discrete_map=color_map,
        hover_data=["User_ID", "Activity_Type", "anomaly_score", "Risk_Level"],
        template=TEMPLATE,
        title="DBSCAN Clustering — Noise Points = Potential Intrusions",
        labels={"pca_x": "PC-1", "pca_y": "PC-2"},
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
def plot_user_risk_profile(df: pd.DataFrame, user_ids: list) -> go.Figure:
    sub = df[df["User_ID"].isin(user_ids)].copy()
    sub["Date"] = sub["Timestamp"].dt.date

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Risk Level Timeline",
            "Activity Type Distribution",
            "Anomaly Score Distribution",
            "Hourly Activity Pattern",
        ],
    )

    # Timeline
    ts = sub.groupby(["Date","Risk_Level"]).size().reset_index(name="Count")
    for rl, color in RISK_COLORS.items():
        sl = ts[ts["Risk_Level"] == rl]
        fig.add_trace(
            go.Scatter(x=sl["Date"], y=sl["Count"], mode="lines+markers",
                       name=rl, line=dict(color=color), showlegend=True),
            row=1, col=1
        )

    # Activity distribution
    act_cnt = sub["Activity_Type"].value_counts().reset_index()
    act_cnt.columns = ["Activity_Type", "Count"]
    fig.add_trace(
        go.Bar(x=act_cnt["Count"], y=act_cnt["Activity_Type"],
               orientation="h", marker_color="#5c6bc0", showlegend=False),
        row=1, col=2
    )

    # Anomaly score histogram
    for uid in user_ids:
        u_sub = sub[sub["User_ID"] == uid]
        fig.add_trace(
            go.Histogram(x=u_sub["anomaly_score"], name=f"U{uid}",
                         opacity=0.7, nbinsx=20),
            row=2, col=1
        )

    # Hourly pattern
    sub["Hour"] = sub["Timestamp"].dt.hour
    hourly = sub.groupby("Hour").size().reset_index(name="Count")
    fig.add_trace(
        go.Bar(x=hourly["Hour"], y=hourly["Count"],
               marker_color="#00d084", showlegend=False),
        row=2, col=2
    )

    fig.update_layout(
        height=600,
        template=TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title_text=f"User Profile: {', '.join(map(str, user_ids))}",
        barmode="overlay",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
def plot_feature_importance(df_feat: pd.DataFrame, detector) -> go.Figure:
    if not detector.feature_importances_:
        return go.Figure()
    fi = pd.Series(detector.feature_importances_).sort_values(ascending=True)
    colors = px.colors.sequential.Plasma_r
    fig = go.Figure(go.Bar(
        x=fi.values,
        y=fi.index,
        orientation="h",
        marker=dict(
            color=fi.values,
            colorscale="Plasma",
            showscale=True,
            colorbar=dict(title="Importance"),
        ),
    ))
    fig.update_layout(
        title="Feature Importance (Isolation Forest Mean Abs Contribution)",
        template=TEMPLATE,
        height=400,
        xaxis_title="Mean |Feature Value|",
        yaxis_title="Feature",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
def plot_activity_breakdown(df: pd.DataFrame) -> go.Figure:
    breakdown = (
        df.groupby(["Activity_Type", "Risk_Level"])
        .size()
        .reset_index(name="Count")
    )
    fig = px.bar(
        breakdown,
        x="Activity_Type", y="Count", color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        barmode="stack",
        template=TEMPLATE,
        title="Event Volume by Activity Type and Risk Level",
        labels={"Count": "# Events", "Activity_Type": "Activity"},
    )
    fig.update_layout(
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_tickangle=-30,
    )
    return fig

"""Plot factory functions (Plotly) for the dashboard."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def histogram_plot(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.histogram(df, x="critical_temp", nbins=60, color_discrete_sequence=["#4ECDC4"])
    fig.update_layout(title=title, template="plotly_dark", bargap=0.05, height=400)
    fig.update_xaxes(title="Critical temperature (K)")
    fig.update_yaxes(title="Count")
    return fig


def box_violin(df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(x=df["critical_temp"], name="Box", boxmean=True))
    fig.add_trace(go.Violin(x=df["critical_temp"], name="Violin", points=False, box_visible=True, meanline_visible=True))
    fig.update_layout(title=title, template="plotly_dark", height=400)
    fig.update_xaxes(title="Critical temperature (K)")
    return fig


def scatter_affinity(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df,
        x="mean_ElectronAffinity",
        y="critical_temp",
        trendline="ols",
        color_discrete_sequence=["#A23B72"],
    )
    fig.update_layout(
        title="YBCO: Critical temperature vs. mean electron affinity",
        template="plotly_dark",
        height=450,
    )
    fig.update_xaxes(title="Mean electron affinity (kJ/mol)")
    fig.update_yaxes(title="Critical temperature (K)")
    return fig


def elements_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.violin(
        df,
        x="Number of elements",
        y="critical_temp",
        color="Number of elements",
        box=True,
        points=False,
        color_discrete_sequence=px.colors.sequential.Teal,
    )
    fig.update_layout(
        title="Critical temperature by number of elements",
        template="plotly_dark",
        height=500,
        showlegend=False,
    )
    fig.update_xaxes(title="Number of elements")
    fig.update_yaxes(title="Critical temperature (K)")
    return fig


def density_plot(df: pd.DataFrame) -> go.Figure:
    fig = px.density_contour(
        df,
        x="critical_temp",
        y="Number of elements",
        nbinsx=40,
        nbinsy=12,
    )
    fig.update_traces(contours_coloring="fill", colorscale="Agsunset", showscale=True)
    fig.update_layout(template="plotly_dark", height=450, title="Density: temperature vs. element count")
    fig.update_xaxes(title="Critical temperature (K)")
    fig.update_yaxes(title="Number of elements")
    return fig


def properties_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap of material properties vs critical temperature."""
    properties = [
        "critical_temp",
        "mean_atomic_mass",
        "mean_Density",
        "mean_FusionHeat",
        "mean_ThermalConductivity",
        "mean_ElectronAffinity",
        "mean_Valence",
    ]
    available_props = [p for p in properties if p in df.columns]
    corr_matrix = df[available_props].corr()
    
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix.values,
            texttemplate="%{z:.3f}",
            textfont={"size": 10},
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=500,
        title="Pearson Correlation Matrix: Material Properties",
    )
    return fig


def correlations_with_temp(df: pd.DataFrame) -> go.Figure:
    """Bar chart of property correlations with critical temperature."""
    properties = {
        "Atomic Mass": "mean_atomic_mass",
        "Density": "mean_Density",
        "Fusion Heat": "mean_FusionHeat",
        "Thermal Conductivity": "mean_ThermalConductivity",
        "Electron Affinity": "mean_ElectronAffinity",
        "Valence": "mean_Valence",
    }
    
    correlations = []
    labels = []
    for label, col in properties.items():
        if col in df.columns:
            corr = df[col].corr(df["critical_temp"])
            correlations.append(corr)
            labels.append(label)
    
    fig = go.Figure(
        data=[go.Bar(
            x=labels,
            y=correlations,
            marker=dict(
                color=correlations,
                colorscale="RdBu",
                cmid=0,
                showscale=True,
            ),
            text=[f"{c:.3f}" for c in correlations],
            textposition="auto",
        )]
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="Pearson Correlation with Critical Temperature",
        xaxis_title="Material Property",
        yaxis_title="Correlation Coefficient",
        yaxis=dict(range=[-1, 1]),
    )
    return fig


def plot_elbow_curve(metrics: dict) -> go.Figure:
    """Plot elbow curve for K-means clustering."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=metrics["k_range"],
        y=metrics["inertias"],
        mode="lines+markers",
        name="Inertia",
        line=dict(color="#4ECDC4", width=3),
        marker=dict(size=10),
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        title="Elbow Method: Optimal Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        showlegend=False,
    )
    return fig


def plot_silhouette_scores(metrics: dict) -> go.Figure:
    """Plot silhouette scores for different k values."""
    fig = go.Figure()
    
    # Highlight optimal k
    colors = ["#4ECDC4" if k == metrics["optimal_k"] else "#B0BEC5" for k in metrics["k_range"]]
    
    fig.add_trace(go.Bar(
        x=metrics["k_range"],
        y=metrics["silhouette_scores"],
        marker=dict(color=colors),
        text=[f"{s:.3f}" for s in metrics["silhouette_scores"]],
        textposition="outside",
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=350,
        title=f"Silhouette Score Analysis (Optimal k={metrics['optimal_k']})",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Silhouette Score",
        showlegend=False,
    )
    return fig


def plot_clusters_2d(X_2d: pd.DataFrame, labels: pd.Series, df_subset: pd.DataFrame) -> go.Figure:
    """Plot clusters in 2D space using PCA."""
    plot_df = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Cluster": labels,
        "Critical_Temp": df_subset["critical_temp"].values,
    })
    
    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data={"Critical_Temp": ":.1f"},
        color_continuous_scale="Turbo",
        template="plotly_dark",
        title="Superconductor Clusters (PCA 2D Projection)",
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=500)
    return fig


def plot_cluster_characteristics(df: pd.DataFrame, labels: np.ndarray) -> go.Figure:
    """Plot average critical temperature per cluster."""
    cluster_df = df.copy()
    cluster_df["Cluster"] = labels
    
    cluster_stats = cluster_df.groupby("Cluster")["critical_temp"].agg(["mean", "median", "count"]).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=cluster_stats["Cluster"].astype(str),
        y=cluster_stats["mean"],
        name="Mean Tc",
        marker=dict(color="#4ECDC4"),
        text=[f"{m:.1f} K" for m in cluster_stats["mean"]],
        textposition="outside",
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=400,
        title="Average Critical Temperature by Cluster",
        xaxis_title="Cluster",
        yaxis_title="Mean Critical Temperature (K)",
        showlegend=False,
    )
    
    return fig

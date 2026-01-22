"""Streamlit dashboard for superconducting materials analysis (frontend only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from superconductor_backend import (
    CRITICAL_TEMP_THRESHOLD,
    FORMULA_FILE,
    TRAIN_FILE,
    box_violin,
    correlations_with_temp,
    density_plot,
    dunn_matrix,
    elements_distribution,
    find_optimal_clusters,
    histogram_plot,
    kendall_corr,
    kruskal_pvalue,
    load_raw_data,
    mann_whitney,
    mann_whitney_detailed,
    normality_pvalue,
    pearson_corr,
    perform_clustering,
    perform_kmeans_clustering,
    plot_cluster_characteristics,
    plot_clusters_2d,
    plot_elbow_curve,
    plot_silhouette_scores,
    preprocess_data,
    prepare_clustering_data,
    properties_heatmap,
    reduce_to_2d,
    scatter_affinity,
    select_families,
    spearman_corr,
    split_temperature_buckets,
)


def _apply_style() -> None:
    """Use the Finance app styling if available; otherwise set a minimal theme."""
    try:
        from helpers.ui_style import apply_style as base_style

        base_style()
    except Exception:
        st.set_page_config(
            page_title="Superconductor Materials Analysis",
            page_icon="🧊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(
            """
            <style>
            :root {
                --surface: rgba(78, 205, 196, 0.1);
                --border: rgba(78, 205, 196, 0.3);
                --text: #E8F0FE;
                --muted: #B0BEC5;
                --accent: #4ECDC4;
                --negative: #FF6B6B;
                --positive: #06D6A0;
            }
            
            /* Main app background */
            .stApp { 
                background: linear-gradient(135deg, #0b1021 0%, #1a1f3a 100%);
                color: var(--text); 
            }
            
            /* Top header/toolbar */
            header[data-testid="stHeader"] {
                background-color: rgba(11, 16, 33, 0.95);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid var(--border);
            }
            
            /* Sidebar styling */
            section[data-testid="stSidebar"] {
                background-color: rgba(11, 16, 33, 0.95);
                border-right: 1px solid var(--border);
            }
            
            section[data-testid="stSidebar"] > div {
                background-color: transparent;
            }
            
            /* Sidebar header text */
            section[data-testid="stSidebar"] h2 {
                color: var(--accent);
            }
            
            section[data-testid="stSidebar"] label {
                color: var(--text);
            }
            .metric-card { 
                background: var(--surface); 
                border: 1px solid var(--border); 
                border-radius: 12px; 
                padding: 16px 18px;
                transition: all 0.3s ease;
                cursor: pointer;
            }
            .metric-card:hover {
                background: rgba(78, 205, 196, 0.15);
                border-color: var(--accent);
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(78, 205, 196, 0.2);
            }
            .metric-value { 
                font-size: 1.6rem; 
                font-weight: 700; 
                color: var(--accent); 
            }
            .metric-label { 
                font-size: 0.85rem; 
                color: var(--muted); 
                text-transform: uppercase; 
                letter-spacing: 0.06em;
                margin-top: 4px;
            }
            .section-box { 
                background: var(--surface); 
                border: 1px solid var(--border); 
                border-radius: 14px; 
                padding: 20px 22px;
                transition: all 0.3s ease;
            }
            .section-box:hover {
                border-color: var(--accent);
                background: rgba(78, 205, 196, 0.12);
            }
            .stSlider > div > div > div > div {
                background: var(--accent);
            }
            input[type="range"]:hover {
                filter: brightness(1.2);
            }
            .stDownloadButton > button {
                background-color: var(--accent);
                color: #0b1021;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            .stDownloadButton > button:hover {
                background-color: #5fe0d6;
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(78, 205, 196, 0.4);
            }
            .dataframe {
                font-size: 0.9rem;
            }
            .dataframe th {
                background-color: var(--surface) !important;
                color: var(--accent) !important;
                font-weight: 600 !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-size: 0.8rem !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def metric_card(value: str, label: str, subtext: str = "", color: str = "") -> None:
    bg = f"background:{color};" if color else ""
    st.markdown(
        f"<div class='metric-card' style='{bg}'><div class='metric-value'>{value}</div><div class='metric-label'>{label}</div><div style='color:#B0BEC5;font-size:0.85rem;margin-top:6px;'>{subtext}</div></div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    _apply_style()
    st.title("🧊 Superconductor Materials Analytics")
    st.caption("Interactive EDA and hypothesis testing for critical temperature drivers.")

    with st.sidebar:
        st.header("⚙️ Configuration")
        threshold = st.slider(
            "HTS threshold (K)",
            min_value=1.0,
            max_value=300.0,
            value=CRITICAL_TEMP_THRESHOLD,
            step=0.1,
            help="Temperature threshold separating Low Tc (LTS) from High Tc (HTS) superconductors"
        )

    train_df, formula_df = load_raw_data()
    data = preprocess_data(train_df, formula_df)

    lts, hts = split_temperature_buckets(data, threshold)
    ybco, iron = select_families(data, threshold)
    mbt = data[data["Number of elements"] < 4]

    tabs = st.tabs([
        "Overview",
        "LTS vs HTS",
        "YBCO vs Fe-based",
        "Electron Affinity",
        "Element Count",
        "Material Properties",
        "Clustering Analysis",
        "Data",
    ])

    with tabs[0]:
        st.subheader("Quick stats")
        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card(f"{len(data):,}", "Samples")
        with col2:
            metric_card(f"{data['critical_temp'].mean():.1f} K", "Mean Tc")
        with col3:
            metric_card(f"{data['Number of elements'].median():.0f}", "Median elements")

        st.plotly_chart(histogram_plot(data, "Critical temperature distribution"), use_container_width=True)

    with tabs[1]:
        st.subheader("Low vs High temperature superconductors")
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(box_violin(lts, "LTS"), use_container_width=True)
            metric_card(f"n = {len(lts)}", "Samples (LTS)")
            metric_card(f"μ = {lts['critical_temp'].mean():.1f} K", "Mean Tc (LTS)")
            metric_card(f"p = {normality_pvalue(lts['critical_temp']):.2e}", "Normality (LTS)")
        with col_b:
            st.plotly_chart(box_violin(hts, "HTS"), use_container_width=True)
            metric_card(f"n = {len(hts)}", "Samples (HTS)")
            metric_card(f"μ = {hts['critical_temp'].mean():.1f} K", "Mean Tc (HTS)")
            metric_card(f"p = {normality_pvalue(hts['critical_temp']):.2e}", "Normality (HTS)")
        # Test if HTS > LTS (hts in first position for alternative='greater')
        p_mw, u_stat, med_hts, med_lts = mann_whitney_detailed(hts["critical_temp"], lts["critical_temp"], alternative="greater")
        st.markdown(f"Mann-Whitney U-test (HTS > LTS):")
        st.markdown(f"  - Median HTS: **{med_hts:.1f} K**, Median LTS: **{med_lts:.1f} K**")
        st.markdown(f"  - U-statistic: **{u_stat:,.0f}**, p-value: **{p_mw:.2e}**")

    with tabs[2]:
        st.subheader("Cuprates vs Iron-based")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(box_violin(ybco, "YBCO"), use_container_width=True)
            metric_card(f"n = {len(ybco)}", "Samples (YBCO)")
            metric_card(f"μ = {ybco['critical_temp'].mean():.1f} K", "Mean Tc (YBCO)")
        with col2:
            st.plotly_chart(box_violin(iron, "Fe-based"), use_container_width=True)
            metric_card(f"n = {len(iron)}", "Samples (Fe-based)")
            metric_card(f"μ = {iron['critical_temp'].mean():.1f} K", "Mean Tc (Fe-based)")
        p_ybco = normality_pvalue(ybco["critical_temp"])
        p_iron = normality_pvalue(iron["critical_temp"])
        st.markdown(f"Normality p-values → YBCO: **{p_ybco:.2e}**, Fe-based: **{p_iron:.2e}**")
        # Test if YBCO > Fe-based (ybco in first position for alternative='greater')
        p_mw_families, u_stat_fam, med_ybco, med_iron = mann_whitney_detailed(ybco["critical_temp"], iron["critical_temp"], alternative="greater")
        st.markdown(f"Mann-Whitney U-test (YBCO > Fe-based):")
        st.markdown(f"  - Median YBCO: **{med_ybco:.1f} K**, Median Fe-based: **{med_iron:.1f} K**")
        st.markdown(f"  - U-statistic: **{u_stat_fam:,.0f}**, p-value: **{p_mw_families:.2e}**")

    with tabs[3]:
        st.subheader("Electron affinity relationship (YBCO)")
        
        # Add selector for property to plot
        property_options = {
            "Electron Affinity": "mean_ElectronAffinity",
            "Atomic Mass": "mean_atomic_mass",
            "Density": "mean_Density",
            "Thermal Conductivity": "mean_ThermalConductivity",
            "Valence": "mean_Valence",
            "Fusion Heat": "mean_FusionHeat",
        }
        selected_prop_name = st.selectbox(
            "Select property to correlate with critical temperature:",
            list(property_options.keys()),
            index=0
        )
        selected_prop = property_options[selected_prop_name]
        
        if selected_prop in ybco.columns:
            fig_scatter = px.scatter(
                ybco,
                x=selected_prop,
                y="critical_temp",
                trendline="ols",
                trendline_color_override="red",
                template="plotly_dark",
                labels={
                    selected_prop: selected_prop_name,
                    "critical_temp": "Critical Temperature (K)",
                },
                title=f"YBCO: {selected_prop_name} vs Critical Temperature",
            )
            fig_scatter.update_layout(height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Calculate and display correlations
            r_p, p_p = pearson_corr(ybco[selected_prop], ybco["critical_temp"])
            r_s, p_s = spearman_corr(ybco[selected_prop], ybco["critical_temp"])
            r_k, p_k = kendall_corr(ybco[selected_prop], ybco["critical_temp"])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                metric_card(f"r = {r_p:.3f}", "Pearson", f"p = {p_p:.2e}")
            with col2:
                metric_card(f"r = {r_s:.3f}", "Spearman", f"p = {p_s:.2e}")
            with col3:
                metric_card(f"τ = {r_k:.3f}", "Kendall", f"p = {p_k:.2e}")

    with tabs[4]:
        st.subheader("Effect of number of elements")
        st.plotly_chart(elements_distribution(data), use_container_width=True)
        st.plotly_chart(density_plot(data), use_container_width=True)
        p_kw = kruskal_pvalue(mbt)
        st.markdown(f"Kruskal-Wallis p-value (1 vs 2 vs 3 elements): **{p_kw:.2e}**")
        with st.expander("Dunn post-hoc p-values"):
            st.dataframe(dunn_matrix(mbt))

    with tabs[5]:
        st.subheader("Material Properties & Correlations")
        st.plotly_chart(properties_heatmap(data), use_container_width=True)
        st.plotly_chart(correlations_with_temp(data), use_container_width=True)

    with tabs[6]:
        st.subheader("🔬 Clustering Analysis")
        st.markdown("Identify natural groupings of superconductors using machine learning")
        
        # Data subset selection
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            data_subset = st.selectbox(
                "Data Subset",
                options=["All Materials", "LTS Only", "HTS Only"],
                help=f"Choose which materials to cluster (LTS threshold: {threshold} K)"
            )
        
        with col2:
            algorithm = st.selectbox(
                "Clustering Algorithm",
                options=["K-means", "Hierarchical", "Gaussian Mixture", "DBSCAN"],
                help="""
                **K-means**: Fast, works well for spherical clusters\n
                **Hierarchical**: Tree-based, good for nested clusters\n
                **Gaussian Mixture**: Probabilistic, allows soft clustering\n
                **DBSCAN**: Density-based, finds arbitrary shapes and handles noise
                """
            )
        
        with col3:
            if algorithm == "DBSCAN":
                eps = st.slider("DBSCAN epsilon", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("Min samples", 3, 20, 5, 1)
        
        # Filter data based on selection
        if data_subset == "LTS Only":
            cluster_data = lts.copy()
            subset_label = f"LTS (Tc ≤ {threshold} K)"
        elif data_subset == "HTS Only":
            cluster_data = hts.copy()
            subset_label = f"HTS (Tc > {threshold} K)"
        else:
            cluster_data = data.copy()
            subset_label = "All Materials"
        
        # Feature selection UI
        available_features = {
            "Critical Temperature": "critical_temp",
            "Atomic Mass": "mean_atomic_mass",
            "Density": "mean_Density",
            "Fusion Heat": "mean_FusionHeat",
            "Thermal Conductivity": "mean_ThermalConductivity",
            "Electron Affinity": "mean_ElectronAffinity",
            "Valence": "mean_Valence",
            "Number of Elements": "Number of elements",
        }
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_features = st.multiselect(
                "Select features for clustering:",
                options=list(available_features.keys()),
                default=list(available_features.keys()),
                help="Choose which material properties to use for identifying clusters"
            )
        with col2:
            sample_size_input = st.number_input(
                "Sample size",
                min_value=1000,
                max_value=min(10000, len(cluster_data)),
                value=min(5000, len(cluster_data)),
                step=500,
                help="Number of materials to analyze (larger = slower but more accurate)"
            )
        
        if len(selected_features) < 2:
            st.warning("⚠️ Please select at least 2 features for clustering")
            return
        
        # Map selected feature names to column names
        selected_cols = [available_features[f] for f in selected_features]
        
        # Prepare data for clustering with selected features
        with st.spinner("Preparing features for clustering..."):
            feature_data = cluster_data[selected_cols].dropna()
            
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(feature_data)
            
            # Sample data
            sample_size = min(sample_size_input, len(feature_data))
            sample_indices = feature_data.sample(n=sample_size, random_state=42).index
            X_sample = X_scaled[feature_data.index.isin(sample_indices)]
            data_sample = cluster_data.loc[sample_indices]
        
        algorithm_map = {
            "K-means": "kmeans",
            "Hierarchical": "hierarchical",
            "Gaussian Mixture": "gmm",
            "DBSCAN": "dbscan"
        }
        algo_key = algorithm_map[algorithm]
        
        st.info(f"📊 **Subset:** {subset_label} ({len(cluster_data):,} total) | **Algorithm:** {algorithm} | **Features:** {len(selected_features)} | **Samples:** {sample_size:,}")
        
        # Show optimization metrics for algorithms that need n_clusters
        if algorithm != "DBSCAN":
            # Find optimal clusters
            with st.spinner("Analyzing optimal number of clusters..."):
                metrics = find_optimal_clusters(X_sample, max_k=8)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_elbow_curve(metrics), use_container_width=True)
            with col2:
                st.plotly_chart(plot_silhouette_scores(metrics), use_container_width=True)
            
            # User selects number of clusters
            n_clusters = st.slider(
                "Select number of clusters (k)",
                min_value=2,
                max_value=8,
                value=metrics["optimal_k"],
                help=f"Recommended: {metrics['optimal_k']} clusters (based on silhouette score)"
            )
        else:
            n_clusters = None
        
        # Perform clustering
        with st.spinner(f"Performing {algorithm} clustering..."):
            if algorithm == "DBSCAN":
                labels = perform_clustering(X_sample, algo_key, eps=eps, min_samples=min_samples)
                n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                st.info(f"🔍 DBSCAN found **{n_clusters_found} clusters** and **{sum(labels == -1)} noise points**")
            else:
                labels = perform_clustering(X_sample, algo_key, n_clusters=n_clusters)
            
            X_2d = reduce_to_2d(X_sample)
        
        # Display 2D cluster visualization
        st.plotly_chart(plot_clusters_2d(X_2d, labels, data_sample), use_container_width=True)
        
        # Display cluster characteristics below
        st.plotly_chart(plot_cluster_characteristics(data_sample, labels), use_container_width=True)
        
        # Cluster statistics table
        st.subheader("Cluster Statistics")
        cluster_df = data_sample.copy()
        cluster_df["Cluster"] = labels
        
        # Filter out noise points for DBSCAN
        if algorithm == "DBSCAN":
            display_df = cluster_df[cluster_df["Cluster"] != -1]
            if sum(labels == -1) > 0:
                st.caption(f"Note: {sum(labels == -1)} noise points excluded from statistics")
        else:
            display_df = cluster_df
        
        agg_dict = {"critical_temp": ["mean", "median", "std", "count"]}
        for col in selected_cols:
            if col != "critical_temp":
                agg_dict[col] = "mean"
        
        cluster_summary = display_df.groupby("Cluster").agg(agg_dict).round(2)
        
        cluster_summary.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in cluster_summary.columns]
        st.dataframe(cluster_summary, use_container_width=True)

    with tabs[7]:
        st.subheader("Data preview")
        
        # Create a cleaner display dataframe with renamed columns
        display_df = data.head(200).copy()
        display_df.columns = [
            col.replace("mean_", "").replace("_", " ").title() 
            for col in display_df.columns
        ]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**Showing first 200 of {len(data):,} samples**")
        with col2:
            st.download_button(
                label="📥 Download CSV",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name="superconductors_processed.csv",
                mime="text/csv",
                type="primary"
            )
        
        st.dataframe(display_df, use_container_width=True, height=600)


if __name__ == "__main__":
    main()

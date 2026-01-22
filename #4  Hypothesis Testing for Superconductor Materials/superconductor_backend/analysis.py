"""Statistical routines for superconducting materials analysis."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pingouin as pg
import scikit_posthocs as sp
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score


def describe_distribution(series: pd.Series) -> Dict[str, float]:
    clean = series.dropna()
    return {
        "mean": clean.mean(),
        "median": clean.median(),
        "std": clean.std(ddof=1),
        "skew": stats.skew(clean),
        "kurtosis": stats.kurtosis(clean),
    }


def normality_pvalue(series: pd.Series) -> float:
    clean = series.dropna()
    if len(clean) < 8:
        return np.nan
    _, p_value = stats.normaltest(clean)
    return float(p_value)


def mann_whitney(a: pd.Series, b: pd.Series, alternative: str = "two-sided") -> float:
    """Mann-Whitney U test. 
    
    Parameters:
    - alternative: 'two-sided' (default), 'greater' (a > b), or 'less' (a < b)
    
    Returns:
    - p-value as float
    """
    a_clean = a.dropna()
    b_clean = b.dropna()
    
    # Validate we have data
    if len(a_clean) == 0 or len(b_clean) == 0:
        return np.nan
    
    # Use scipy for more direct control
    u_stat, p_val = stats.mannwhitneyu(a_clean, b_clean, alternative=alternative)
    return float(p_val)


def mann_whitney_detailed(a: pd.Series, b: pd.Series, alternative: str = "two-sided") -> Tuple[float, float, float, float]:
    """Mann-Whitney U test with diagnostics. 
    
    Returns: (p_value, u_statistic, median_a, median_b)
    """
    a_clean = a.dropna()
    b_clean = b.dropna()
    
    if len(a_clean) == 0 or len(b_clean) == 0:
        return np.nan, np.nan, np.nan, np.nan
    
    u_stat, p_val = stats.mannwhitneyu(a_clean, b_clean, alternative=alternative)
    return float(p_val), float(u_stat), float(a_clean.median()), float(b_clean.median())


def pearson_corr(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """Pearson correlation coefficient and p-value."""
    res = pg.corr(a, b, method="pearson")
    return float(res.iloc[0]["r"]), float(res.iloc[0]["p-val"])


def spearman_corr(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """Spearman rank correlation coefficient and p-value."""
    res = pg.corr(a, b, method="spearman")
    return float(res.iloc[0]["r"]), float(res.iloc[0]["p-val"])


def kendall_corr(a: pd.Series, b: pd.Series) -> Tuple[float, float]:
    """Kendall's Tau correlation coefficient and p-value."""
    res = pg.corr(a, b, method="kendall")
    return float(res.iloc[0]["r"]), float(res.iloc[0]["p-val"])


def kruskal_pvalue(df: pd.DataFrame) -> float:
    res = pg.kruskal(data=df, dv="critical_temp", between="Number of elements")
    return float(res.iloc[0]["p-unc"])


def dunn_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return sp.posthoc_dunn(df, val_col="critical_temp", group_col="Number of elements", p_adjust="holm")


def prepare_clustering_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    """Prepare and scale features for clustering analysis.
    
    Returns: (scaled_df, scaler, feature_names)
    """
    # Select numeric features for clustering
    feature_cols = [
        "critical_temp",
        "mean_atomic_mass",
        "mean_Density",
        "mean_FusionHeat",
        "mean_ThermalConductivity",
        "mean_ElectronAffinity",
        "mean_Valence",
        "Number of elements",
    ]
    
    available_cols = [col for col in feature_cols if col in df.columns]
    X = df[available_cols].dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return pd.DataFrame(X_scaled, columns=available_cols, index=X.index), scaler, np.array(available_cols)


def find_optimal_clusters(X: np.ndarray, max_k: int = 10) -> Dict[str, any]:
    """Find optimal number of clusters using elbow method and silhouette score.
    
    Returns: Dict with inertias, silhouette_scores, and optimal k
    """
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        davies_bouldin_scores.append(davies_bouldin_score(X, labels))
    
    # Find optimal k (best silhouette score)
    optimal_k = K_range[np.argmax(silhouette_scores)]
    
    return {
        "k_range": list(K_range),
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "davies_bouldin_scores": davies_bouldin_scores,
        "optimal_k": optimal_k,
    }


def perform_kmeans_clustering(X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, KMeans]:
    """Perform K-means clustering.
    
    Returns: (cluster_labels, kmeans_model)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    return labels, kmeans


def perform_hierarchical_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Perform Agglomerative (Hierarchical) clustering.
    
    Returns: cluster_labels
    """
    clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = clusterer.fit_predict(X)
    return labels


def perform_gmm_clustering(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Perform Gaussian Mixture Model clustering.
    
    Returns: cluster_labels
    """
    gmm = GaussianMixture(n_components=n_clusters, random_state=42, n_init=10)
    labels = gmm.fit_predict(X)
    return labels


def perform_dbscan_clustering(X: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> np.ndarray:
    """Perform DBSCAN (density-based) clustering.
    
    Returns: cluster_labels (noise points labeled as -1)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels


def perform_clustering(X: np.ndarray, algorithm: str, n_clusters: int = None, **kwargs) -> np.ndarray:
    """Unified clustering interface.
    
    Parameters:
    - algorithm: 'kmeans', 'hierarchical', 'gmm', or 'dbscan'
    - n_clusters: number of clusters (not used for DBSCAN)
    - kwargs: algorithm-specific parameters
    
    Returns: cluster_labels
    """
    if algorithm == "kmeans":
        labels, _ = perform_kmeans_clustering(X, n_clusters)
        return labels
    elif algorithm == "hierarchical":
        return perform_hierarchical_clustering(X, n_clusters)
    elif algorithm == "gmm":
        return perform_gmm_clustering(X, n_clusters)
    elif algorithm == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        return perform_dbscan_clustering(X, eps, min_samples)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def reduce_to_2d(X: np.ndarray, method: str = "pca") -> np.ndarray:
    """Reduce features to 2D for visualization.
    
    Parameters:
    - method: 'pca' (default)
    
    Returns: 2D array of shape (n_samples, 2)
    """
    if method == "pca":
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(X)
    else:
        raise ValueError(f"Unknown method: {method}")

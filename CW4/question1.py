# Question 1: K-Means Clustering (20 marks)
# Clusters ECG heartbeats into 3 groups using K-Means.
# Evaluates with and without PCA dimensionality reduction.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

from dataloader import X_ecg, y_ecg, CLASS_NAMES
from plot_style import apply_style, BLUE, finish_figure
from clustering import (
    reassign_labels, compute_metrics, print_metrics,
    plot_confusion_matrix, print_summary, FIGURES_DIR,
)

apply_style()


# Scale features - wavelet features have different ranges (mean, std, kurtosis etc.)
# K-Means uses Euclidean distance, so unscaled features with larger ranges dominate.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ecg)


def run_kmeans(X, name, random_state=4):
    """
    Run K-Means, show raw results, then reassign labels.
    K-Means cluster IDs are arbitrary (no access to true labels),
    so we use the Hungarian algorithm to find the optimal mapping
    from cluster IDs to true class labels.
    """
    km = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    y_pred_raw = km.fit_predict(X)

    # Show raw (unaligned) accuracy to demonstrate the problem
    raw_acc = np.mean(y_ecg == y_pred_raw)
    print(f"  Raw cluster accuracy (before label alignment): {raw_acc:.4f}")
    print(f"  Cluster sizes: {dict(zip(*np.unique(y_pred_raw, return_counts=True)))}")

    # Reassign labels using Hungarian algorithm
    y_pred = reassign_labels(y_ecg, y_pred_raw)
    metrics = compute_metrics(y_ecg, y_pred)
    print(f"  Aligned accuracy (after label reassignment):   {metrics['accuracy']:.4f}")
    return y_pred, metrics


# Part 1: K-Means on all 83 features (scaled)
def run_full_features():
    print("\n" + "=" * 60)
    print("  Part 1: K-Means on All 83 Features (Scaled)")
    print("=" * 60)
    print(f"  Dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"  Classes: {dict(zip(CLASS_NAMES, np.bincount(y_ecg)))}")

    y_pred, metrics = run_kmeans(X_scaled, "K-Means (83 features)")
    print_metrics("K-Means (83 features)", metrics)
    plot_confusion_matrix("K-Means (83 features)", y_ecg, y_pred, "q1_cm_full.png")
    return metrics


# Part 2: PCA dimensionality reduction
def run_pca_analysis():
    print("\n" + "=" * 60)
    print("  Part 2: PCA Dimensionality Reduction")
    print("=" * 60)

    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components_90 = np.argmax(cumvar >= 0.90) + 1
    print(f"  Components for >90% variance: {n_components_90}")
    print(f"  Variance captured: {cumvar[n_components_90 - 1]:.4f}")

    # Plot cumulative explained variance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar) + 1), cumvar, 'o-', markersize=3, color=BLUE)
    ax.axhline(y=0.90, color='red', linestyle='--', label='90% threshold')
    ax.axvline(x=n_components_90, color='gray', linestyle=':', label=f'{n_components_90} components')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('PCA Cumulative Explained Variance')
    ax.legend()
    finish_figure(fig, os.path.join(FIGURES_DIR, 'q1_pca_variance.png'))
    print(f"  Saved: q1_pca_variance.png")

    return n_components_90


# Part 3: K-Means on PCA-reduced features (with whitening).
# whiten=True scales each component to unit variance so all components
# contribute equally to Euclidean distance. Without whitening, the first
# few components dominate distance calculations and lower-variance
# components are effectively ignored. K-Means specifically benefits from
# whitening because its objective is purely distance-based.
def run_pca_kmeans(n_components):
    print("\n" + "=" * 60)
    print(f"  Part 3: K-Means on PCA-Reduced Features ({n_components} components, whitened)")
    print("=" * 60)

    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Reduced: {X_scaled.shape[1]} -> {X_pca.shape[1]} features")
    print(f"  Variance retained: {sum(pca.explained_variance_ratio_):.4f}")

    y_pred, metrics = run_kmeans(X_pca, f"K-Means (PCA {n_components})")
    print_metrics(f"K-Means (PCA {n_components})", metrics)
    plot_confusion_matrix(f"K-Means (PCA {n_components})", y_ecg, y_pred,
                          "q1_cm_pca.png")
    return metrics


# Compute on import (available to Q3 for cross-comparison)
metrics_full = run_full_features()
n_comp = run_pca_analysis()
metrics_pca = run_pca_kmeans(n_comp)

# Main
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Q1: K-Means Clustering of ECG Heartbeats")
    print("=" * 60)

    print_summary("K-Means: Full Features vs PCA", [
        ("K-Means (83 features)", metrics_full),
        (f"K-Means (PCA {n_comp})", metrics_pca),
    ])

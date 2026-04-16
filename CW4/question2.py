# Question 2: Gaussian Mixture Model Clustering (20 marks)
# Clusters ECG heartbeats into 3 groups using GMM-EM.
# Evaluates with and without PCA dimensionality reduction.

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from dataloader import X_ecg, y_ecg, CLASS_NAMES
from plot_style import apply_style
from clustering import (
    reassign_labels, compute_metrics, print_metrics,
    plot_confusion_matrix, print_summary,
)

apply_style()

# Scale features (same justification as Q1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ecg)


def run_gmm(X, name, random_state=9):
    """
    Run GMM-EM, show raw results, then reassign labels.
    GMM assigns probabilistic cluster memberships; predict() returns
    the most likely cluster for each sample.
    """
    gmm = GaussianMixture(n_components=3, random_state=random_state)
    y_pred_raw = gmm.fit_predict(X)

    raw_acc = np.mean(y_ecg == y_pred_raw)
    print(f"  Raw cluster accuracy (before label alignment): {raw_acc:.4f}")
    print(f"  Cluster sizes: {dict(zip(*np.unique(y_pred_raw, return_counts=True)))}")

    y_pred = reassign_labels(y_ecg, y_pred_raw)
    metrics = compute_metrics(y_ecg, y_pred)
    print(f"  Aligned accuracy (after label reassignment):   {metrics['accuracy']:.4f}")
    return y_pred, metrics


# Part 1: GMM on all 83 features (scaled)
def run_full_features():
    print("\n" + "=" * 60)
    print("  Part 1: GMM on All 83 Features (Scaled)")
    print("=" * 60)
    print(f"  Dataset: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
    print(f"  Classes: {dict(zip(CLASS_NAMES, np.bincount(y_ecg)))}")

    y_pred, metrics = run_gmm(X_scaled, "GMM (83 features)")
    print_metrics("GMM (83 features)", metrics)
    plot_confusion_matrix("GMM (83 features)", y_ecg, y_pred, "q2_cm_full.png")
    return metrics


# Part 2: GMM on PCA-reduced features
def run_pca_gmm():
    print("\n" + "=" * 60)
    print("  Part 2: GMM on PCA-Reduced Features")
    print("=" * 60)

    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(cumvar >= 0.90) + 1

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    print(f"  Reduced: {X_scaled.shape[1]} -> {X_pca.shape[1]} features")
    print(f"  Variance retained: {sum(pca.explained_variance_ratio_):.4f}")

    y_pred, metrics = run_gmm(X_pca, f"GMM (PCA {n_components})")
    print_metrics(f"GMM (PCA {n_components})", metrics)
    plot_confusion_matrix(f"GMM (PCA {n_components})", y_ecg, y_pred, "q2_cm_pca.png")
    return metrics, n_components


# Compute on import (available to Q3 for cross-comparison)
metrics_full = run_full_features()
metrics_pca, n_comp = run_pca_gmm()

# Main
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Q2: GMM-EM Clustering of ECG Heartbeats")
    print("=" * 60)

    print_summary("GMM: Full Features vs PCA", [
        ("GMM (83 features)", metrics_full),
        (f"GMM (PCA {n_comp})", metrics_pca),
    ])

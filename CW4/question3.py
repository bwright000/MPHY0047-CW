# Question 3: Agglomerative Clustering (20 marks)
# Clusters ECG heartbeats using agglomerative clustering with 4 linkage methods.
# Selects best linkage, evaluates with PCA, then compares all 3 clustering methods (Q1-Q3).

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from dataloader import X_ecg, y_ecg, CLASS_NAMES
from plot_style import apply_style
from clustering import (
    reassign_labels, compute_metrics, print_metrics,
    plot_confusion_matrix, print_summary,
)

# Import Q1 and Q2 results for cross-comparison.
# Each method uses its best-performing PCA configuration:
#   Q1 (K-Means): whitened PCA  - whitening helps centroid-distance methods
#   Q2 (GMM):     unwhitened PCA - whitening damages covariance estimation
#   Q3 (Ward):   unwhitened PCA - whitening damages hierarchical merges
# See §3.7 for the whitening sensitivity analysis that motivates this choice.
from question1 import metrics_full as q1_full, metrics_pca as q1_pca
from question2 import metrics_full as q2_full, metrics_pca as q2_pca

apply_style()

# Scale features (same as Q1/Q2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_ecg)

LINKAGES = ['single', 'complete', 'average', 'ward']


def run_agglomerative(X, linkage):
    """
    Run agglomerative clustering with given linkage, reassign labels.
    Returns (y_pred, metrics).
    """
    ac = AgglomerativeClustering(n_clusters=3, linkage=linkage)
    y_pred_raw = ac.fit_predict(X)

    raw_acc = np.mean(y_ecg == y_pred_raw)
    print(f"  Linkage: {linkage:8s} | Raw accuracy: {raw_acc:.4f} | Cluster sizes: {dict(zip(*np.unique(y_pred_raw, return_counts=True)))}")
    y_pred = reassign_labels(y_ecg, y_pred_raw)
    metrics = compute_metrics(y_ecg, y_pred)
    print(f"  Linkage: {linkage:8s} | Aligned accuracy: {metrics['accuracy']:.4f}")
    return y_pred, metrics

# Part 1: Try all 4 linkage methods on full features
def run_linkage_comparison():
    """
    Run all 4 linkage methods, print accuracy for each, return results dict.

    Pseudocode:
    - for each linkage in LINKAGES:
        - run_agglomerative(X_scaled, linkage)
        - store results
        - print accuracy
    - identify best linkage by accuracy
    - print full metrics + confusion matrix for best only
    - return all results and best linkage name
    """
    results = {}
    predictions = {}
    for linkage in LINKAGES:
        y_pred, metrics = run_agglomerative(X_scaled, linkage)
        results[linkage] = metrics
        predictions[linkage] = y_pred
    best_linkage = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest linkage: {best_linkage} (accuracy: {results[best_linkage]['accuracy']:.4f})")
    print_metrics(f"Agglomerative ({best_linkage})", results[best_linkage])    
    plot_confusion_matrix(f"Agglomerative ({best_linkage})", y_ecg, predictions[best_linkage], f"q3_cm_{best_linkage}.png")
    return results, best_linkage


# Part 2: PCA with best linkage
def run_pca_agglomerative(best_linkage):
    """
    Run best linkage on PCA-reduced features.

    Pseudocode:
    - fit PCA on X_scaled, find n_components for >90% variance
    - transform X_scaled to X_pca
    - run_agglomerative(X_pca, best_linkage)
    - print metrics + confusion matrix
    - return metrics
    """
    # Determine n_components for >=90% variance, then refit at that size.
    pca_probe = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_probe.explained_variance_ratio_)
    n_components_90 = int(np.argmax(cumvar >= 0.90)) + 1

    pca = PCA(n_components=n_components_90)
    X_pca = pca.fit_transform(X_scaled)
    y_pred, metrics = run_agglomerative(X_pca, best_linkage)
    print_metrics(f"Agglomerative ({best_linkage}) - PCA", metrics)
    plot_confusion_matrix(f"Agglomerative ({best_linkage}) PCA", y_ecg, y_pred, f"q3_cm_{best_linkage}_pca.png")
    return metrics


# Part 3: Cross-method comparison (Q1 vs Q2 vs Q3)
def run_cross_comparison(q3_full, q3_pca, best_linkage):
    """
    Print comparison table across all 3 methods.

    Pseudocode:
    - build results list from:
        - q1_full, q1_pca (imported from question1)
        - q2_full, q2_pca (imported from question2)
        - q3_full, q3_pca (passed in)
    - print_summary with all 6 rows
    - print per-class recall comparison for full-feature results:
        - extract recall per class from each confusion matrix
        - show which beat types are easiest/hardest per method
    """
    # Full comparison table
    print_summary("Cross-Method Comparison (All Features)", [
        ("K-Means (83 features)", q1_full),
        ("GMM (83 features)", q2_full),
        (f"Agglomerative {best_linkage} (83)", q3_full),
    ])

    print_summary("Cross-Method Comparison (PCA)", [
        ("K-Means (PCA 20)", q1_pca),
        ("GMM (PCA 20)", q2_pca),
        (f"Agglomerative {best_linkage} (PCA 20)", q3_pca),
    ])

    # Per-class recall comparison (full features only)
    print(f"\n{'=' * 70}")
    print("  Per-Class Recall Comparison (All Features)")
    print(f"{'=' * 70}")
    print(f"  {'Method':<30} {'N':>8} {'S':>8} {'V':>8}")
    print(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}")

    for name, metrics in [("K-Means", q1_full), ("GMM", q2_full),
                          (f"Agglomerative ({best_linkage})", q3_full)]:
        cm = metrics['cm']
        # Per-class recall = diagonal / row sum
        recalls = cm.diagonal() / cm.sum(axis=1)
        print(f"  {name:<30} {recalls[0]:>8.4f} {recalls[1]:>8.4f} {recalls[2]:>8.4f}")


# Main
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Q3: Agglomerative Clustering of ECG Heartbeats")
    print("=" * 60)

    # Step 1: Compare linkage methods
    linkage_results, best_linkage = run_linkage_comparison()

    # Step 2: PCA with best linkage
    q3_pca = run_pca_agglomerative(best_linkage)

    # Step 3: Cross-method comparison
    q3_full = linkage_results[best_linkage]
    run_cross_comparison(q3_full, q3_pca, best_linkage)

# Shared clustering utilities for Q1, Q2, Q3.
# Label re-assignment, metrics, and confusion matrix plotting.

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os

from plot_style import finish_figure
from dataloader import CLASS_NAMES

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def reassign_labels(y_true, y_pred):
    """
    Find optimal label mapping from predicted clusters to true labels.
    Uses the Hungarian algorithm on the confusion matrix to maximise
    the total number of correctly assigned samples.

    Returns re-mapped y_pred array.
    """
    cm = confusion_matrix(y_true, y_pred)
    # Hungarian algorithm minimises cost; negate to maximise matches
    row_ind, col_ind = linear_sum_assignment(-cm)
    # Build mapping: predicted label -> true label
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[label] for label in y_pred])


def compute_metrics(y_true, y_pred):
    """Compute clustering evaluation metrics. Returns dict."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cm': confusion_matrix(y_true, y_pred),
    }


def print_metrics(name, metrics):
    """Print clustering metrics as a formatted table."""
    cm = metrics['cm']
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print(f"  Confusion Matrix:")
    header = "              " + "  ".join(f"{c:>6}" for c in CLASS_NAMES)
    print(header)
    for i, row_name in enumerate(CLASS_NAMES):
        row = "    " + f"{row_name:>9}" + "  ".join(f"{cm[i, j]:>6d}" for j in range(cm.shape[1]))
        print(row)


def plot_confusion_matrix(name, y_true, y_pred, filename):
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=CLASS_NAMES,
        cmap='Blues',
        ax=ax,
    )
    ax.set_title(f'{name}\nConfusion Matrix')
    path = os.path.join(FIGURES_DIR, filename)
    finish_figure(fig, path)
    print(f"  Saved: {path}")


def print_summary(title, results):
    """
    Print a summary comparison table.
    results: list of (name, metrics_dict) tuples.
    """
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Method':<35} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, m in results:
        print(f"  {name:<35} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")

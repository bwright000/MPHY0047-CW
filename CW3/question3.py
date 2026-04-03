# Question 3: COVID CT Classification with LDA/QDA (10 marks)
# Classifies CT images as COVID-positive vs COVID-negative.
# Two classifiers: Linear Discriminant Analysis, Quadratic Discriminant Analysis.
# 5-fold stratified cross-validation on flattened 120x120 grayscale images.

import numpy as np
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import os

from dataloader import load_covid_data
from plot_style import apply_style, finish_figure
from classifiers import run_cv, compute_metrics, print_metrics, plot_confusion_matrix, print_summary

apply_style()

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_DIR, 'figures')


# Load and explore dataset

print("Loading COVID CT images...")
X_img, y, filenames = load_covid_data()
print(f"  Images loaded: {X_img.shape[0]}")
print(f"  Image size: {X_img.shape[1]}x{X_img.shape[2]}")
print(f"  COVID (1): {np.sum(y == 1)}, Non-COVID (0): {np.sum(y == 0)}")


def plot_sample_images(X_img, y):
    """Display a grid of sample COVID and Non-COVID images."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))

    covid_idx = np.where(y == 1)[0]
    non_covid_idx = np.where(y == 0)[0]

    for i in range(5):
        axes[0, i].imshow(X_img[covid_idx[i]], cmap='gray')
        axes[0, i].set_title('COVID', fontsize=9)
        axes[0, i].axis('off')

        axes[1, i].imshow(X_img[non_covid_idx[i]], cmap='gray')
        axes[1, i].set_title('Non-COVID', fontsize=9)
        axes[1, i].axis('off')

    fig.suptitle('Sample CT Images', fontsize=13)
    path = os.path.join(FIGURES_DIR, 'q3_sample_images.png')
    finish_figure(fig, path)
    print(f"  Saved: {path}")


# Flatten images to feature vectors
X = X_img.reshape(X_img.shape[0], -1)
print(f"  Feature vector length: {X.shape[1]}")


# Classifiers

# QDA requires n_samples > n_features per class. With 14,400 features
# and ~300 samples per class, we reduce dimensionality via PCA first.
CLASSIFIERS = [
    ('LDA', LinearDiscriminantAnalysis(), None),
    ('QDA', Pipeline([
        ('pca', PCA(n_components=100, random_state=42)),
        ('qda', QuadraticDiscriminantAnalysis(reg_param=0.5)),
    ]), None),
]


# Main

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Q3: COVID CT Classification - LDA / QDA")
    print("=" * 60)

    plot_sample_images(X_img, y)

    all_results = []

    for name, clf, param_grid in CLASSIFIERS:
        print(f"\n  Running {name}...")
        y_true, y_pred, _ = run_cv(name, clf, X, y, param_grid=param_grid)
        metrics = compute_metrics(y_true, y_pred)
        print_metrics(name, metrics)
        all_results.append((name, metrics))

        plot_confusion_matrix(name, y_true, y_pred, f'q3_cm_{name.lower()}.png')

    print_summary("Summary Comparison", all_results)

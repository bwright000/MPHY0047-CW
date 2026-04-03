# Shared classification utilities for Q3, Q4, Q5.
# Provides cross-validation, metrics reporting, and confusion matrix plotting.

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    precision_score, f1_score, accuracy_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os

from plot_style import finish_figure

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

N_FOLDS = 5


def run_cv(name, estimator, X, y, param_grid=None, n_folds=N_FOLDS):
    """
    Run stratified k-fold CV, optionally with grid search.

    Parameters
    ----------
    name : str - classifier name (for printing)
    estimator : sklearn estimator or pipeline
    X, y : feature matrix and labels
    param_grid : dict or None - if provided, runs GridSearchCV per fold
    n_folds : int - number of CV folds

    Returns
    -------
    y_true, y_pred : arrays of concatenated true/predicted labels
    best_params : list of best params per fold (None entries if no grid search)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_true_all = []
    y_pred_all = []
    best_params = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if param_grid is not None:
            gs = GridSearchCV(
                clone(estimator), param_grid,
                cv=4,
                scoring='accuracy',
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            y_pred = gs.best_estimator_.predict(X_test)
            best_params.append(gs.best_params_)
            print(f"    Fold {fold + 1}: {np.sum(y_pred == y_test)}/{len(y_test)} correct | best: {gs.best_params_}")
        else:
            clf = clone(estimator)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            best_params.append(None)
            print(f"    Fold {fold + 1}: {np.sum(y_pred == y_test)}/{len(y_test)} correct")

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return np.array(y_true_all), np.array(y_pred_all), best_params


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics.

    Returns dict with precision, f1, accuracy, recall_pos, recall_neg, cm.
    """
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred),
        'recall_pos': recall_score(y_true, y_pred, pos_label=1),
        'recall_neg': recall_score(y_true, y_pred, pos_label=0),
        'cm': confusion_matrix(y_true, y_pred),
    }


def print_metrics(name, metrics, class_labels=('Non-COVID', 'COVID'), show_recall=False):
    """Print classification metrics as a formatted table."""
    cm = metrics['cm']
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  F1 Score:           {metrics['f1']:.4f}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    if show_recall:
        print(f"  Recall ({class_labels[1]}):     {metrics['recall_pos']:.4f}")
        print(f"  Recall ({class_labels[0]}): {metrics['recall_neg']:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"              {class_labels[0]:>9}  {class_labels[1]:>5}")
    print(f"    {class_labels[0]:>9}    {cm[0, 0]:>4d}    {cm[0, 1]:>4d}")
    print(f"    {class_labels[1]:>9}    {cm[1, 0]:>4d}    {cm[1, 1]:>4d}")


def plot_confusion_matrix(name, y_true, y_pred, filename,
                          class_labels=('Non-COVID', 'COVID'),
                          cv_label='5-fold CV'):
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=list(class_labels),
        cmap='Blues',
        ax=ax,
    )
    ax.set_title(f'{name}\nConfusion Matrix ({cv_label})')
    path = os.path.join(FIGURES_DIR, filename)
    finish_figure(fig, path)
    print(f"  Saved: {path}")


def print_summary(title, results, show_recall=False):
    """
    Print a summary comparison table.

    Parameters
    ----------
    title : str
    results : list of (name, metrics_dict) tuples
    show_recall : bool - include per-class recall columns
    """
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    if show_recall:
        print(f"  {'Classifier':<32} {'Prec':>6} {'F1':>6} {'Acc':>6} {'R+':>6} {'R-':>6}")
        print(f"  {'-' * 32} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")
        for name, m in results:
            print(f"  {name:<32} {m['precision']:>6.3f} {m['f1']:>6.3f} {m['accuracy']:>6.3f} {m['recall_pos']:>6.3f} {m['recall_neg']:>6.3f}")
    else:
        print(f"  {'Classifier':<32} {'Prec':>8} {'F1':>8} {'Acc':>8}")
        print(f"  {'-' * 32} {'-' * 8} {'-' * 8} {'-' * 8}")
        for name, m in results:
            print(f"  {name:<32} {m['precision']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>8.4f}")

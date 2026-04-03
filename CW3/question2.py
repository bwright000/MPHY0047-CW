# Question 2: JIGSAWS Classification (25 marks)
# Classifies surgeons as expert vs novice using kinematic features from Q1.
# Three classifiers: Bagging DT (Gini), Random Forest (entropy), KNN.
# Leave-One-SuperTrial-Out (LOSO) cross-validation.

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import os
from question1 import sig_params
from dataloader import NUM_TRIALS
from plot_style import apply_style, finish_figure
apply_style()

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


# Load features from Q1

data = np.load(os.path.join(_DIR, 'q1_features.npz'), allow_pickle=True)
X_all = data['X']                          # (40, 22)
y = data['y']                              # (40,)
trial_ids = data['trial_ids']              # (40,)  values 0..4
feature_names = list(data['feature_names'])


# Feature selection
# 12 of 22 parameters significantly distinguish experts from novices
# (Mann-Whitney U, p < 0.05 in Q1).

SELECTED_FEATURES = sig_params

selected_idx = [feature_names.index(f) for f in SELECTED_FEATURES]
X = X_all[:, selected_idx]

print(f"Feature matrix: {X_all.shape[1]} total features -> {X.shape[1]} selected")
print(f"Selected features:")
for name in SELECTED_FEATURES:
    print(f"  - {name}")


# LOSO cross-validation

def loso_split(trial_ids):
    """
    Generate LOSO train/test index pairs.
    Yields (train_idx, test_idx) for each of 5 folds,
    where each fold holds out one trial number.
    """
    for trial in range(NUM_TRIALS):
        test_mask = (trial_ids == trial)
        train_mask = ~test_mask
        yield np.where(train_mask)[0], np.where(test_mask)[0]


# Classifier definitions
# Each entry: (name, estimator, param_grid_or_None, needs_scaling)

CLASSIFIERS = [
    (
        'Bagging Decision Tree (Gini)',
        BaggingClassifier(
            estimator=DecisionTreeClassifier(criterion='gini'),
            n_estimators=50,
            random_state=42,
        ),
        None,
        False,
    ),
    (
        'Random Forest (Entropy)',
        RandomForestClassifier(criterion='entropy', random_state=42),
        {'n_estimators': [10, 25, 50, 100, 200]},
        False,
    ),
    (
        'KNN',
        KNeighborsClassifier(),
        {
            'n_neighbors': [1, 3, 5, 7, 9],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
        },
        True,
    ),
]


# Run classification

def run_classifier(name, estimator, param_grid, needs_scaling, X, y, trial_ids):
    """
    Run a classifier with LOSO CV.
    Returns (y_true_all, y_pred_all, best_params_per_fold).
    """
    y_true_all = []
    y_pred_all = []
    best_params = []

    for fold, (train_idx, test_idx) in enumerate(loso_split(trial_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Fit scaler on training data only
        if needs_scaling:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Grid search with inner CV, or direct fit
        if param_grid is not None:
            gs = GridSearchCV(
                estimator, param_grid,
                cv=4,
                scoring='accuracy',
                n_jobs=-1,
            )
            gs.fit(X_train, y_train)
            clf = gs.best_estimator_
            best_params.append(gs.best_params_)
        else:
            clf = clone(estimator)
            clf.fit(X_train, y_train)
            best_params.append(None)

        y_pred = clf.predict(X_test)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    return np.array(y_true_all), np.array(y_pred_all), best_params


def print_results(name, y_true, y_pred, best_params):
    """Print classification metrics for one classifier."""
    prec = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Precision:  {prec:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Novice  Expert")
    print(f"    Actual Novice  {cm[0, 0]:>4d}    {cm[0, 1]:>4d}")
    print(f"    Actual Expert  {cm[1, 0]:>4d}    {cm[1, 1]:>4d}")

    if best_params[0] is not None:
        print(f"\n  Best hyperparameters per LOSO fold:")
        for fold, params in enumerate(best_params):
            print(f"    Fold {fold + 1}: {params}")

    return prec, f1, acc, cm


def plot_confusion_matrix(name, y_true, y_pred, filename):
    """Save a confusion matrix plot."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=['Novice', 'Expert'],
        cmap='Blues',
        ax=ax,
    )
    ax.set_title(f'{name}\nConfusion Matrix (LOSO CV)')
    path = os.path.join(FIGURES_DIR, filename)
    finish_figure(fig, path)
    print(f"  Saved: {path}")


# Main

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Q2: JIGSAWS Classification - Expert vs Novice")
    print("=" * 60)
    print(f"\n  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Classes: {np.sum(y == 0)} novice, {np.sum(y == 1)} expert")
    print(f"  CV scheme: Leave-One-SuperTrial-Out (5 folds of 8 samples)")

    all_results = []

    for name, estimator, param_grid, needs_scaling in CLASSIFIERS:
        y_true, y_pred, best_params = run_classifier(
            name, estimator, param_grid, needs_scaling, X, y, trial_ids,
        )
        prec, f1, acc, cm = print_results(name, y_true, y_pred, best_params)
        all_results.append((name, prec, f1, acc))

        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plot_confusion_matrix(name, y_true, y_pred, f'q2_cm_{safe_name}.png')

    # Summary comparison table
    print(f"\n{'=' * 60}")
    print("  Summary Comparison")
    print(f"{'=' * 60}")
    print(f"  {'Classifier':<35} {'Prec':>8} {'F1':>8} {'Acc':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8} {'-' * 8}")
    for name, prec, f1, acc in all_results:
        print(f"  {name:<35} {prec:>8.4f} {f1:>8.4f} {acc:>8.4f}")

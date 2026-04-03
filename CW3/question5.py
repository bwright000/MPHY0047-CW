# Question 5: Imbalanced COVID CT Classification (25 marks)
# Creates imbalanced dataset (349 COVID vs 197 Non-COVID) by removing last 200 Non-COVID.
# Runs SVM (linear + RBF) with and without class_weight='balanced'.
# Compares with Q4's balanced-dataset results.

import numpy as np
from sklearn.svm import SVC

from dataloader import load_covid_data_imbalanced, extract_hog_features
from plot_style import apply_style
from classifiers import run_cv, compute_metrics, print_metrics, plot_confusion_matrix, print_summary

apply_style()

C_VALUES = [0.01, 0.1, 1, 10, 100]
GAMMA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1]


# Load imbalanced data and extract HOG features

print("Loading imbalanced COVID CT dataset...")
X_img, y, filenames = load_covid_data_imbalanced()
print(f"  Images loaded: {X_img.shape[0]}")
print(f"  COVID (1): {np.sum(y == 1)}, Non-COVID (0): {np.sum(y == 0)}")
print(f"  Imbalance ratio: {np.sum(y == 1) / np.sum(y == 0):.2f}:1")

print("Extracting HOG features...")
X = extract_hog_features(X_img)
print(f"  HOG feature vector length: {X.shape[1]}")


# SVM classifiers - imbalanced and balanced variants

CLASSIFIERS = [
    (
        'SVM Linear (imbalanced)',
        SVC(kernel='linear', random_state=42),
        {'C': C_VALUES},
    ),
    (
        'SVM RBF (imbalanced)',
        SVC(kernel='rbf', random_state=42),
        {'C': C_VALUES, 'gamma': GAMMA_VALUES},
    ),
    (
        'SVM Linear (balanced weights)',
        SVC(kernel='linear', class_weight='balanced', random_state=42),
        {'C': C_VALUES},
    ),
    (
        'SVM RBF (balanced weights)',
        SVC(kernel='rbf', class_weight='balanced', random_state=42),
        {'C': C_VALUES, 'gamma': GAMMA_VALUES},
    ),
]


# Main

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Q5: Imbalanced COVID CT Classification")
    print("=" * 60)
    print(f"\n  Dataset: {np.sum(y == 1)} COVID vs {np.sum(y == 0)} Non-COVID")

    all_results = []

    for name, estimator, param_grid in CLASSIFIERS:
        print(f"\n  Running {name}...")
        y_true, y_pred, best_params = run_cv(name, estimator, X, y, param_grid=param_grid)
        metrics = compute_metrics(y_true, y_pred)
        print_metrics(name, metrics, show_recall=True)
        all_results.append((name, metrics))

        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plot_confusion_matrix(name, y_true, y_pred, f'q5_cm_{safe_name}.png')

    # Summary
    print_summary("Imbalanced vs Balanced Weights", all_results, show_recall=True)

    # Full comparison including Q4
    print(f"\n{'=' * 70}")
    print("  Full Comparison: Q4 (balanced data) vs Q5 (imbalanced data)")
    print(f"{'=' * 70}")
    print(f"  {'Setting':<32} {'Prec':>6} {'F1':>6} {'Acc':>6} {'R+':>6} {'R-':>6}")
    print(f"  {'-' * 32} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6}")
    print(f"  {'SVM RBF Q4 (balanced data)':<32} {'0.814':>6} {'0.801':>6} {'0.816':>6} {'---':>6} {'---':>6}")
    for name, m in all_results:
        if 'RBF' in name:
            print(f"  {name:<32} {m['precision']:>6.3f} {m['f1']:>6.3f} {m['accuracy']:>6.3f} {m['recall_pos']:>6.3f} {m['recall_neg']:>6.3f}")

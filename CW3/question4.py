# Question 4: COVID CT Classification with SVM + HOG (25 marks)
# Extracts HOG features from CT images, then classifies with SVM (linear + RBF).
# Grid search over C and gamma. 5-fold stratified cross-validation.
# Compares with Q3 LDA/QDA results and published literature.

import numpy as np
from sklearn.svm import SVC

from dataloader import load_covid_data, extract_hog_features
from plot_style import apply_style
from classifiers import run_cv, compute_metrics, print_metrics, plot_confusion_matrix

apply_style()

C_VALUES = [0.01, 0.1, 1, 10, 100]
GAMMA_VALUES = [0.0001, 0.001, 0.01, 0.1, 1]


# Load data and extract HOG features

print("Loading COVID CT images...")
X_img, y, filenames = load_covid_data()
print(f"  Images loaded: {X_img.shape[0]}")
print(f"  COVID (1): {np.sum(y == 1)}, Non-COVID (0): {np.sum(y == 0)}")

print("Extracting HOG features...")
X = extract_hog_features(X_img)
print(f"  HOG feature vector length: {X.shape[1]}")


# SVM classifiers with grid search

CLASSIFIERS = [
    (
        'SVM Linear',
        SVC(kernel='linear', random_state=42),
        {'C': C_VALUES},
    ),
    (
        'SVM RBF',
        SVC(kernel='rbf', random_state=42),
        {'C': C_VALUES, 'gamma': GAMMA_VALUES},
    ),
]


# Main

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Q4: COVID CT Classification - SVM + HOG")
    print("=" * 60)

    all_results = []

    for name, estimator, param_grid in CLASSIFIERS:
        print(f"\n  Running {name}...")
        y_true, y_pred, best_params = run_cv(name, estimator, X, y, param_grid=param_grid)
        metrics = compute_metrics(y_true, y_pred)
        print_metrics(name, metrics)
        all_results.append((name, metrics))

        safe_name = name.lower().replace(' ', '_')
        plot_confusion_matrix(name, y_true, y_pred, f'q4_cm_{safe_name}.png')

    # Comparison with Q3
    print(f"\n{'=' * 60}")
    print("  Comparison: Q3 (LDA/QDA) vs Q4 (SVM + HOG)")
    print(f"{'=' * 60}")
    print(f"  {'Classifier':<20} {'Prec':>8} {'F1':>8} {'Acc':>8}")
    print(f"  {'-' * 20} {'-' * 8} {'-' * 8} {'-' * 8}")
    print(f"  {'LDA (Q3)':<20} {'0.6934':>8} {'0.6934':>8} {'0.7131':>8}")
    print(f"  {'QDA (Q3)':<20} {'0.7631':>8} {'0.7781':>8} {'0.7882':>8}")
    for name, m in all_results:
        print(f"  {name + ' (Q4)':<20} {m['precision']:>8.4f} {m['f1']:>8.4f} {m['accuracy']:>8.4f}")

    # Comparison with published literature
    print(f"\n{'=' * 60}")
    print("  Comparison with Published Results")
    print(f"{'=' * 60}")
    print(f"  {'Method':<35} {'F1':>8} {'Acc':>8}")
    print(f"  {'-' * 35} {'-' * 8} {'-' * 8}")
    print(f"  {'Our SVM RBF + HOG':<35} {all_results[1][1]['f1']:>8.4f} {all_results[1][1]['accuracy']:>8.4f}")
    print(f"  {'He et al. Self-Trans DenseNet-169':<35} {'0.85':>8} {'0.86':>8}")
    print(f"  {'Yang et al. TL+CSSL+Masks DN-169':<35} {'0.90':>8} {'0.89':>8}")

# Question 4: Image Alignment and Transformation Analysis
# Computes rigid transforms (rotation, translation) via ECC algorithm,
# performs expert vs novice statistical testing, and linear regression against quality scores.

from dataloader import (test_img, gold_img, gen_impr, crit_perc,
                        NUM_PARTICIPANTS, NUM_VIEWS,
                        EXPERT_RANGE, NOVICE_RANGE, MISSING)
from plot_style import apply_style, scatter_points, reference_line, finish_figure, BLUE
from scipy import stats
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

apply_style()



VIEW_NAMES = [f"View {i+1}" for i in range(NUM_VIEWS)]


def is_missing(p, v):
    """Check if participant p, view v is a missing data entry."""
    return (p, v) in MISSING


def compute_rigid_transforms():
    """
    Compute rotation (degrees) and translation (pixels) for each test image
    vs its gold standard using the ECC algorithm with MOTION_EUCLIDEAN.
    
    Returns:
        rotation_vals, translation_vals: each a 20×10 array (NaN for missing)
    """
    rotation_vals = np.full((NUM_PARTICIPANTS, NUM_VIEWS), np.nan)
    translation_vals = np.full((NUM_PARTICIPANTS, NUM_VIEWS), np.nan)
    for v in range(NUM_VIEWS):

        gold = np.squeeze(gold_img[0][v]).astype(np.uint8)
        for p in range(NUM_PARTICIPANTS):
            if is_missing(p, v):
                continue

            test = np.squeeze(test_img[p][v]).astype(np.uint8)
            # ECC Alignment setup (as per image_align2.py)
            warp_mode = cv2.MOTION_EUCLIDEAN
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-10)

            try:
                (cc, warp_matrix) = cv2.findTransformECC(gold, test, warp_matrix, warp_mode, criteria)
            except cv2.error:
                # ECC failed to converge - Treat as NaN
                continue
            # Extract rotation and translation from warp_matrix
            rotation_vals[p][v] = np.degrees(
                np.arctan2(warp_matrix[1, 0], warp_matrix[0, 0]))
            translation_vals[p][v] = np.sqrt(
                warp_matrix[0, 2]**2 + warp_matrix[1, 2]**2)
    return rotation_vals, translation_vals

def linear_regression_q4(x_vals, y_vals):
    """
    Perform linear regression for each view.
    rotation/translation (independent) -> crit_perc/gen_impr (dependent).

    Args:
        x_vals: 20×10 array of independent variable (rotation or translation)
        y_vals: 20×10 array of dependent variable (crit_perc or gen_impr)

    Returns:
        list of dicts per view with keys: view, slope, intercept, predicted, y_true, rmse, r2
    """
    results = []
    for v in range(NUM_VIEWS):
        x = x_vals[:, v]
        y = y_vals[:, v]
        # Skip NaN in x (missing ECC) and -1 in y (missing scores)
        valid = ~np.isnan(x) & (y != -1)
        if np.sum(valid) < 2:
            results.append({
                "view": VIEW_NAMES[v], "slope": np.nan, "intercept": np.nan,
                "predicted": np.array([]), "y_true": np.array([]),
                "rmse": np.nan, "r2": np.nan
            })
            continue
        x_valid = x[valid]
        y_valid = y[valid]
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        predicted = slope * x_valid + intercept
        rmse = np.sqrt(np.mean((predicted - y_valid) ** 2))
        r2 = r_value ** 2
        results.append({
            "view": VIEW_NAMES[v], "slope": slope, "intercept": intercept,
            "predicted": predicted, "y_true": y_valid,
            "rmse": rmse, "r2": r2
        })
    return results


def get_group_values(metric_vals, view_idx, group_range):
    """
    Extract valid (non-NaN) metric values for a participant group in a given view.

    Args:
        metric_vals: 20x10 array of metric values
        view_idx: view index (0-9)
        group_range: tuple (start, end) for participant indices

    Returns:
        numpy array of valid metric values for the group
    """
    values = []
    for p in range(group_range[0], group_range[1]):
        val = metric_vals[p][view_idx]
        if not np.isnan(val):
            values.append(val)
    return np.array(values)


# Main execution.

if __name__ == "__main__":

    os.makedirs("figures", exist_ok=True)

    # Compute rigid transforms.
    rotation_vals, translation_vals = compute_rigid_transforms()

    # Part i: List rotation and translation values.
    print("PART i: Rotation and Translation Values")

    metric_names = ["Rotation (degrees)", "Translation (pixels)"]
    metric_arrays = [rotation_vals, translation_vals]

    # Print full metric tables
    for m_name, m_vals in zip(metric_names, metric_arrays):
        print(f"Full {m_name} values per participant per view")
        header = f"{'Participant':<14}" + "".join([f"{VIEW_NAMES[v]:<10}" for v in range(NUM_VIEWS)])
        print(header)
        for p in range(NUM_PARTICIPANTS):
            row = f"{'P' + str(p+1):<14}"
            for v in range(NUM_VIEWS):
                val = m_vals[p][v]
                row += f"{val:<10.4f}" if not np.isnan(val) else f"{'N/A':<10}"
            print(row)

    # Part ii: Statistical testing (expert vs novice).
    print("PART ii: Mann-Whitney U Test — Expert vs Novice")
    print("\nH0: No difference in alignment metric between expert and novice groups")
    print("H1: Expert and novice groups differ in alignment metric")
    print(f"Significance level: alpha = 0.05\n")

    # Track significance counts per metric to determine best differentiator
    sig_counts = {name: 0 for name in metric_names}

    for m_name, m_vals in zip(metric_names, metric_arrays):
        print(f"\n{m_name}")
        print(f"{'View':<10} {'Expert mean':<14} {'Novice mean':<14} {'U-stat':<12} {'p-value':<12} {'Significant?':<12}")

        for v in range(NUM_VIEWS):
            expert_vals = get_group_values(m_vals, v, EXPERT_RANGE)
            novice_vals = get_group_values(m_vals, v, NOVICE_RANGE)

            # Mann-Whitney U test (two-sided)
            u_stat, p_value = stats.mannwhitneyu(expert_vals, novice_vals,
                                                  alternative='two-sided')
            sig = "Yes" if p_value < 0.05 else "No"
            if p_value < 0.05:
                sig_counts[m_name] += 1

            print(f"{VIEW_NAMES[v]:<10} {expert_vals.mean():<14.4f} {novice_vals.mean():<14.4f} "
                  f"{u_stat:<12.2f} {p_value:<12.6f} {sig:<12}")

    # Summary: which metric best differentiates expert vs novice
    print("Summary: Number of views with significant differences (p < 0.05)")
    for m_name in metric_names:
        print(f"  {m_name}: {sig_counts[m_name]} / {NUM_VIEWS} views")

    best_metric = max(sig_counts, key=sig_counts.get)
    print(f"\nBest differentiating metric: {best_metric} "
          f"(significant in {sig_counts[best_metric]} / {NUM_VIEWS} views)")

    # Part iii: Linear Regression.
    print("\nPART iii: Linear Regression (rotation/translation -> quality scores)")

    indep_vars = [("Rotation", rotation_vals), ("Translation", translation_vals)]
    dep_vars = [("crit_perc", crit_perc), ("gen_impr", gen_impr)]

    # Collect all regression results across all 4 combinations for ranking
    all_results = []

    for indep_name, indep_vals in indep_vars:
        for dep_name, dep_vals in dep_vars:
            combo = f"{indep_name} -> {dep_name}"
            print(f"\n{combo}")
            print(f"{'View':<10} {'Slope':<10} {'Intercept':<12} {'RMSE':<10} {'R²':<10}")

            results = linear_regression_q4(indep_vals, dep_vals)

            for r in results:
                print(f"{r['view']:<10} {r['slope']:<10.4f} {r['intercept']:<12.4f} "
                      f"{r['rmse']:<10.4f} {r['r2']:<10.4f}")

            # Tag each result with its combo name for global ranking
            for r in results:
                r['combo'] = combo
            all_results.extend(results)

    # Identify top 3 views across ALL combinations by R²
    valid_results = [r for r in all_results if not np.isnan(r['r2'])]
    ranked = sorted(valid_results, key=lambda r: r['r2'], reverse=True)

    print("\nTop 3 Best Performing Views (by R² across all combinations)")

    for r in ranked[:3]:
        print(f"  {r['view']} — {r['combo']}: R² = {r['r2']:.4f}, RMSE = {r['rmse']:.4f}")

        # Plot true vs estimated
        fig, ax = plt.subplots()
        scatter_points(ax, r['y_true'], r['predicted'], color=BLUE)
        min_val = min(r['y_true'].min(), r['predicted'].min())
        max_val = max(r['y_true'].max(), r['predicted'].max())
        reference_line(ax, min_val, max_val)
        ax.set_title(f"{r['view']} \u2014 {r['combo']}\n"
                     f"RMSE = {r['rmse']:.2f}, R\u00b2 = {r['r2']:.4f}")
        ax.set_xlabel(f"True {r['combo'].split(' -> ')[1]}")
        ax.set_ylabel(f"Estimated {r['combo'].split(' -> ')[1]}")
        ax.legend()
        fname = (f"figures/q4iii_{r['combo'].replace(' -> ', '_').replace(' ', '_')}"
                 f"_{r['view'].lower().replace(' ', '_')}.png")
        finish_figure(fig, fname)

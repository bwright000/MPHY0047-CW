from dataloader import gen_impr, crit_perc, NUM_PARTICIPANTS, NUM_VIEWS, EXPERT_RANGE, NOVICE_RANGE, MISSING, get_valid_scores
from plot_style import apply_style, scatter_points, reference_line, finish_figure, BLUE, GREEN, GREY, RED
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os

apply_style()
os.makedirs("figures", exist_ok=True)
VIEW_NAMES = [f"View {i+1}" for i in range(NUM_VIEWS)]


def calculate_pearson(view_idx):
    """
    Calculate Pearson correlation coefficient between general impression
    and criteria percentage scores for a given view.

    Pearson's r measures the linear relationship between two variables:

        r = Σ((xi - x̄)(yi - ȳ)) / sqrt(Σ(xi - x̄)² * Σ(yi - ȳ)²)

    where:
        xi, yi = individual score pairs
        x̄, ȳ  = mean of each score set

    r ranges from -1 (perfect negative) to +1 (perfect positive).
    A higher |r| indicates stronger linear agreement between the two scores.

    Args:
        view_idx: Index of the view (0-9)

    Returns:
        r: Pearson correlation coefficient
        p_value: Two-tailed p-value for testing non-correlation
    """
    gi, cp = get_valid_scores(view_idx)
    r, p_value = stats.pearsonr(gi, cp)
    return r, p_value

def linear_regression(view_idx):
    """
    Perform linear regression: gen_impr (independent) → crit_perc (dependent).
    Returns slope, intercept, predicted values, RMSE, R².
    """
    gi, cp = get_valid_scores(view_idx)
    slope, intercept, r_value, p_value, std_err = stats.linregress(gi, cp)
    gi = np.array(gi)
    cp = np.array(cp)
    predicted = slope * gi + intercept
    rmse = np.sqrt(np.mean((predicted - cp) ** 2))
    r_squared = r_value ** 2
    return slope, intercept, predicted, rmse, r_squared



def plot_true_vs_estimated(view_idx, cp, predicted, rmse, r_squared):
    """
    Plot true criteria percentage vs estimated criteria percentage for a given view.
    Points are color-coded by criteria percentage range (40-70% and 70-100%)
    to support performance commentary across different score ranges.

    Args:
        view_idx: Index of the view (0-9)
        cp: Array of true criteria percentage scores
        predicted: Array of predicted criteria percentage scores
        rmse: Root mean square error of the regression
        r_squared: R² score of the regression
    """
    fig, ax = plt.subplots()

    cp = np.array(cp)
    predicted = np.array(predicted)

    # Define masks for the two criteria percentage ranges of interest
    mask_low = (cp >= 40) & (cp < 70)
    mask_high = (cp >= 70) & (cp <= 100)
    mask_other = ~(mask_low | mask_high)

    # Scatter plot with color-coding by criteria percentage range
    if np.any(mask_other):
        scatter_points(ax, cp[mask_other], predicted[mask_other],
                       color=GREY, label='CP < 40%', alpha=0.55)
    if np.any(mask_low):
        scatter_points(ax, cp[mask_low], predicted[mask_low],
                       color=BLUE, label='CP 40\u201370%')
    if np.any(mask_high):
        scatter_points(ax, cp[mask_high], predicted[mask_high],
                       color=GREEN, label='CP 70\u2013100%')

    # Perfect prediction reference line (y = x)
    min_val = min(cp.min(), predicted.min())
    max_val = max(cp.max(), predicted.max())
    reference_line(ax, min_val, max_val)

    ax.set_title(f'{VIEW_NAMES[view_idx]} \u2014 True vs Estimated Criteria Percentage\n'
                 f'RMSE = {rmse:.2f}, R\u00b2 = {r_squared:.4f}')
    ax.set_xlabel('True Criteria Percentage (%)')
    ax.set_ylabel('Estimated Criteria Percentage (%)')
    ax.legend()
    finish_figure(fig, f'figures/q1_true_vs_estimated_{VIEW_NAMES[view_idx].lower().replace(" ", "_")}.png')


# Main execution.

if __name__ == "__main__":

    # Part i: Pearson Correlation.
    print("PART i: Pearson Correlation Coefficient")
    print(f"{'View':<10} {'Pearson r':<12} {'p-value':<12}")

    pearson_results = []
    for v in range(NUM_VIEWS):
        r, p = calculate_pearson(v)
        pearson_results.append((v, r, p))
        print(f"{VIEW_NAMES[v]:<10} {r:<12.4f} {p:<12.6f}")

    # Identify the view with the highest |r| (strongest linear agreement)
    best_view = max(pearson_results, key=lambda x: abs(x[1]))
    print(f"\nHighest agreement: {VIEW_NAMES[best_view[0]]} (r = {best_view[1]:.4f})")

    # Part ii: Linear Regression.
    print("\nPART ii: Linear Regression (gen_impr -> crit_perc)")
    print(f"{'View':<10} {'Slope':<10} {'Intercept':<12} {'RMSE':<10} {'R²':<10}")

    regression_results = []
    for v in range(NUM_VIEWS):
        slope, intercept, predicted, rmse, r_sq = linear_regression(v)
        gi, cp = get_valid_scores(v)
        cp = np.array(cp)
        regression_results.append((v, slope, intercept, predicted, cp, rmse, r_sq))
        print(f"{VIEW_NAMES[v]:<10} {slope:<10.4f} {intercept:<12.4f} {rmse:<10.4f} {r_sq:<10.4f}")

    # Part iii: Plot 3 best performing views.
    print("\nPART iii: True vs Estimated Plots (3 best views by R²)")

    # Rank by R² (highest = best fit) and select top 3
    ranked = sorted(regression_results, key=lambda x: x[6], reverse=True)
    for v, slope, intercept, predicted, cp, rmse, r_sq in ranked[:3]:
        print(f"Plotting {VIEW_NAMES[v]} (R² = {r_sq:.4f})")
        plot_true_vs_estimated(v, cp, predicted, rmse, r_sq)

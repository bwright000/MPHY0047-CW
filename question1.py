import math
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import expert_total, expert_needle, expert_knot, novice_total, novice_needle, novice_knot


# QUESTION 1

def calculate_mean(data):
    '''
    Compute the arithmetic mean of a list of numbers.
    Formula: x_bar = (1/n) * sum(x_i)  for i = 1..n
    '''

    n = len(data)
    total = 0
    for x in data:
        total += x
    return total / n

def calculate_median(data):
    '''
    Compute the median of a list of numbers.
    For sorted data x_(1) <= x_(2) <= ... <= x_(n):
      If n is odd:  median = x_((n+1)/2)
      If n is even: median = (x_(n/2) + x_(n/2 + 1)) / 2
    '''

    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2

    if n % 2 == 0:
        median = (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        median = sorted_data[mid]
    
    return median

def calculate_variance(data, population=True):
    '''
    Compute the variance of a list of numbers.
    Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
    Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
    The (n-1) divisor is Bessel's correction for unbiased estimation.
    '''

    n = len(data)
    mean = calculate_mean(data)
    variance_sum = 0

    for x in data:
        variance_sum += (x - mean) ** 2

    if population:
        variance = variance_sum / n
    else:
        variance = variance_sum / (n - 1)

    return variance

def calculate_stddev(data, population=True):
    '''
    Compute the standard deviation of a list of numbers.
    Formula: sigma = sqrt(variance)
    Uses population or sample variance depending on the population flag.
    '''

    variance = calculate_variance(data, population)
    stddev = math.sqrt(variance)
    return stddev

def skewness(data, population=True):
    '''
    Compute Fisher's skewness (population) of a list of numbers.
    Formula: gamma_1 = (1/n) * sum( ((x_i - mu) / sigma)^3 )
    Positive = right-skewed, Negative = left-skewed, 0 = symmetric.
    '''

    n = len(data)
    mean = calculate_mean(data)
    std = calculate_stddev(data, population=True)

    skew = 0
    for x in data:
        skew += ((x - mean)/std)**3
    return skew / n


def kurtosis(data, population=True):
    '''
    Compute standard (Pearson) kurtosis (population) of a list of numbers.
    Formula: kappa = (1/n) * sum( ((x_i - mu) / sigma)^4 )
    NOTE: This is standard kurtosis where a normal distribution = 3.
    Excess kurtosis = kappa - 3 (where normal = 0).
    '''
    n = len(data)
    mean = calculate_mean(data)
    std = calculate_stddev(data, population=True)
    
    kurt = 0
    for x in data:
        kurt += ((x -mean)/std)**4
    return kurt / n


def summary_table(data, population=True):
    '''
    Generate a summary table of statistics for a list of numbers.
    '''

    mean = calculate_mean(data)
    median = calculate_median(data)
    variance = calculate_variance(data, population)
    stddev = calculate_stddev(data, population)
    skew = skewness(data, population)
    kurt = kurtosis(data, population)

    summary = {
        'Mean': mean,
        'Median': median,
        'Variance': variance,
        'Standard Deviation': stddev,
        'Skewness': skew,
        'Kurtosis': kurt
    }

    return summary


def print_summary_table(summary, group_name, task_name):
    '''
    Print the summary table in a formatted way.
    '''

    print(f"Summary Statistics for {group_name} - {task_name}:")
    print("=" * 50)
    for stat, value in summary.items():
        print(f"{stat:25}: {value:.4f}")
    print("\n")

def plot_histogram(data, group_name, task_name, color):
    '''
    Plot a single histogram for one group.
    '''
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=10, alpha=0.7, color=color, edgecolor='black')
    plt.title(f'Histogram of {task_name} - {group_name}')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'figures/histogram_{task_name.lower().replace(" ", "_")}_{group_name.lower()}.png', dpi=150)
    plt.show()


def plot_boxplot(data, group_name, task_name, color, ylim=None):
    '''
    Plot a single boxplot for one group.
    ylim: optional (ymin, ymax) tuple to enforce a shared y-axis scale across groups.
    '''
    plt.figure(figsize=(6, 5))
    bp = plt.boxplot(data, patch_artist=True)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    plt.title(f'Boxplot of {task_name} - {group_name}')
    plt.ylabel('Time (seconds)')
    plt.xticks([1], [group_name])
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig(f'figures/boxplot_{task_name.lower().replace(" ", "_")}_{group_name.lower()}.png', dpi=150)
    plt.show()


def generate_all_figures():
    '''
    Generate all 12 figures: 3 time parameters x 2 plot types x 2 groups.
    '''
    import os
    os.makedirs('figures', exist_ok=True)

    params = [
        ('Total Duration', expert_total, novice_total),
        ('Needle Passing', expert_needle, novice_needle),
        ('Knot Tying', expert_knot, novice_knot)
    ]

    figure_count = 0
    for param_name, exp_data, nov_data in params:
        # Compute shared y-axis range so expert and novice boxplots use the same scale,
        # making visual comparison easier. 5% padding added for readability.
        all_values = exp_data + nov_data
        ymin = min(all_values)
        ymax = max(all_values)
        padding = (ymax - ymin) * 0.05
        shared_ylim = (ymin - padding, ymax + padding)

        # Expert histogram
        plot_histogram(exp_data, 'Experts', param_name, 'blue')
        figure_count += 1
        print(f"Figure {figure_count}: Histogram - {param_name} - Experts")

        # Novice histogram
        plot_histogram(nov_data, 'Novices', param_name, 'orange')
        figure_count += 1
        print(f"Figure {figure_count}: Histogram - {param_name} - Novices")

        # Expert boxplot (shared y-axis with novice)
        plot_boxplot(exp_data, 'Experts', param_name, 'blue', ylim=shared_ylim)
        figure_count += 1
        print(f"Figure {figure_count}: Boxplot - {param_name} - Experts")

        # Novice boxplot (shared y-axis with expert)
        plot_boxplot(nov_data, 'Novices', param_name, 'orange', ylim=shared_ylim)
        figure_count += 1
        print(f"Figure {figure_count}: Boxplot - {param_name} - Novices")

    print(f"\nTotal figures generated: {figure_count}")


def identify_outliers(data, group_name, task_name):
    '''
    Identify outliers using Tukey's fences (Tukey, 1977).
    Tukey's inner fences:
      Lower fence = Q1 - 1.5 * IQR
      Upper fence = Q3 + 1.5 * IQR
      where IQR = Q3 - Q1
    Any value outside [lower fence, upper fence] is classified as an outlier.

    Quartiles computed via linear interpolation (Hyndman & Fan Method 7,
    the NumPy/Excel default):
      h = (n - 1) * p       (0-indexed)
      Q_p = x[floor(h)] * (1 - frac(h)) + x[floor(h) + 1] * frac(h)
    '''
    sorted_data = sorted(data)
    n = len(sorted_data)

    # Calculate Q1 and Q3 manually
    q1_idx = (n - 1) * 0.25
    q3_idx = (n - 1) * 0.75

    # Linear interpolation for quartiles
    q1_lower = int(q1_idx)
    q1_frac = q1_idx - q1_lower
    q1 = sorted_data[q1_lower] * (1 - q1_frac) + sorted_data[q1_lower + 1] * q1_frac

    q3_lower = int(q3_idx)
    q3_frac = q3_idx - q3_lower
    q3 = sorted_data[q3_lower] * (1 - q3_frac) + sorted_data[q3_lower + 1] * q3_frac

    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [x for x in data if x < lower_bound or x > upper_bound]

    print(f"Outlier Analysis for {group_name} - {task_name}:")
    print(f"  Q1 = {q1:.2f}, Q3 = {q3:.2f}, IQR = {iqr:.2f}")
    print(f"  Lower bound = {lower_bound:.2f}, Upper bound = {upper_bound:.2f}")
    if outliers:
        print(f"  Outliers detected: {outliers}")
    else:
        print(f"  No outliers detected")
    print()

    return outliers


def analyze_robustness():
    '''
    Compare robustness of time parameters using the Coefficient of Variation.
    Formula: CV = sigma / x_bar
    Lower CV = less relative dispersion = more robust.
    CV is dimensionless, allowing comparison across parameters with different scales.
    Ref: NIST/SEMATECH e-Handbook of Statistical Methods
    '''
    params = [
        ('Total Duration', expert_total, novice_total),
        ('Needle Passing', expert_needle, novice_needle),
        ('Knot Tying', expert_knot, novice_knot)
    ]

    print("=" * 60)
    print("ROBUSTNESS ANALYSIS - Coefficient of Variation (CV)")
    print("CV = Standard Deviation / Mean (lower = more robust)")
    print("=" * 60)

    cv_results = []
    for param_name, exp_data, nov_data in params:
        # Expert CV
        exp_mean = calculate_mean(exp_data)
        exp_std = calculate_stddev(exp_data)
        exp_cv = exp_std / exp_mean if exp_mean != 0 else 0

        # Novice CV
        nov_mean = calculate_mean(nov_data)
        nov_std = calculate_stddev(nov_data)
        nov_cv = nov_std / nov_mean if nov_mean != 0 else 0

        # Combined CV (pooled)
        combined = exp_data + nov_data
        combined_mean = calculate_mean(combined)
        combined_std = calculate_stddev(combined)
        combined_cv = combined_std / combined_mean if combined_mean != 0 else 0

        cv_results.append((param_name, exp_cv, nov_cv, combined_cv))

        print(f"\n{param_name}:")
        print(f"  Expert CV:   {exp_cv:.4f} ({exp_cv*100:.1f}%)")
        print(f"  Novice CV:   {nov_cv:.4f} ({nov_cv*100:.1f}%)")
        print(f"  Combined CV: {combined_cv:.4f} ({combined_cv*100:.1f}%)")

    # Determine most robust parameter
    most_robust = min(cv_results, key=lambda x: x[3])
    least_robust = max(cv_results, key=lambda x: x[3])

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print(f"  Most robust parameter:  {most_robust[0]} (CV = {most_robust[3]*100:.1f}%)")
    print(f"  Least robust parameter: {least_robust[0]} (CV = {least_robust[3]*100:.1f}%)")
    print("=" * 60)

    return cv_results


if __name__ == "__main__":
    # Summary statistics for all datasets
    datasets = [
        (expert_total, "Experts", "Total Duration"),
        (expert_needle, "Experts", "Needle Passing"),
        (expert_knot, "Experts", "Knot Tying"),
        (novice_total, "Novices", "Total Duration"),
        (novice_needle, "Novices", "Needle Passing"),
        (novice_knot, "Novices", "Knot Tying"),
    ]

    print("=" * 60)
    print("QUESTION 1: Descriptive Statistics")
    print("=" * 60)

    for data, group, task in datasets:
        summary = summary_table(data)
        print_summary_table(summary, group, task)

    # Outlier identification
    print("\n" + "=" * 60)
    print("OUTLIER IDENTIFICATION (IQR Method)")
    print("=" * 60 + "\n")

    for data, group, task in datasets:
        identify_outliers(data, group, task)

    # Robustness analysis
    print("\n")
    analyze_robustness()

    # Histograms and boxplots (12 figures total: 3 params x 2 plot types x 2 groups)
    print("\n" + "=" * 60)
    print("GENERATING 12 FIGURES (3 params x 2 plot types x 2 groups)")
    print("=" * 60)

    generate_all_figures()
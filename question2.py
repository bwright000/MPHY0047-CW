import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import expert_total, expert_needle, expert_knot, novice_total, novice_needle, novice_knot
from scipy import stats
from question1 import calculate_mean, calculate_median, calculate_variance, calculate_stddev, skewness, kurtosis
# QUESTION 2

# Global Variables

ALPHA = 0.05 # Significance level
OUTPUT_DIR = "outputs/"
data = {
    'total': {
        'experts': expert_total,
        'novices': novice_total
    },
    'needle': {
        'experts': expert_needle,
        'novices': novice_needle
    },
    'knot': {
        'experts': expert_knot,
        'novices': novice_knot
    }
}
def calculate_descriptive_stats(data):
    '''
    Calculate descriptive statistics for a list of numbers.
    Returns a dictionary with mean, median, variance, and standard deviation.
    '''

    stats = {
        'mean': calculate_mean(data),
        'median': calculate_median(data),
        'variance': calculate_variance(data),
        'stddev': calculate_stddev(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'skewness': skewness(data),
        'kurtosis': kurtosis(data),
        'iqr': np.percentile(data, 75) - np.percentile(data, 25)
    }
    return stats

def cohens_d(group1, group2):
    '''
    Calculate Cohen's d effect size between two groups.
    Formula: d = (x_bar_1 - x_bar_2) / s_pooled
    where s_pooled = sqrt((s1^2 + s2^2) / 2)
    This is Cohen's (1988, p.67) original unweighted pooled SD.
    Negative d = group 1 has lower values than group 2.
    Ref: Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences, 2nd ed.
    '''
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)  # Sample variance (Bessel-corrected)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt((var1 + var2) / 2)  # Unweighted pooled SD

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d

def interpret_cohens_d(d):
    '''
    Interpret Cohen's d magnitude using conventional thresholds.
    Cohen's (1988) thresholds:
      |d| < 0.2:  Negligible
      0.2 <= |d| < 0.5: Small
      0.5 <= |d| < 0.8: Medium
      |d| >= 0.8: Large
    '''
    if abs(d) < 0.2:
        return "Negligible"
    elif abs(d) < 0.5:
        return "Small"
    elif abs(d) < 0.8:
        return "Medium"
    else:
        return "Large"
 
param_names = {
    'total': 'Total Time',
    'needle': 'Needle Passing Time',
    'knot': 'Knot Tying Time'
}
descriptive_results = {}
for param_key, param_name in param_names.items():
    exp_stats = calculate_descriptive_stats(data[param_key]['experts'])
    nov_stats = calculate_descriptive_stats(data[param_key]['novices'])

    descriptive_results[param_key] = {
        'experts': exp_stats,
        'novices': nov_stats
    }
    # Print Descriptive Statistics
    print(f"Descriptive Statistics for {param_name}")
    print("Experts:", exp_stats)
    print("Novices:", nov_stats)

# Shapiro-Wilk normality test (Shapiro & Wilk, 1965):
#   H0: The data are normally distributed
#   H1: The data are NOT normally distributed
#   Decision: reject H0 if p <= alpha (data non-normal)
#   W statistic close to 1 indicates normality.
print("\n Normality Test - Shapiro-Wilk")

normality_results = {}
for param_key, param_name in param_names.items():
    for group in ['experts', 'novices']:
        dataset_name = f"{group.capitalize()} - {param_name}"
        w_stat, p_val = stats.shapiro(data[param_key][group])
        is_normal = p_val > ALPHA

        normality_results[f"{group}_{param_key}"] = {
            'w': w_stat,
            'p': p_val,
            'normal': is_normal
        }
        status = "Normal" if is_normal else "Not Normal"
        print(f"{dataset_name}: w={w_stat:.4f}, p={p_val:.4f} => {status}")

n_violations = sum(1 for r in normality_results.values() if not r['normal'])

# Distribution shape assessment (supplementary to Shapiro-Wilk):
#   |skewness| > 1 indicates substantial asymmetry (Bulmer, 1979)
#   |excess kurtosis| > 2 indicates substantial tail deviation (George & Mallery, 2010)
#   Note: excess kurtosis = kurtosis - 3 (our kurtosis function returns standard kurtosis)
for param_key, param_name in param_names.items():
    for group in ['experts', 'novices']:
        dataset_name = f"{group.capitalize()} - {param_name}"
        skew_val = descriptive_results[param_key][group]['skewness']
        kurt_val = descriptive_results[param_key][group]['kurtosis']
        concerns = abs(skew_val) > 1 or abs(kurt_val - 3) > 2
        status = "Yes" if concerns else "No"
        print(f"{dataset_name}: Skewness={skew_val:.2f}, Kurtosis={kurt_val:.2f}, Concerns={status}")

# Levene's test for homogeneity of variance:
#   H0: sigma_1^2 = sigma_2^2 (equal variances)
#   H1: sigma_1^2 != sigma_2^2
#   Decision: reject H0 if p <= alpha (unequal variances)
#   SciPy default uses the median variant (Brown-Forsythe), robust to non-normality.
variance_results = {}

for param_key, param_name in param_names.items():
    exp_var = descriptive_results[param_key]['experts']['variance']
    nov_var = descriptive_results[param_key]['novices']['variance']
    ratio = max(exp_var, nov_var) / min(exp_var, nov_var)
    
    lev_stat, lev_p = stats.levene(
        data[param_key]['experts'], 
        data[param_key]['novices']
    )
    
    equal_var = lev_p > ALPHA
    variance_results[param_key] = {
        'exp_var': exp_var,
        'nov_var': nov_var,
        'ratio': ratio,
        'levene_p': lev_p,
        'equal': equal_var
    }
    
    status = "Yes" if equal_var else "No"

# Mann-Whitney U test (non-parametric alternative to t-test, Mann & Whitney, 1947):
#   H0: The distributions of both groups are identical
#   H1: The distributions differ (one group tends to have higher values)
#   Method: Ranks all observations from both groups, compares rank sums.
#   Chosen over t-test because: small samples (n=9, n=11), normality violated in 3/6 datasets.
test_results = {}

for param_key, param_name in param_names.items():
    u_stat, p_val = stats.mannwhitneyu(
        data[param_key]['experts'],
        data[param_key]['novices'],
        alternative='two-sided'
    )

    significant = p_val < ALPHA

    d = cohens_d(data[param_key]['experts'], data[param_key]['novices'])

    test_results[param_key] = {
        'u_stat': u_stat,
        'p_value': p_val,
        'significant': significant,
        'cohens_d': d
    }
    
    status = "Yes (p < 0.05)" if significant else "No"

print("\nFinal Results Table:")
print("-" * 95)
print(f"{'Parameter':<20} {'Expert Mdn (IQR)':<22} {'Novice Mdn (IQR)':<22} {'U':>6} {'p':>8} {'d':>7} {'Sig?':>6}")
print("-" * 95)

for param_key, param_name in param_names.items():
    exp = descriptive_results[param_key]['experts']
    nov = descriptive_results[param_key]['novices']
    res = test_results[param_key]
    
    exp_str = f"{exp['median']:.1f} ({exp['q1']:.1f}-{exp['q3']:.1f})"
    nov_str = f"{nov['median']:.1f} ({nov['q1']:.1f}-{nov['q3']:.1f})"
    sig = "Yes" if res['significant'] else "No"
    
    print(f"{param_name:<20} {exp_str:<22} {nov_str:<22} {res['u_stat']:>6.1f} {res['p_value']:>8.4f} {res['cohens_d']:>7.2f} {sig:>6}")

print("-" * 95)
print("Mdn = Median, IQR = Interquartile Range (Q1-Q3)")

# Generate box plots for each time parameter
import os
os.makedirs('figures', exist_ok=True)

figure_names = {
    'total': 'Total Duration',
    'needle': 'Needle Passing',
    'knot': 'Knot Tying'
}

for param_key, display_name in figure_names.items():
    plt.figure(figsize=(8, 5))
    bp = plt.boxplot(
        [data[param_key]['experts'], data[param_key]['novices']],
        tick_labels=['Experts', 'Novices'],
        patch_artist=True
    )
    bp['boxes'][0].set_facecolor('blue')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('orange')
    bp['boxes'][1].set_alpha(0.7)
    plt.title(f'{display_name} - Experts vs Novices')
    plt.ylabel('Time (seconds)')
    plt.tight_layout()
    fname = f'figures/boxplot_q2_{display_name.lower().replace(" ", "_")}.png'
    plt.savefig(fname, dpi=150)
    plt.show()
    print(f"Saved: {fname}")
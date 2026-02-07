import math
import numpy as np
import pandas as pd
from scipy import stats
from question1 import calculate_mean, calculate_median, calculate_variance, calculate_stddev, skewness, kurtosis
from question2 import calculate_descriptive_stats, cohens_d, interpret_cohens_d, ALPHA

# QUESTION 3

# Global Variables

OUTPUT_DIR = "outputs/"

# Load and parse gesture sequences from Excel
# The spreadsheet has experts in rows 0-8, a blank row, a second header,
# then novices in rows 11-21. Sequences span columns 1-6, terminated by '-'.

df = pd.read_excel('error_data.xlsx')

def extract_sequence(row):
    '''
    Extract integer gesture sequence from one DataFrame row.
    Iterates columns 1-6, stops at '-' or NaN.
    '''
    seq = []
    for col_idx in range(1, 7):
        val = str(row.iloc[col_idx]).strip()
        if val == '-' or val == 'nan':
            break
        seq.append(int(float(val)))
    return seq

expert_sequences = {}
for i in range(0, 9):
    row = df.iloc[i]
    pid = int(row.iloc[0])
    expert_sequences[pid] = extract_sequence(row)

novice_sequences = {}
for i in range(11, 22):
    row = df.iloc[i]
    pid = int(row.iloc[0])
    novice_sequences[pid] = extract_sequence(row)

# Error metric computation
# Ideal sequence: 1-3-4-4-(4)-5
# Rules: S2 present (+1), S3 absent (+1), S5 absent (+1), <2 S4 (+1)

def count_errors(seq):
    '''
    Apply error annotation rules from the coursework specification.
    Ideal sequence: S1-S3-S4-S4-(S4)-S5
    Error rules:
      +1 if S2 (disentangling threads) is present in the sequence
      +1 if S3 (picking appropriate instruments) is absent
      +1 if S5 (suture cutting) is absent
      +1 if fewer than 2 S4 (knot tying) gestures
    Returns (error_count, list_of_reasons).
    '''
    errors = 0
    reasons = []

    if 2 in seq:
        errors += 1
        reasons.append('S2 present')
    if 3 not in seq:
        errors += 1
        reasons.append('S3 absent')
    if 5 not in seq:
        errors += 1
        reasons.append('S5 absent')
    if seq.count(4) < 2:
        errors += 1
        reasons.append('<2 S4')

    return errors, reasons

# Verify against coursework examples
assert count_errors([1,2,4,5]) == (3, ['S2 present', 'S3 absent', '<2 S4'])
assert count_errors([1,3,4,4,4,5]) == (0, [])

expert_errors = []
for pid, seq in expert_sequences.items():
    errs, reasons = count_errors(seq)
    expert_errors.append(errs)
    reason_str = ', '.join(reasons) if reasons else 'none'
    print(f"Expert {pid:>3}: {str(seq):<28} errors = {errs}  ({reason_str})")

novice_errors = []
for pid, seq in novice_sequences.items():
    errs, reasons = count_errors(seq)
    novice_errors.append(errs)
    reason_str = ', '.join(reasons) if reasons else 'none'
    print(f"Novice {pid:>3}: {str(seq):<28} errors = {errs}  ({reason_str})")

# Store as data dict matching Q2 structure
data = {
    'error': {
        'experts': expert_errors,
        'novices': novice_errors
    }
}

# Descriptive Statistics

descriptive_results = {}
exp_stats = calculate_descriptive_stats(data['error']['experts'])
nov_stats = calculate_descriptive_stats(data['error']['novices'])

descriptive_results['error'] = {
    'experts': exp_stats,
    'novices': nov_stats
}

print(f"\nDescriptive Statistics for Error Metric")
print("Experts:", exp_stats)
print("Novices:", nov_stats)

# Same statistical testing pipeline as Q2 - see question2.py for full methodology.
# Shapiro-Wilk: H0 = data is normal, reject if p <= alpha
print("\n Normality Test - Shapiro-Wilk")

normality_results = {}
for group in ['experts', 'novices']:
    dataset_name = f"{group.capitalize()} - Error Metric"
    w_stat, p_val = stats.shapiro(data['error'][group])
    is_normal = p_val > ALPHA

    normality_results[f"{group}_error"] = {
        'w': w_stat,
        'p': p_val,
        'normal': is_normal
    }
    status = "Normal" if is_normal else "Not Normal"
    print(f"{dataset_name}: w={w_stat:.4f}, p={p_val:.4f} => {status}")

n_violations = sum(1 for r in normality_results.values() if not r['normal'])

for group in ['experts', 'novices']:
    dataset_name = f"{group.capitalize()} - Error Metric"
    skew_val = descriptive_results['error'][group]['skewness']
    kurt_val = descriptive_results['error'][group]['kurtosis']
    concerns = abs(skew_val) > 1 or abs(kurt_val - 3) > 2
    status = "Yes" if concerns else "No"
    print(f"{dataset_name}: Skewness={skew_val:.2f}, Kurtosis={kurt_val:.2f}, Concerns={status}")

# Homogeneity of Variance - Levene's Test

variance_results = {}

exp_var = descriptive_results['error']['experts']['variance']
nov_var = descriptive_results['error']['novices']['variance']
ratio = max(exp_var, nov_var) / min(exp_var, nov_var) if min(exp_var, nov_var) > 0 else float('inf')

lev_stat, lev_p = stats.levene(
    data['error']['experts'],
    data['error']['novices']
)

equal_var = lev_p > ALPHA
variance_results['error'] = {
    'exp_var': exp_var,
    'nov_var': nov_var,
    'ratio': ratio,
    'levene_p': lev_p,
    'equal': equal_var
}

status = "Yes" if equal_var else "No"

# Mann-Whitney U Test (see question2.py for full hypothesis definitions)
# Selected because: discrete count data, normality violated, small samples

test_results = {}

u_stat, p_val = stats.mannwhitneyu(
    data['error']['experts'],
    data['error']['novices'],
    alternative='two-sided'
)

significant = p_val < ALPHA
d = cohens_d(data['error']['experts'], data['error']['novices'])

test_results['error'] = {
    'u_stat': u_stat,
    'p_value': p_val,
    'significant': significant,
    'cohens_d': d
}

status = "Yes (p < 0.05)" if significant else "No"

# Final Results Table

print("\nFinal Results Table:")
print("-" * 120)
print(f"{'Parameter':<20} {'Expert Mdn (IQR)':<22} {'Novice Mdn (IQR)':<22} {'U':>6} {'p':>8} {'d':>7} {'Sig?':>6}")
print("-" * 120)

exp = descriptive_results['error']['experts']
nov = descriptive_results['error']['novices']
res = test_results['error']

exp_str = f"{exp['median']:.1f} ({exp['q1']:.1f}-{exp['q3']:.1f})"
nov_str = f"{nov['median']:.1f} ({nov['q1']:.1f}-{nov['q3']:.1f})"
sig = "Yes" if res['significant'] else "No"

print(f"{'Error Metric':<20} {exp_str:<22} {nov_str:<22} {res['u_stat']:>6.1f} {res['p_value']:>8.4f} {res['cohens_d']:>7.2f} {sig:>6}")

print("-" * 120)
print("Mdn = Median, IQR = Interquartile Range (Q1-Q3)")

# Generate box plot for error metric
import matplotlib.pyplot as plt
import os
os.makedirs('figures', exist_ok=True)

plt.figure(figsize=(8, 5))
bp = plt.boxplot(
    [data['error']['experts'], data['error']['novices']],
    tick_labels=['Experts', 'Novices'],
    patch_artist=True
)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('orange')
bp['boxes'][1].set_alpha(0.7)
plt.title('Error Metric - Experts vs Novices')
plt.ylabel('Error Count')
plt.tight_layout()
plt.savefig('figures/boxplot_q3_error_metric.png', dpi=150)
plt.show()
print("Saved: figures/boxplot_q3_error_metric.png")
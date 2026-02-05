# QUESTION 5: Metric Ranking for Discrimination

import pandas as pd

# Results from Questions 2, 3, and 4
metrics_data = {
    'Metric': [
        'Fixation Sparsity',
        'Total Duration',
        'Knot Tying Time',
        'Needle Passing Time',
        'Error Metric'
    ],
    'p_value': [0.0031, 0.0011, 0.0062, 0.0402, 0.7995],
    'cohens_d': [-1.57, -1.56, -1.29, -0.94, 0.10],
    'significant': [True, True, True, True, False]
}

df = pd.DataFrame(metrics_data)

# Calculate absolute Cohen's d for ranking
df['abs_cohens_d'] = df['cohens_d'].abs()

# Determine effect size category
def effect_size_category(d):
    d = abs(d)
    if d < 0.2:
        return 'Negligible'
    elif d < 0.5:
        return 'Small'
    elif d < 0.8:
        return 'Medium'
    else:
        return 'Large'

df['effect_category'] = df['cohens_d'].apply(effect_size_category)

# Rank by: 1) Significance (significant first), 2) Absolute Cohen's d (descending)
df['rank_score'] = df['significant'].astype(int) * 100 + df['abs_cohens_d']
df = df.sort_values('rank_score', ascending=False).reset_index(drop=True)
df['Rank'] = range(1, len(df) + 1)

# Display results
print("=" * 80)
print("QUESTION 5: METRIC RANKING FOR EXPERT-NOVICE DISCRIMINATION")
print("=" * 80)

print("\n### Ranking Criteria ###")
print("1. Statistical significance (p < 0.05)")
print("2. Effect size magnitude (|Cohen's d|)")

print("\n### Results Table ###")
print("-" * 80)
print(f"{'Rank':<6} {'Metric':<22} {'p-value':<10} {'Cohen d':<10} {'Effect':<12} {'Sig?':<6}")
print("-" * 80)

for _, row in df.iterrows():
    sig_str = "Yes" if row['significant'] else "No"
    print(f"{row['Rank']:<6} {row['Metric']:<22} {row['p_value']:<10.4f} {row['cohens_d']:<10.2f} {row['effect_category']:<12} {sig_str:<6}")

print("-" * 80)

print("\n### Ranking Summary ###")
for _, row in df.iterrows():
    print(f"{row['Rank']}. {row['Metric']}")

print("\n### Key Conclusions ###")
print("""
1. BEST DISCRIMINATOR: Fixation Sparsity
   - Highest effect size (|d| = 1.57) captures cognitive differences in visual attention
   - Experts show focused gaze (3.6% sparsity) vs novices' scattered gaze (5.2%)

2. STRONG DISCRIMINATORS: Time-based metrics (Total Duration, Knot Tying, Needle Passing)
   - All show significant differences with large effect sizes
   - Total Duration is most robust (lowest CV = 38.3%)

3. POOREST DISCRIMINATOR: Error Metric
   - Not significant (p = 0.80), negligible effect (d = 0.10)
   - Procedural knowledge is similar; execution quality differs

RECOMMENDATION: Prioritize eye-tracking (fixation sparsity) and time-based metrics
for surgical skill assessment. Error metrics need refinement to capture quality.
""")

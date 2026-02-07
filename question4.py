import numpy as np
from PIL import Image
import os
from scipy import stats
from question1 import calculate_mean, calculate_median, calculate_variance, calculate_stddev, skewness, kurtosis
from question2 import calculate_descriptive_stats, cohens_d, interpret_cohens_d, ALPHA


def calculate_sparsity(image_path):
    '''
    Calculates the fixation sparsity for a single heatmap image.
    Formula: Sparsity = N_nonwhite / (W * H)
    where W = 1920, H = 1080 (image resolution), total = 2,073,600 pixels.
    Non-white pixels (value < 255) indicate locations where fixations were recorded.
    Higher sparsity = more dispersed gaze; lower sparsity = more focused attention.
    '''
    img = Image.open(image_path).convert('L') # Grayscale conversion
    pixels = np.array(img)
    total_pixels = 1920 * 1080
    non_white_count = np.sum(pixels < 255) # Counts non-white pixels
    sparsity = non_white_count / total_pixels
    return sparsity

expert_dir = "fixation_maps/fixation_maps/experts"
novice_dir = "fixation_maps/fixation_maps/novice"

expert_sparsity = []
for filename in os.listdir(expert_dir):
    if filename.endswith('.png'):
        path = os.path.join(expert_dir, filename)
        expert_sparsity.append(calculate_sparsity(path))

novice_sparsity = []
for filename in os.listdir(novice_dir):
    if filename.endswith('.png'):
        path = os.path.join(novice_dir, filename)
        novice_sparsity.append(calculate_sparsity(path))

data = {
    'sparsity': {
    'experts': expert_sparsity,
    'novices': novice_sparsity
    }
}

# Descriptive Statistics
print("\nDescriptive Statistics for Fixation Sparsity")
exp_stats = calculate_descriptive_stats(data['sparsity']['experts'])
nov_stats = calculate_descriptive_stats(data['sparsity']['novices'])
print("Experts:", exp_stats)
print("Novices:", nov_stats)

# Normality Test - Shapiro-Wilk
print("\nNormality Test - Shapiro-Wilk")
exp_w, exp_p = stats.shapiro(data['sparsity']['experts'])
nov_w, nov_p = stats.shapiro(data['sparsity']['novices'])

exp_normal = exp_p > ALPHA
nov_normal = nov_p > ALPHA

print(f"Experts: w={exp_w:.4f}, p={exp_p:.4f} => {'Normal' if exp_normal else 'Not Normal'}")
print(f"Novices: w={nov_w:.4f}, p={nov_p:.4f} => {'Normal' if nov_normal else 'Not Normal'}")

# Homogeneity of Variance - Levene's Test
lev_stat, lev_p = stats.levene(data['sparsity']['experts'], data['sparsity']['novices'])
equal_var = lev_p > ALPHA
print(f"\nLevene's Test: statistic={lev_stat:.4f}, p={lev_p:.4f} => {'Equal variance' if equal_var else 'Unequal variance'}")

# Data-driven test selection decision tree:
#   Step 1: Shapiro-Wilk -> both groups normal?
#     Yes -> Step 2: Levene's test -> equal variances?
#       Yes -> Independent t-test (most powerful parametric test)
#       No  -> Welch's t-test (does not assume equal variances)
#     No  -> Mann-Whitney U (non-parametric, no normality assumption)
# See question2.py for full hypothesis definitions of each test.
print("\nStatistical Test Selection:")
if exp_normal and nov_normal:
    if equal_var:
        # Both normal, equal variances -> Independent t-test
        test_name = "Independent t-test"
        t_stat, p_val = stats.ttest_ind(data['sparsity']['experts'], data['sparsity']['novices'])
        print(f"Both groups normal, equal variances -> {test_name}")
        print(f"{test_name}: t={t_stat:.4f}, p={p_val:.4f}")
    else:
        # Both normal, unequal variances -> Welch's t-test
        test_name = "Welch's t-test"
        t_stat, p_val = stats.ttest_ind(data['sparsity']['experts'], data['sparsity']['novices'], equal_var=False)
        print(f"Both groups normal, unequal variances -> {test_name}")
        print(f"{test_name}: t={t_stat:.4f}, p={p_val:.4f}")
else:
    # Non-normal -> Mann-Whitney U
    test_name = "Mann-Whitney U"
    u_stat, p_val = stats.mannwhitneyu(data['sparsity']['experts'], data['sparsity']['novices'], alternative='two-sided')
    print(f"Normality violated -> {test_name}")
    print(f"{test_name}: U={u_stat:.4f}, p={p_val:.4f}")

# Effect Size
d = cohens_d(data['sparsity']['experts'], data['sparsity']['novices'])
significant = p_val < ALPHA

print(f"\nCohen's d: {d:.4f} ({interpret_cohens_d(d)})")
print(f"Significant at alpha={ALPHA}: {'Yes' if significant else 'No'}")

# Final Results Table
print("\n" + "=" * 100)
print("FINAL RESULTS TABLE")
print("=" * 100)
print(f"{'Metric':<20} {'Expert Median (IQR)':<25} {'Novice Median (IQR)':<25} {'p-value':>10} {'Cohen d':>10} {'Sig?':>6}")
print("-" * 100)

exp_str = f"{exp_stats['median']:.4f} ({exp_stats['q1']:.4f}-{exp_stats['q3']:.4f})"
nov_str = f"{nov_stats['median']:.4f} ({nov_stats['q1']:.4f}-{nov_stats['q3']:.4f})"
sig_str = "Yes" if significant else "No"

print(f"{'Fixation Sparsity':<20} {exp_str:<25} {nov_str:<25} {p_val:>10.4f} {d:>10.4f} {sig_str:>6}")
print("-" * 100)

# Generate box plot for fixation sparsity
import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)

plt.figure(figsize=(8, 5))
bp = plt.boxplot(
    [data['sparsity']['experts'], data['sparsity']['novices']],
    tick_labels=['Experts', 'Novices'],
    patch_artist=True
)
bp['boxes'][0].set_facecolor('blue')
bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor('orange')
bp['boxes'][1].set_alpha(0.7)
plt.title('Fixation Sparsity - Experts vs Novices')
plt.ylabel('Sparsity (ratio)')
plt.tight_layout()
plt.savefig('figures/boxplot_q4_fixation_sparsity.png', dpi=150)
plt.show()
print("Saved: figures/boxplot_q4_fixation_sparsity.png")
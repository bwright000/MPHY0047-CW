# Question 2: Image Similarity Metrics
# Computes SSI, MI, and CS between test images and gold standards,
# identifies top participants per view, and performs expert vs novice statistical testing.

from dataloader import (test_img, gold_img, NUM_PARTICIPANTS, NUM_VIEWS,
                        EXPERT_RANGE, NOVICE_RANGE, MISSING)
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score
from scipy import stats
import numpy as np

VIEW_NAMES = [f"View {i+1}" for i in range(NUM_VIEWS)]


def is_missing(p, v):
    """Check if participant p, view v is a missing data entry."""
    return (p, v) in MISSING


def compute_similarity_metrics():
    """
    Compute SSI, MI, and CS for each test image against its gold standard.

    SSI (Structural Similarity Index): Compares luminance, contrast, and structure
        between two images. Computed via skimage.metrics.structural_similarity.

    MI (Mutual Information): Entropy-based metric measuring shared information
        between two images. MI = E_a + E_b - E_ab.
        Computed via sklearn.metrics.mutual_info_score on flattened pixel arrays.

    CS (Cosine Similarity): Measures the cosine of the angle between two image
        vectors (flattened). CS = dot(a, b) / (||a|| * ||b||).

    Returns:
        ssi_vals, mi_vals, cs_vals: each a 20x10 numpy array (NaN for missing entries)
    """
    ssi_vals = np.full((NUM_PARTICIPANTS, NUM_VIEWS), np.nan)
    mi_vals = np.full((NUM_PARTICIPANTS, NUM_VIEWS), np.nan)
    cs_vals = np.full((NUM_PARTICIPANTS, NUM_VIEWS), np.nan)

    for v in range(NUM_VIEWS):
        # Gold standard image for this view (squeeze to 2D if needed - Just to remove any additional dimensions (If applicable))
        gold = np.squeeze(gold_img[0][v]).astype(np.float64)

        for p in range(NUM_PARTICIPANTS):
            if is_missing(p, v):
                continue

            # Test image for this participant and view
            test = np.squeeze(test_img[p][v]).astype(np.float64)

            # SSI: structural similarity using full image data range
            data_range = max(gold.max(), test.max()) - min(gold.min(), test.min())
            ssi_vals[p][v] = ssim(gold, test, data_range=data_range)

            # MI: mutual information from flattened pixel intensity arrays
            mi_vals[p][v] = mutual_info_score(gold.flatten().astype(int),
                                              test.flatten().astype(int))

            # CS: cosine similarity between flattened image vectors
            a = test.flatten()
            b = gold.flatten()
            cs_vals[p][v] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    return ssi_vals, mi_vals, cs_vals


def get_top3_participants(metric_vals, view_idx):
    """
    Return the top 3 participant indices (0-indexed) for a given view,
    ranked by descending metric value. Skips NaN entries.
    """
    scores = metric_vals[:, view_idx]
    # Get indices of valid (non-NaN) entries, sorted by score descending
    valid_idx = np.where(~np.isnan(scores))[0]
    ranked = valid_idx[np.argsort(scores[valid_idx])[::-1]]
    return ranked[:3]


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

    # Compute all similarity metrics.
    ssi_vals, mi_vals, cs_vals = compute_similarity_metrics()

    # Part i: Report top 3 participants per view per metric.
    print("PART i: Top 3 Participants per View by Similarity Metric")

    metric_names = ["SSI", "MI", "CS"]
    metric_arrays = [ssi_vals, mi_vals, cs_vals]

    for m_name, m_vals in zip(metric_names, metric_arrays):
        print(f"\n{m_name}")
        print(f"{'View':<10} {'#1':<20} {'#2':<20} {'#3':<20}")
        for v in range(NUM_VIEWS):
            top3 = get_top3_participants(m_vals, v)
            # Report as 1-indexed participant numbers with their metric values
            entries = []
            for idx in top3:
                entries.append(f"P{idx+1} ({m_vals[idx][v]:.4f})")
            print(f"{VIEW_NAMES[v]:<10} {entries[0]:<20} {entries[1]:<20} {entries[2]:<20}")

    # Print full metric tables for reference.
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
    print("\nH0: No difference in similarity metric between expert and novice groups")
    print("H1: Expert and novice groups differ in similarity metric")
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

# Question 1: Kinematic Feature Extraction (15 marks)
# Computes 12 metrics (22 parameters total) from the JIGSAWS suturing dataset.
# Each parameter is an (8, 5) array - 8 participants x 5 trials.
# Slave left (PSM1) and slave right (PSM2) computed independently.
# Exports feature matrix to q1_features.npz for Q2.

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

from dataloader import (
    suturing, PARTICIPANTS,
    EXPERT_INDICES, NOVICE_INDICES,
    NUM_PARTICIPANTS, NUM_TRIALS, DT,
    SL_XYZ, SL_TVEL, SL_RVEL,
    SR_XYZ, SR_TVEL, SR_RVEL,
    get_label,
)
from plot_style import apply_style, BLUE, ORANGE, finish_figure

apply_style()

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def time_of_completion(trial):
    """
    Time of completion.
        T = N * dt   (seconds)
    where N = number of samples, dt = 1/30 s.
    """
    return trial.shape[0] * DT


def total_path_length(xyz):
    """
    Total path length (Euclidean).
        PL = sum_{i=1}^{N-1} ||p(i+1) - p(i)||_2
    """
    diffs = np.diff(xyz, axis=0)
    return np.sum(np.linalg.norm(diffs, axis=1))


def economy_of_area(xyz, plane, path_length):
    """
    Economy of area in a 2D plane.
        EA = sqrt((max(a) - min(a)) * (max(b) - min(b))) / PL

    Parameters
    ----------
    xyz : (N, 3) array of positions
    plane : str, one of 'xy', 'xz', 'yz'
    path_length : float, precomputed total path length
    """
    idx = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}[plane]
    a, b = xyz[:, idx[0]], xyz[:, idx[1]]
    area = (a.max() - a.min()) * (b.max() - b.min())
    return np.sqrt(area) / path_length


def economy_of_volume(xyz, path_length):
    """
    Economy of volume.
        EV = cbrt((max(x)-min(x)) * (max(y)-min(y)) * (max(z)-min(z))) / PL
    """
    ranges = xyz.max(axis=0) - xyz.min(axis=0)
    return np.cbrt(np.prod(ranges)) / path_length


def avg_linear_velocity(vel):
    """
    Average linear velocity.
        LV = (1/N) * sum ||v(i)||_2
    Uses the translational velocity columns provided in the dataset.
    """
    return np.mean(np.linalg.norm(vel, axis=1))


def avg_linear_acceleration(vel):
    """
    Average linear acceleration via finite differences of velocity.
        a(i) = (v(i+1) - v(i)) / dt
        LA = (1/(N-1)) * sum ||a(i)||_2

    Uses unitary spacing (step=1) as specified, so a(i) = diff(v, axis=0).
    The dt factor cancels when computing the norm per-sample.
    Note: we use step=1 (not dividing by dt) as instructed.
    """
    acc = np.diff(vel, axis=0)
    return np.mean(np.linalg.norm(acc, axis=1))


def avg_rotational_velocity(rvel):
    """
    Average rotational velocity.
        RV = (1/N) * sum ||omega(i)||_2
    Uses the rotational velocity columns provided in the dataset.
    """
    return np.mean(np.linalg.norm(rvel, axis=1))


def avg_rotational_acceleration(rvel):
    """
    Average rotational acceleration via finite differences.
        alpha(i) = (omega(i+1) - omega(i)) / dt
        RA = (1/(N-1)) * sum ||alpha(i)||_2

    Uses unitary spacing (step=1) as specified.
    """
    alpha = np.diff(rvel, axis=0)
    return np.mean(np.linalg.norm(alpha, axis=1))


def motion_smoothness(xyz, vel, path_length):
    """
    Normalised jerk metric (motion smoothness).
        MS = sqrt(T^5 / (2 * PL^2)) * sum ||j(i)||^2 * dt

    where:
        a(i) = diff(v, axis=0)          (acceleration, finite difference step=1)
        j(i) = diff(a, axis=0)          (jerk, finite difference step=1)
        T    = N * dt                   (total time)
        PL   = total path length

    Lower values indicate smoother motion.
    """
    N = xyz.shape[0]
    T = N * DT

    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)

    jerk_sq_sum = np.sum(np.sum(jerk ** 2, axis=1)) * DT
    normalisation = np.sqrt(T ** 5 / (2.0 * path_length ** 2))

    return normalisation * jerk_sq_sum


def avg_distance_between_manipulators(xyz_left, xyz_right):
    """
    Average Euclidean distance between left and right slave manipulators.
        D = (1/N) * sum ||p_left(i) - p_right(i)||_2
    """
    return np.mean(np.linalg.norm(xyz_left - xyz_right, axis=1))


METRIC_NAMES = [
    'Time of Completion (s)',
    'Path Length Left', 'Path Length Right',
    'EA_xy Left', 'EA_xy Right',
    'EA_xz Left', 'EA_xz Right',
    'EA_yz Left', 'EA_yz Right',
    'EV Left', 'EV Right',
    'Avg Lin Vel Left', 'Avg Lin Vel Right',
    'Avg Lin Acc Left', 'Avg Lin Acc Right',
    'Avg Rot Vel Left', 'Avg Rot Vel Right',
    'Avg Rot Acc Left', 'Avg Rot Acc Right',
    'Motion Smooth Left', 'Motion Smooth Right',
    'Avg Dist Between Manip',
]

NUM_PARAMS = len(METRIC_NAMES)  # 22


def compute_all_features():
    """
    Compute all 22 kinematic parameters for every suturing trial.

    Returns
    -------
    features : dict mapping metric name -> (8, 5) ndarray
    """
    features = {name: np.zeros((NUM_PARTICIPANTS, NUM_TRIALS)) for name in METRIC_NAMES}

    for p in range(NUM_PARTICIPANTS):
        for t in range(NUM_TRIALS):
            trial = suturing[p, t]

            sl_xyz  = trial[:, SL_XYZ]
            sl_vel  = trial[:, SL_TVEL]
            sl_rvel = trial[:, SL_RVEL]
            sr_xyz  = trial[:, SR_XYZ]
            sr_vel  = trial[:, SR_TVEL]
            sr_rvel = trial[:, SR_RVEL]

            # 1. Time of completion (shared)
            features['Time of Completion (s)'][p, t] = time_of_completion(trial)

            # 2. Path length (per manipulator)
            pl_l = total_path_length(sl_xyz)
            pl_r = total_path_length(sr_xyz)
            features['Path Length Left'][p, t] = pl_l
            features['Path Length Right'][p, t] = pl_r

            # 3-5. Economy of area (per plane, per manipulator)
            for plane in ['xy', 'xz', 'yz']:
                key_l = f'EA_{plane} Left'
                key_r = f'EA_{plane} Right'
                features[key_l][p, t] = economy_of_area(sl_xyz, plane, pl_l)
                features[key_r][p, t] = economy_of_area(sr_xyz, plane, pl_r)

            # 6. Economy of volume
            features['EV Left'][p, t] = economy_of_volume(sl_xyz, pl_l)
            features['EV Right'][p, t] = economy_of_volume(sr_xyz, pl_r)

            # 7. Avg linear velocity
            features['Avg Lin Vel Left'][p, t] = avg_linear_velocity(sl_vel)
            features['Avg Lin Vel Right'][p, t] = avg_linear_velocity(sr_vel)

            # 8. Avg linear acceleration
            features['Avg Lin Acc Left'][p, t] = avg_linear_acceleration(sl_vel)
            features['Avg Lin Acc Right'][p, t] = avg_linear_acceleration(sr_vel)

            # 9. Avg rotational velocity
            features['Avg Rot Vel Left'][p, t] = avg_rotational_velocity(sl_rvel)
            features['Avg Rot Vel Right'][p, t] = avg_rotational_velocity(sr_rvel)

            # 10. Avg rotational acceleration
            features['Avg Rot Acc Left'][p, t] = avg_rotational_acceleration(sl_rvel)
            features['Avg Rot Acc Right'][p, t] = avg_rotational_acceleration(sr_rvel)

            # 11. Motion smoothness
            features['Motion Smooth Left'][p, t] = motion_smoothness(sl_xyz, sl_vel, pl_l)
            features['Motion Smooth Right'][p, t] = motion_smoothness(sr_xyz, sr_vel, pl_r)

            # 12. Avg distance between manipulators (shared)
            features['Avg Dist Between Manip'][p, t] = avg_distance_between_manipulators(sl_xyz, sr_xyz)

    return features


def print_table(name, data):
    """Print an (8, 5) array as a formatted table with participant labels."""
    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    header = f"  {'Participant':<14}" + "".join(f"{'Trial ' + str(t+1):>12}" for t in range(NUM_TRIALS))
    print(header)
    print(f"  {'-' * 14}" + "-" * 60)
    for p in range(NUM_PARTICIPANTS):
        label = f"{PARTICIPANTS[p]} ({'E' if p in EXPERT_INDICES else 'N'})"
        row = f"  {label:<14}" + "".join(f"{data[p, t]:>12.4f}" for t in range(NUM_TRIALS))
        print(row)


def run_mann_whitney(features):
    """
    Mann-Whitney U test for each parameter: experts vs novices.
    Returns list of (name, U, p_value, significant) tuples.
    """
    ALPHA = 0.05
    results = []
    for name in METRIC_NAMES:
        data = features[name]
        expert_vals = data[EXPERT_INDICES, :].flatten()
        novice_vals = data[NOVICE_INDICES, :].flatten()
        u_stat, p_val = stats.mannwhitneyu(expert_vals, novice_vals, alternative='two-sided')
        results.append((name, u_stat, p_val, p_val < ALPHA))
    return results


def print_mann_whitney(results):
    """Print Mann-Whitney U test results as a formatted table."""
    print(f"\n{'=' * 80}")
    print("  Mann-Whitney U: Expert vs Novice (alpha = 0.05)")
    print(f"{'=' * 80}")
    print(f"  {'Parameter':<28} {'U':>8} {'p-value':>12} {'Significant?':>14}")
    print(f"  {'-' * 28} {'-' * 8} {'-' * 12} {'-' * 14}")
    for name, u, p, sig in results:
        marker = "***" if sig else ""
        print(f"  {name:<28} {u:>8.1f} {p:>12.6f} {marker:>14}")


def plot_boxplots(features):
    """Create side-by-side box plots for experts vs novices, 4 per figure."""
    names = list(METRIC_NAMES)
    n_plots = len(names)
    plots_per_fig = 4
    n_figs = (n_plots + plots_per_fig - 1) // plots_per_fig

    for fig_idx in range(n_figs):
        start = fig_idx * plots_per_fig
        end = min(start + plots_per_fig, n_plots)
        batch = names[start:end]

        fig, axes = plt.subplots(1, len(batch), figsize=(4 * len(batch), 5))
        if len(batch) == 1:
            axes = [axes]

        for ax, name in zip(axes, batch):
            data = features[name]
            expert_vals = data[EXPERT_INDICES, :].flatten()
            novice_vals = data[NOVICE_INDICES, :].flatten()

            bp = ax.boxplot(
                [expert_vals, novice_vals],
                tick_labels=['Expert', 'Novice'],
                patch_artist=True,
                widths=0.5,
            )
            bp['boxes'][0].set_facecolor(BLUE)
            bp['boxes'][1].set_facecolor(ORANGE)
            for box in bp['boxes']:
                box.set_alpha(0.7)

            ax.set_title(name, fontsize=10)
            ax.set_ylabel('Value')

        fig.suptitle('Expert vs Novice Comparison', fontsize=13, y=1.02)
        path = os.path.join(FIGURES_DIR, f'q1_boxplots_{fig_idx + 1}.png')
        finish_figure(fig, path)
        print(f"  Saved: {path}")


def save_feature_matrix(features):
    """
    Build (40, 22) feature matrix and (40,) label vector, save to .npz.
    Row order: participants 0-7, trials 0-4 (row = p * 5 + t).
    """
    X = np.zeros((NUM_PARTICIPANTS * NUM_TRIALS, NUM_PARAMS))
    y = np.zeros(NUM_PARTICIPANTS * NUM_TRIALS, dtype=int)
    trial_ids = np.zeros(NUM_PARTICIPANTS * NUM_TRIALS, dtype=int)

    for p in range(NUM_PARTICIPANTS):
        for t in range(NUM_TRIALS):
            row = p * NUM_TRIALS + t
            for j, name in enumerate(METRIC_NAMES):
                X[row, j] = features[name][p, t]
            y[row] = get_label(p)
            trial_ids[row] = t

    path = os.path.join(_DIR, 'q1_features.npz')
    np.savez(path, X=X, y=y, trial_ids=trial_ids, feature_names=METRIC_NAMES)
    print(f"\n  Feature matrix saved: {path}")
    print(f"  X shape: {X.shape}, y shape: {y.shape}")
    return X, y, trial_ids


# Compute on import (available to Q2 and other modules)

features = compute_all_features()
mw_results = run_mann_whitney(features)
sig_params = [name for name, _, _, sig in mw_results if sig]


# Main

if __name__ == '__main__':
    print("Computing kinematic features for suturing task...")

    # Print 8x5 tables
    for name in METRIC_NAMES:
        print_table(name, features[name])

    # Mann-Whitney U results
    print_mann_whitney(mw_results)

    print(f"\n  Parameters that significantly distinguish experts from novices:")
    for name in sig_params:
        print(f"    - {name}")
    if not sig_params:
        print("    (none at alpha = 0.05)")

    # Box plots
    print("\nGenerating box plots...")
    plot_boxplots(features)

    # Save for Q2
    save_feature_matrix(features)

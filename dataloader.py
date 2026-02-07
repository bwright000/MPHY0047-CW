# Data loader for MPHY0047 Coursework 1.
# Loads time parameters (in seconds) from CSV files for expert (n=9) and novice (n=11) groups.
# Three parameters per participant:
#   - total time: duration of entire suturing task (needle passing to suture cutting)
#   - needle passing time: duration of needle passing subtask
#   - knot tying time: duration of first knot tying subtask

import math
import pandas as pd
import matplotlib.pyplot as plt


experts_df = pd.read_csv('time_experts.csv')
novices_df = pd.read_csv('time_novices.csv')

# Extract the 3 time parameters as lists

expert_total = experts_df['total time'].tolist()
expert_needle = experts_df['needle passing time'].tolist()
expert_knot = experts_df['knot tying time'].tolist()

novice_total = novices_df['total time'].tolist()
novice_needle = novices_df['needle passing time'].tolist()
novice_knot = novices_df['knot tying time'].tolist()
def tests():
    print("Data loaded successfully.")
    print(f"Number of expert samples: {len(expert_total)}")
    print(f"Number of novice samples: {len(novice_total)}")
    print(f"Expert total times (first 5): {expert_total[:5]}")
    print(f"Novice total times (first 5): {novice_total[:5]}")

# tests()

# Data loader for MPHY0047 Coursework 1.
# Loads time parameters (in seconds) from CSV files for expert (n=9) and novice (n=11) groups.
# Three parameters per participant:
#   - total time: duration of entire suturing task (needle passing to suture cutting)
#   - needle passing time: duration of needle passing subtask
#   - knot tying time: duration of first knot tying subtask

import math
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

# Load cw2.mat
cw2_data = sio.loadmat('CW2/Provided/cw2.mat')
# Extract arrays
test_img = cw2_data['test_img']
gold_img = cw2_data['gold_img']
gen_impr = cw2_data['gen_impr']
crit_perc = cw2_data['crit_perc']

# GLOBAL VARIABLES
NUM_PARTICIPANTS = 20
NUM_VIEWS = 10
EXPERT_RANGE = (0, 7)
NOVICE_RANGE = (7, 20)
MISSING = [(8,9), (12,7), (13,9), (14,0), (15,3)]

def get_valid_scores(view_idx):
    """
    Return (gen_impr_list, crit_perc_list) for a view, excluding missing entries.
    """
    gi_scores = []
    cp_scores = []
    for participant in range(NUM_PARTICIPANTS):
        if gen_impr[participant][view_idx] != -1:
            gi_scores.append(gen_impr[participant][view_idx])
            cp_scores.append(crit_perc[participant][view_idx])
    return gi_scores, cp_scores
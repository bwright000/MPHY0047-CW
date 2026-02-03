import math
import pandas as pd
import matplotlib.pyplot as plt


experts_df = pd.read_csv('experts_data.csv')
novices_df = pd.read_csv('novices_data.csv')

# Extreact the 3 time relevant parameters as lists

expert_total = experts_df['total time'].tolist()
expert_needle = experts_df['needle passing time'].tolist()
expert_knot = experts_df['knot tying time'].tolist()

novice_total = noovices_df['total time'].tolist()
novice_needle = noovices_df['needle passing time'].tolist()
novice_knot = noovices_df['knot tying time'].tolist()



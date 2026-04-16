# Data loader for MPHY0047 Coursework 4.
# Loads ECG preprocessed features (Q1-Q3), raw ECG signal (Q4),
# and heart-statlog CVD data (Q5).

import os
import numpy as np
import pandas as pd

# Resolve paths relative to this file
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROVIDED = os.path.join(_DIR, 'Provided')

# --- ECG Preprocessed Data (Q1-Q3) ---
# 12730 heartbeats, 83 wavelet features, 5 classes (0-4).
# Spec requires 1800 heartbeats from 3 AAMI classes (N=0, S=1, V=2).
# Filter to classes 0, 1, 2 and sample 600 per class.
_ecg_full = pd.read_csv(os.path.join(_PROVIDED, 'ecg_signals.preprocessed.csv'))
_ecg_filtered = _ecg_full[_ecg_full['classes'].isin([0, 1, 2])]
_ecg_sampled = (_ecg_filtered.groupby('classes')
                .sample(n=600, random_state=42)
                .reset_index(drop=True))

X_ecg = _ecg_sampled.iloc[:, :-1].values      # (1800, 83)
y_ecg = _ecg_sampled.iloc[:, -1].values        # (1800,) classes 0, 1, 2
ECG_FEATURE_NAMES = list(_ecg_sampled.columns[:-1])

CLASS_NAMES = ['N', 'S', 'V']
CLASS_MAP = {0: 'N', 1: 'S', 2: 'V'}
NUM_CLASSES = 3

# --- Raw ECG Signal (Q4) ---
_ecg_signal_df = pd.read_csv(
    os.path.join(_PROVIDED, 'single_ecg_signal.csv'), header=None
)
ecg_signal = _ecg_signal_df.iloc[:, 0].values   # (3000,)
ECG_SAMPLE_RATE = 360  # Hz

# --- Heart-Statlog CVD Data (Q5) ---
heart_data = pd.read_csv(os.path.join(_PROVIDED, 'heart-statlog.csv'))
HEART_FEATURE_NAMES = list(heart_data.columns[:-1])
HEART_CLASS_COL = 'class'

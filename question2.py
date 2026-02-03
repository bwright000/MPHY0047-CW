import math
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import expert_total, expert_needle, expert_knot, novice_total, novice_needle, novice_knot
from scipy import stats
from qusetion1 import compute_mean, calculate_median, calculate_variance, calculate_stddev, skewness, kurtosis
# QUESTION 2

# Global Variables

ALPHA = 0.05 # Significance level
OUTPUT_DIR = "outputs/"

def calculate_descriptive_stats(data):
    '''
    Calculate descriptive statistics for a list of numbers.
    Returns a dictionary with mean, median, variance, and standard deviation.
    '''

    stats = {
        'mean': compute_mean(data),
        'median': calculate_median(data),
        'variance': calculate_variance(data),
        'stddev': calculate_stddev(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        skewness: skewness(data),
        kurtosis: kurtosis(data)
    }
    return stats

def cohens_d(group1, group2):
    '''
    Calculate Cohen's d effect size between two groups.
    '''
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt((var1 + var2) / 2)

    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d
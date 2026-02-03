import math
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import expert_total, expert_needle, expert_knot, novice_total, novice_needle, novice_knot


# QUESTION 1

def compute_mean(data):
    '''
    Compute the mean of a list of numbers.
    '''

    n = len(data)
    total = 0
    for x in data:
        total += x
    return total / n

def calculate_median(data):
    '''
    Compute the median of a list of numbers.
    '''

    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2

    if n % 2 == 0:
        median = (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        median = sorted_data[mid]
    
    return median

def calculate_variance(data, population=True):
    '''
    Compute the variance of a list of numbers.
    If population is True, compute population variance.
    If population is False, compute sample variance.
    '''

    n = len(data)
    mean = compute_mean(data)
    variance_sum = 0

    for x in data:
        variance_sum += (x - mean) ** 2

    if population:
        variance = variance_sum / n
    else:
        variance = variance_sum / (n - 1)

    return variance

def calculate_stddev(data, population=True):
    '''
    Compute the standard deviation of a list of numbers.
    If population is True, compute population standard deviation.
    If population is False, compute sample standard deviation.
    '''

    variance = calculate_variance(data, population)
    stddev = math.sqrt(variance)
    return stddev

def skewness(data, population=True):
    '''
    Compute the skewness of a list of numbers.
    If Population is tru, compute population skewness.
    If Population is false, compute sample skewness.
    '''

    n = len(data)
    mean = calculate_mean(data)
    std = calculate_stddev(data, population=True)

    skew = 0
    for x in data:
        skew += ((x - mean)/std)**3
    return skew / n


def kurtosis(data, population=True):
    '''
    Compute the kurtosis of a list of numbers.
    '''
    n = len(data)
    mean = calculate_mean(data)
    std = calculate_stddev(data, population=True)
    
    kurt = 0
    for x in data:
        kurt += ((x -mean)/std)**4
    return kurt / n


def summary_table(data, population=True):
    '''
    Generate a summary table of statistics for a list of numbers.
    '''

    mean = compute_mean(data)
    median = calculate_median(data)
    variance = calculate_variance(data, population)
    stddev = calculate_stddev(data, population)
    skew = skewness(data, population)
    kurt = kurtosis(data, population)

    summary = {
        'Mean': mean,
        'Median': median,
        'Variance': variance,
        'Standard Deviation': stddev,
        'Skewness': skew,
        'Kurtosis': kurt
    }

    return summary


def print_summary_table(summary, group_name, task_name):
    '''
    Print the summary table in a formatted way.
    '''

    print(f"Summary Statistics for {group_name} - {task_name}:")
    print("=" * 50)
    for stat, value in summary.items():
        print(f"{stat:25}: {value:.4f}")
    print("\n")

def plots_histograms_and_boxplots(data_expert, data_novice, task_name):
    '''
    Plot histograms and boxplots for expert and novice data.
    '''

    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(data_expert, bins=10, alpha=0.5, label='Experts', color='blue')
    plt.hist(data_novice, bins=10, alpha=0.5, label='Novices', color='orange')
    plt.title(f'Histogram of {task_name} Time')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.legend()

    # Boxplot
    plt.subplot(1, 2, 2)
    plt.boxplot([data_expert, data_novice], labels=['Experts', 'Novices'])
    plt.title(f'Boxplot of {task_name} Time')
    plt.ylabel('Time')

    plt.tight_layout()
    plt.show()
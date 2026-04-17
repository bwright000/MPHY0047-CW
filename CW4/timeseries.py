# Shared time-series utilities for Q4.
# ADF stationarity testing, ACF/PACF plotting, forecast metrics.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from plot_style import finish_figure, BLUE, RED, GREEN, ORANGE

_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def adf_summary(series, name):
    """Run Augmented Dickey-Fuller on a series. Returns dict of ADF stat, p-value, lags."""
    x = np.asarray(series)
    x = x[~np.isnan(x)]
    stat, pvalue, used_lag, nobs, crit_vals, _ = adfuller(x, autolag='AIC')
    return {
        'name': name,
        'adf_stat': stat,
        'p_value': pvalue,
        'used_lag': used_lag,
        'nobs': nobs,
        'crit_1pct': crit_vals['1%'],
        'crit_5pct': crit_vals['5%'],
        'stationary_5pct': pvalue < 0.05,
    }


def print_adf_table(rows):
    """Pretty-print a list of adf_summary dicts as a comparison table."""
    print(f"{'Transform':<20} {'ADF stat':>12} {'p-value':>12} {'Crit 5%':>12} {'Stationary?':>14}")
    print("-" * 72)
    for r in rows:
        flag = 'YES' if r['stationary_5pct'] else 'no'
        print(f"{r['name']:<20} {r['adf_stat']:>12.4f} {r['p_value']:>12.4g} "
              f"{r['crit_5pct']:>12.4f} {flag:>14}")


def select_best_transform(rows):
    """
    Select transform with strongest stationarity.
    Rank by both ADF statistic (more negative = better) AND p-value (lower = better).
    Lessons_learned Issue 4: never rank on one metric alone.
    """
    # Only consider transforms that pass the 5% test
    passing = [r for r in rows if r['stationary_5pct']]
    candidates = passing if passing else rows
    # Among passing, pick most-negative ADF stat (strongest rejection of unit root)
    return min(candidates, key=lambda r: r['adf_stat'])


def plot_transforms(raw, transforms, filename):
    """Plot raw signal + each transform in a 2x2 grid so the effect is visible."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False)
    panels = [('Raw', raw, BLUE)] + [(n, s, c) for (n, s), c in zip(transforms, [GREEN, ORANGE, RED])]
    for ax, (name, signal, colour) in zip(axes.flat, panels):
        ax.plot(signal, color=colour, linewidth=0.7)
        ax.set_title(name)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Amplitude')
    fig.suptitle('ECG Signal and Stationarising Transforms')
    finish_figure(fig, os.path.join(FIGURES_DIR, filename))


def plot_adf_bar(rows, filename):
    """Bar chart comparing ADF statistics across transforms."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = [r['name'] for r in rows]
    stats = [r['adf_stat'] for r in rows]
    colours = [GREEN if r['stationary_5pct'] else RED for r in rows]
    ax.bar(names, stats, color=colours, edgecolor='black', linewidth=0.6)
    ax.axhline(y=rows[0]['crit_5pct'], color='gray', linestyle='--',
               linewidth=1.0, label='5% critical value')
    ax.set_ylabel('ADF Statistic (more negative = more stationary)')
    ax.set_title('ADF Test Across Transformations')
    ax.legend()
    finish_figure(fig, os.path.join(FIGURES_DIR, filename))


def plot_acf_pacf(series, nlags, filename):
    """Plot ACF and PACF side by side with 95% CI bands."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(series, lags=nlags, ax=axes[0])
    axes[0].set_title('ACF — Autocorrelation Function')
    axes[0].set_xlabel('Lag')
    plot_pacf(series, lags=nlags, ax=axes[1], method='ywm')
    axes[1].set_title('PACF — Partial Autocorrelation Function')
    axes[1].set_xlabel('Lag')
    finish_figure(fig, os.path.join(FIGURES_DIR, filename))


def find_seasonal_period(series, min_lag=30, max_lag=500):
    """
    Estimate the seasonal period as the lag (>min_lag) of the first maximum
    of the autocorrelation function. ECG at 360 Hz with ~1 beat/sec should
    yield a period around 300 samples.
    """
    from statsmodels.tsa.stattools import acf
    x = np.asarray(series)
    nlags = min(max_lag, len(x) // 2 - 1)
    acf_vals = acf(x, nlags=nlags, fft=True)
    # Search for the argmax strictly beyond min_lag, ignoring the lag-0 spike
    search = acf_vals[min_lag:]
    peak = int(np.argmax(search)) + min_lag
    return peak, acf_vals[peak]


def forecast_metrics(y_true, y_pred, model_name, llf=None, aic=None, bic=None):
    """Compute RSS, RMSE, and return dict together with model info (AIC/BIC if provided)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Align lengths (guard against off-by-one from differencing/forecasting)
    n = min(len(y_true), len(y_pred))
    residuals = y_true[:n] - y_pred[:n]
    rss = float(np.sum(residuals ** 2))
    rmse = float(np.sqrt(rss / n))
    return {
        'model': model_name,
        'n': n,
        'rss': rss,
        'rmse': rmse,
        'aic': aic,
        'bic': bic,
    }


def print_forecast_table(rows):
    """Print RSS + RMSE + AIC + BIC comparison table."""
    print(f"{'Model':<16} {'RSS':>14} {'RMSE':>10} {'AIC':>12} {'BIC':>12}")
    print("-" * 68)
    for r in rows:
        aic_str = f"{r['aic']:>12.2f}" if r['aic'] is not None else f"{'n/a':>12}"
        bic_str = f"{r['bic']:>12.2f}" if r['bic'] is not None else f"{'n/a':>12}"
        print(f"{r['model']:<16} {r['rss']:>14.2f} {r['rmse']:>10.4f} {aic_str} {bic_str}")


def plot_forecasts(train, test, forecasts, filename):
    """Overlay forecasts from multiple models against true test values."""
    fig, ax = plt.subplots(figsize=(12, 5))
    # Show the tail of the train set so the split is visible
    tail_n = 200
    ax.plot(np.arange(-tail_n, 0), train[-tail_n:], color='gray',
            linewidth=0.8, label='Train (tail)')
    ax.plot(np.arange(len(test)), test, color='black',
            linewidth=1.0, label='Test (true)')
    colour_cycle = [BLUE, GREEN, ORANGE, RED]
    for (name, fc), colour in zip(forecasts.items(), colour_cycle):
        fc_arr = np.asarray(fc)[:len(test)]
        ax.plot(np.arange(len(fc_arr)), fc_arr,
                color=colour, linewidth=1.0, alpha=0.85, label=name)
    ax.axvline(x=0, color='k', linestyle=':', linewidth=1.0)
    ax.set_xlabel('Sample (0 = start of test window)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Forecasts vs Test on Stationarised Signal')
    ax.legend(loc='upper right', ncol=2)
    finish_figure(fig, os.path.join(FIGURES_DIR, filename))

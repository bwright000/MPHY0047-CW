# Question 4: Time-series modelling and forecasting of a single ECG record.
# Workflow:
#   1. Load the 3000-sample record and test stationarity (ADF).
#   2. Compare transforms: first-difference, sqrt, log.
#   3. Select the most stationary version and split train/test (1800/1200).
#   4. Seasonal decomposition of the training signal.
#   5. Choose AR/MA orders from ACF/PACF.
#   6. Fit AR, MA, ARMA, ARIMA; forecast and compare by RSS (+ RMSE/AIC/BIC).

import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

from dataloader import ecg_signal, ECG_SAMPLE_RATE
from plot_style import apply_style, BLUE, RED, GREEN, ORANGE, finish_figure
from timeseries import (
    adf_summary, print_adf_table, select_best_transform,
    plot_transforms, plot_adf_bar, plot_acf_pacf, find_seasonal_period,
    forecast_metrics, print_forecast_table, plot_forecasts,
    FIGURES_DIR,
)

apply_style()


# -----------------------------------------------------------------
# Load signal. Dataloader returns 3001 values because the CSV's first
# entry is a spurious 0 (first real ECG sample begins at index 1).
# Slice to the 3000 valid samples specified by the spec.
# -----------------------------------------------------------------
raw = ecg_signal[1:].astype(float)
assert len(raw) == 3000, f"Expected 3000 samples after trim, got {len(raw)}"
assert (raw > 0).all(), "Log/sqrt assume strictly positive signal - aborting"


def run_stationarity():
    """Step 1: Compare ADF across raw, first-diff, sqrt, log."""
    print("\n" + "=" * 72)
    print("  Q4 Step 1: Stationarity Testing (Augmented Dickey-Fuller)")
    print("=" * 72)

    raw_diff = np.diff(raw)
    raw_sqrt = np.sqrt(raw)
    raw_log = np.log(raw)

    rows = [
        adf_summary(raw, 'Raw'),
        adf_summary(raw_diff, 'First difference'),
        adf_summary(raw_sqrt, 'Square root'),
        adf_summary(raw_log, 'Log'),
    ]
    print_adf_table(rows)

    plot_transforms(
        raw,
        [('First difference', raw_diff),
         ('Square root', raw_sqrt),
         ('Log', raw_log)],
        'q4_transforms.png',
    )
    plot_adf_bar(rows, 'q4_adf_transforms.png')

    best = select_best_transform(rows)
    print(f"\n  Selected transform: '{best['name']}' "
          f"(ADF={best['adf_stat']:.4f}, p={best['p_value']:.4g})")

    # Map the chosen label back to the actual series
    series_map = {
        'Raw': raw,
        'First difference': raw_diff,
        'Square root': raw_sqrt,
        'Log': raw_log,
    }
    return best['name'], series_map[best['name']], rows


def run_split(x_stat):
    """Step 2: Train/test split at sample 1800."""
    print("\n" + "=" * 72)
    print("  Q4 Step 2: Train/Test Split")
    print("=" * 72)
    train = x_stat[:1800]
    test = x_stat[1800:]
    print(f"  Train: {len(train)} samples (indices 0 - {len(train) - 1})")
    print(f"  Test:  {len(test)} samples (indices {len(train)} - {len(x_stat) - 1})")
    return train, test


def run_seasonal_decomposition(train):
    """Step 3: Seasonal decomposition on the training signal."""
    print("\n" + "=" * 72)
    print("  Q4 Step 3: Seasonal Decomposition")
    print("=" * 72)

    period, acf_val = find_seasonal_period(train, min_lag=30, max_lag=500)
    heart_rate_bpm = 60 * ECG_SAMPLE_RATE / period
    print(f"  Detected period: {period} samples (ACF peak = {acf_val:.3f})")
    print(f"  Equivalent heart rate at {ECG_SAMPLE_RATE} Hz: {heart_rate_bpm:.1f} bpm")

    decomp = seasonal_decompose(train, period=period, model='additive',
                                extrapolate_trend='freq')

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(train, color=BLUE, linewidth=0.7)
    axes[0].set_ylabel('Observed')
    axes[1].plot(decomp.trend, color=GREEN, linewidth=0.7)
    axes[1].set_ylabel('Trend')
    axes[2].plot(decomp.seasonal, color=ORANGE, linewidth=0.7)
    axes[2].set_ylabel('Seasonal')
    axes[3].plot(decomp.resid, color=RED, linewidth=0.7)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Sample')
    fig.suptitle(f'Seasonal Decomposition (period = {period} samples, '
                 f'~{heart_rate_bpm:.0f} bpm)')
    finish_figure(fig, os.path.join(FIGURES_DIR, 'q4_seasonal_decompose.png'))
    return period


def run_acf_pacf(train, nlags=40):
    """Step 4: Choose AR and MA orders from ACF and PACF."""
    print("\n" + "=" * 72)
    print(f"  Q4 Step 4: ACF / PACF (nlags={nlags})")
    print("=" * 72)
    plot_acf_pacf(train, nlags=nlags, filename='q4_acf_pacf.png')

    # Numeric ACF/PACF to pick orders automatically (as a sanity check).
    from statsmodels.tsa.stattools import acf, pacf
    acf_vals = acf(train, nlags=nlags, fft=True)
    pacf_vals = pacf(train, nlags=nlags, method='ywm')
    # 95% CI approx: 1.96/sqrt(N)
    ci = 1.96 / np.sqrt(len(train))

    # Pick the smallest lag >=1 where the curve falls back inside the band.
    def cutoff(vals):
        for lag in range(1, len(vals)):
            if abs(vals[lag]) < ci:
                return lag
        return len(vals) - 1

    p = cutoff(pacf_vals)
    q = cutoff(acf_vals)
    # Cap at 5 to keep ARIMA(p,d,q) tractable; the ECG signal is very
    # auto-correlated and both functions are slow to decay.
    p_cap, q_cap = min(p, 5), min(q, 5)
    print(f"  PACF first in-band lag (AR order p): {p} -> using p = {p_cap}")
    print(f"  ACF  first in-band lag (MA order q): {q} -> using q = {q_cap}")
    return p_cap, q_cap


def run_forecasts(x_stat_name, train, test, p, q, raw_signal):
    """
    Step 5/6: Fit AR/MA/ARMA on the stationary series; fit ARIMA on the RAW
    series with d=1 so the `I` step is meaningful. For ARIMA we forecast
    raw values then difference them to put the RSS in the same space as the
    other three models (fair comparison).
    """
    print("\n" + "=" * 72)
    print(f"  Q4 Step 5/6: AR/MA/ARMA/ARIMA Forecasting (p={p}, q={q})")
    print("=" * 72)

    rows = []
    forecasts = {}
    n_test = len(test)

    # AR(p)
    ar = AutoReg(train, lags=p, old_names=False).fit()
    ar_fc = ar.predict(start=len(train), end=len(train) + n_test - 1)
    rows.append(forecast_metrics(test, ar_fc, f'AR({p})',
                                 aic=ar.aic, bic=ar.bic))
    forecasts[f'AR({p})'] = ar_fc

    # MA(q) via ARIMA(0,0,q)
    ma = ARIMA(train, order=(0, 0, q)).fit()
    ma_fc = ma.forecast(steps=n_test)
    rows.append(forecast_metrics(test, ma_fc, f'MA({q})',
                                 aic=ma.aic, bic=ma.bic))
    forecasts[f'MA({q})'] = ma_fc

    # ARMA(p,q) via ARIMA(p,0,q)
    arma = ARIMA(train, order=(p, 0, q)).fit()
    arma_fc = arma.forecast(steps=n_test)
    rows.append(forecast_metrics(test, arma_fc, f'ARMA({p},{q})',
                                 aic=arma.aic, bic=arma.bic))
    forecasts[f'ARMA({p},{q})'] = arma_fc

    # ARIMA(p,1,q) — fit on RAW training series so the integration step is
    # doing real work. Forecasts come back in raw units; difference them to
    # align with the stationary-space test (if the chosen transform was
    # first-difference). Otherwise forecast directly.
    if x_stat_name == 'First difference':
        raw_train = raw_signal[:1801]  # one extra sample so diff lines up
        arima = ARIMA(raw_train, order=(p, 1, q)).fit()
        # Forecast enough points to cover the test window AFTER differencing
        raw_fc = arima.forecast(steps=n_test + 1)
        # Prepend the last observed raw training value so diff aligns
        seed = raw_signal[1800]
        arima_fc = np.diff(np.concatenate([[seed], raw_fc]))[:n_test]
    else:
        # If the stationary series was not first-difference, fit ARIMA on the
        # stationary train directly with d=1 as a modelling demonstration.
        arima = ARIMA(train, order=(p, 1, q)).fit()
        arima_fc = arima.forecast(steps=n_test)

    rows.append(forecast_metrics(test, arima_fc, f'ARIMA({p},1,{q})',
                                 aic=arima.aic, bic=arima.bic))
    forecasts[f'ARIMA({p},1,{q})'] = arima_fc

    print_forecast_table(rows)
    best = min(rows, key=lambda r: r['rss'])
    print(f"\n  Lowest RSS: {best['model']} (RSS = {best['rss']:.2f})")

    plot_forecasts(train, test, forecasts, 'q4_forecasts.png')
    return rows, forecasts


transform_name, x_stat, adf_rows = run_stationarity()
train, test = run_split(x_stat)
seasonal_period = run_seasonal_decomposition(train)
p_order, q_order = run_acf_pacf(train)
forecast_rows, forecast_series = run_forecasts(
    transform_name, train, test, p_order, q_order, raw
)


if __name__ == '__main__':
    print("\n" + "=" * 72)
    print("  Q4: Time-Series Modelling — Summary")
    print("=" * 72)
    print(f"  Transform used:     {transform_name}")
    print(f"  Seasonal period:    {seasonal_period} samples")
    print(f"  Orders (p, q):      ({p_order}, {q_order})")
    print(f"  Best model by RSS:  "
          f"{min(forecast_rows, key=lambda r: r['rss'])['model']}")

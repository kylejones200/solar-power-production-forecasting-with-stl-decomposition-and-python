"""
Solar Power Production Forecasting with STL Decomposition

Workflow:
  1. Fetch hourly PVLIB synthetic solar irradiance (clear-sky GHI) as a proxy
     for production — replace with real plant data if available.
  2. Resample to daily totals (kWh/m²).
  3. Decompose with STL (Seasonal-Trend decomposition using LOESS).
  4. Forecast the trend component with ARIMA.
  5. Re-add seasonal + residual to produce a full forecast.
  6. Evaluate with MAE, RMSE, and MAPE on a held-out test set.
  7. Generate three figures.

Usage:
    pip install pvlib statsmodels scikit-learn matplotlib pandas numpy
    python solar_forecast.py
"""

import signalplot
import logging
import warnings

from pathlib import Path
import yaml


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        return _yaml.safe_load(_f) or {}

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL

try:
    import pvlib
    HAS_PVLIB = True
except ImportError:
    HAS_PVLIB = False

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

signalplot.apply(font_family='serif')

# --- Location: Albuquerque, NM (high solar resource) ---
LATITUDE = 35.0844
LONGITUDE = -106.6504
ALTITUDE = 1500
TIMEZONE = "US/Mountain"

TRAIN_YEARS = 3
TEST_DAYS = 90
ARIMA_ORDER = (2, 1, 2)


def generate_solar_series(
    start: str = "2020-01-01",
    end: str = "2023-12-31",
) -> pd.Series:
    """
    Generate daily GHI (W·h/m²) using pvlib clear-sky model.
    Falls back to a synthetic seasonal series if pvlib is unavailable.
    """
    if HAS_PVLIB:
        logger.info("Generating pvlib clear-sky GHI for Albuquerque, NM...")
        times = pd.date_range(start, end, freq="1h", tz=TIMEZONE)
        location = pvlib.location.Location(LATITUDE, LONGITUDE, TIMEZONE, ALTITUDE)
        cs = location.get_clearsky(times)
        daily = cs["ghi"].resample("D").sum() / 1000  # kWh/m²
        daily.name = "ghi_kwh_m2"
        return daily.tz_localize(None)
    logger.info("pvlib not found — using synthetic seasonal series.")
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])
    seasonal = 4.5 + 2.8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    trend = np.linspace(0, 0.3, n)
    noise = np.random.default_rng(42).normal(0, 0.25, n)
    series = pd.Series(
        np.clip(seasonal + trend + noise, 0, None),
        index=dates,
        name="ghi_kwh_m2",
    )
    return series


def stl_decompose(series: pd.Series, period: int = 365) -> STL:
    stl = STL(series, period=period, robust=True)
    return stl.fit()


def fit_arima_on_trend(trend: pd.Series, order: tuple) -> ARIMA:
    model = ARIMA(trend, order=order)
    return model.fit()


def forecast(
    series: pd.Series,
    test_days: int = TEST_DAYS,
) -> tuple[pd.Series, pd.Series, dict]:
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]

    # STL on training data
    result = stl_decompose(train)
    trend_train = result.trend
    seasonal_train = result.seasonal
    resid_train = result.resid

    # ARIMA on trend component
    arima_fit = fit_arima_on_trend(trend_train, ARIMA_ORDER)
    trend_forecast = arima_fit.forecast(steps=test_days)
    trend_forecast.index = test.index

    # Repeat last 365 days of seasonal pattern
    seasonal_cycle = seasonal_train.values[-365:]
    repeats = int(np.ceil(test_days / 365))
    seasonal_forecast = pd.Series(
        np.tile(seasonal_cycle, repeats)[:test_days],
        index=test.index,
    )

    # Residual mean (small, mostly noise)
    resid_mean = float(resid_train.mean())

    forecast_series = trend_forecast + seasonal_forecast + resid_mean
    forecast_series = forecast_series.clip(lower=0)

    mae = mean_absolute_error(test, forecast_series)
    rmse = np.sqrt(mean_squared_error(test, forecast_series))
    mape = float(np.mean(np.abs((test - forecast_series) / test.clip(lower=0.1))) * 100)

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}
    return test, forecast_series, metrics


def plot_full_series(series: pd.Series, test_start: pd.Timestamp, path: str, plot: bool = False):
    if plot:
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [12, 4])))
        ax.plot(series.index, series.values, linewidth=0.7, color="black", label="Daily GHI")
        ax.axvline(test_start, color="red", linestyle="--", linewidth=0.9, label="Train/Test split")
        ax.set_title("Daily Solar Irradiance — Albuquerque, NM (kWh/m²)")
        ax.set_ylabel("GHI (kWh/m²)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved: %s", path)


def plot_stl(result, path: str, plot: bool = False):
    if plot:
        fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
        components = [
            (result.observed, "Observed"),
            (result.trend, "Trend"),
            (result.seasonal, "Seasonal"),
            (result.resid, "Residual"),
        ]
        for ax, (data, label) in zip(axes, components):
            ax.plot(data.index, data.values, linewidth=0.7, color="black")
            ax.set_ylabel(label, fontsize=9)
        axes[0].set_title("STL Decomposition of Daily Solar GHI")
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved: %s", path)


def plot_forecast(actual: pd.Series, predicted: pd.Series, metrics: dict, path: str, plot: bool = False):
    if plot:
        fig, ax = plt.subplots(figsize=tuple(config.get('output', {}).get('figsize', [12, 4])))
        ax.plot(actual.index, actual.values, color="black", linewidth=1.2, label="Actual")
        ax.plot(predicted.index, predicted.values, color="red", linewidth=1.2,
                linestyle="--", label="STL+ARIMA Forecast")
        ax.set_title(
            f"Solar GHI Forecast — Test Set  |  MAE={metrics['MAE']:.2f}  "
            f"RMSE={metrics['RMSE']:.2f}  MAPE={metrics['MAPE (%)']:.1f}%"
        )
        ax.set_ylabel("GHI (kWh/m²)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    logger.info("Saved: %s", path)


def main():
    series = generate_solar_series()
    logger.info("Series: %d days  (%s → %s)", len(series), series.index[0].date(), series.index[-1].date())

    plot_full_series(series, series.index[-TEST_DAYS], "01_solar_full_series.png")

    # STL on full training window for display
    train_result = stl_decompose(series.iloc[:-TEST_DAYS])
    plot_stl(train_result, "02_solar_stl_decomposition.png")

    actual, predicted, metrics = forecast(series)
    logger.info("Forecast metrics: %s", metrics)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3f}")

    plot_forecast(actual, predicted, metrics, "03_solar_forecast.png")
    logger.info("\nOutputs: 01_solar_full_series.png  02_solar_stl_decomposition.png  03_solar_forecast.png")


if __name__ == "__main__":
    main()

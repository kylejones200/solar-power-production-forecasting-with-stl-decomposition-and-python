# Solar Power Production Forecasting with STL Decomposition and Python

*Breaking solar irradiance into trend, seasonality, and noise — then forecasting each component separately*

---

Solar irradiance follows one of the most predictable patterns in all of time series analysis. The sun rises and sets on a fixed schedule. Peak output in July in Albuquerque is reliably higher than in December. And yet solar production forecasting is harder than it looks: weather noise, year-over-year climate drift, and the difference between daily, hourly, and sub-hourly dynamics each add complexity.

The standard mistake is treating daily solar output as a single time series and throwing ARIMA at it. ARIMA handles the trend but struggles with the 365-day seasonal cycle — the seasonal differencing required is both computationally expensive and statistically fragile.

STL decomposition (Seasonal-Trend decomposition using LOESS) solves this cleanly by separating the problem:

1. Trend — the slow drift in baseline production (climate, degradation, capacity changes)
2. Seasonal — the repeating annual solar cycle
3. Residual — weather noise and unexplained variation

You then forecast the trend with ARIMA, project the seasonal component by repeating the most recent cycle, and recombine. Each model does what it's good at.

## Why STL over SARIMA?

SARIMA with `s=365` requires estimating seasonal AR and MA parameters at yearly lags. With daily data spanning a few years, there's simply not enough data to estimate these reliably. STL extracts the seasonal pattern non-parametrically via LOESS, avoiding this problem entirely.

## Data

This example uses `pvlib` to generate a clear-sky GHI (Global Horizontal Irradiance) time series for Albuquerque, NM — a high-solar-resource location at 35°N. In production, replace with measured plant output or NSRDB data.

If `pvlib` is not installed, the script falls back to a synthetic seasonal series with realistic amplitude and noise.

## Methodology

```
daily_ghi = trend + seasonal + residual        (STL)
forecast  = ARIMA(trend) + seasonal[-365:] + mean(residual)
```

- Train: 3 years of daily GHI
- Test: final 90 days (one quarter)
- ARIMA order: (2,1,2) on trend component

## Results

| Metric | Value |
|--------|-------|
| MAE | ~0.35 kWh/m² |
| RMSE | ~0.48 kWh/m² |
| MAPE | ~8–12% |

Results vary with location and weather noise. Clear-sky synthetic data produces lower error; real measured data with cloud cover will show higher MAPE.

## Quickstart

```bash
pip install pvlib statsmodels scikit-learn matplotlib pandas numpy
python solar_forecast.py
```

Outputs:
- `01_solar_full_series.png` — full daily series with train/test split
- `02_solar_stl_decomposition.png` — trend, seasonal, residual components
- `03_solar_forecast.png` — actual vs. forecast on 90-day test set

## Files

| File | Description |
|------|-------------|
| `solar_forecast.py` | Main script: generate data, decompose, forecast, evaluate |

## Related

- [Forecasting Solar Irradiance with Regime-Aware LSTM](https://github.com/kylejones200/forecasting-solar-irradiance-with-regime-aware-lstm)
- [Solar Irradiance with ARIMA, SARIMAX, and Gradient Boosting](https://github.com/kylejones200/time-series-forecasting-solar-irradiance-with-arima-sarimax-and-gradient-boosting-using)

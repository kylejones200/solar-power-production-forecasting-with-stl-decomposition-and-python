# Repository

Companion code for a Medium article.

## Business context

*Breaking solar irradiance into trend, seasonality, and noise — then forecasting each component separately*

Solar irradiance follows one of the most predictable patterns in all of time series analysis. The sun rises and sets on a fixed schedule. Peak output in July in Albuquerque is reliably higher than in December. And yet solar production forecasting is harder than it looks: weather noise, year-over-year climate drift, and the difference between daily, hourly, and sub-hourly dynamics each add complexity.

The standard mistake is treating daily solar output as a single time series and throwing ARIMA at it. ARIMA handles the trend but struggles with the 365-day seasonal cycle — the seasonal differencing required is both computationally expensive and statistically fragile.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
# Repository

Companion code for a Medium article.

## Business context

*Breaking solar irradiance into trend, seasonality, and noise — then forecasting each component separately*

Solar irradiance follows one of the most predictable patterns in all of time series analysis. The sun rises and sets on a fixed schedule. Peak output in July in Albuquerque is reliably higher than in December. And yet solar production forecasting is harder than it looks: weather noise, year-over-year climate drift, and the difference between daily, hourly, and sub-hourly dynamics each add complexity.

The standard mistake is treating daily solar output as a single time series and throwing ARIMA at it. ARIMA handles the trend but struggles with the 365-day seasonal cycle — the seasonal differencing required is both computationally expensive and statistically fragile.

## Rust performance port

Side-by-side **Python vs Rust** implementation of the numeric hot loop — moving average trend. Reference PyO3 benchmark: **see `benchmark_rust.py`** on a release build (local machine; run `benchmark_rust.py` to reproduce).

| Path | Role |
|------|------|
| `src/compute_kernel.py` | Python/numpy reference kernel |
| `rust/core/` | Pure Rust library |
| `rust/py/` | PyO3 bindings |
| `rust/bench/` | Standalone CLI benchmark |
| `benchmark_rust.py` | Python vs Rust timing + correctness check |

```bash
# Rust-only CLI benchmark
cd rust && cargo run --release -p solar_power_production_forecasting_with_stl_decomposition_and_python_bench

# Python vs Rust (PyO3)
pip install maturin numpy
maturin develop --release -m rust/py/Cargo.toml
python benchmark_rust.py
```

Python ML training, solvers, and orchestration stay in Python; Rust targets the numeric hot loops. Stochastic generators validate output shapes; deterministic kernels match at tight floating-point tolerance.


## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
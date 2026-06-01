use solar_power_production_forecasting_with_stl_decomposition_and_python_core::moving_average_trend;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use pyo3::prelude::*;

#[pyfunction]
fn moving_average_trend_py<'py>(py: Python<'py>, series: PyReadonlyArray1<f64>, window: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(moving_average_trend(series.as_slice()?, window).into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (series, window, iterations=500))]
fn bench_kernel_py(series: PyReadonlyArray1<f64>, window: usize, iterations: usize) -> PyResult<f64> {
    let series_buf = series.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = moving_average_trend(&series_buf, window);
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn solar_power_production_forecasting_with_stl_decomposition_and_python_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(moving_average_trend_py, m)?)?;
    m.add_function(wrap_pyfunction!(bench_kernel_py, m)?)?;
    Ok(())
}

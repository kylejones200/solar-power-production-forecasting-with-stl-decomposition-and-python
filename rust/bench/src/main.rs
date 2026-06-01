use solar_power_production_forecasting_with_stl_decomposition_and_python_core::moving_average_trend;

fn main() {
    let s: Vec<f64> = (0..5000).map(|i| (i as f64 * 0.01).sin() + 100.0).collect();
    for _ in 0..2000 {
        let _ = moving_average_trend(&s, 24);
    }
}

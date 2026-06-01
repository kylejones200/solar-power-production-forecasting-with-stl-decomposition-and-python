//! Centered moving average trend (STL-style trend component).

pub fn moving_average_trend(series: &[f64], window: usize) -> Vec<f64> {
    let n = series.len();
    let w = window.max(1);
    if n == 0 {
        return vec![];
    }
    let pad = (w - 1) / 2;
    let mut full = vec![0.0; n + w - 1];
    for k in 0..full.len() {
        let mut sum = 0.0;
        for j in 0..w {
            let ai = k as isize - j as isize;
            if ai >= 0 && (ai as usize) < n {
                sum += series[ai as usize];
            }
        }
        full[k] = sum / w as f64;
    }
    full[pad..pad + n].to_vec()
}

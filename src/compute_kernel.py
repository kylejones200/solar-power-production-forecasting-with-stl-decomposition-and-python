"""Centered moving average trend (STL-style trend component)."""

from __future__ import annotations

import numpy as np


def moving_average_trend(series: np.ndarray, window: int) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    n = len(s)
    w = max(window, 1)
    if n == 0:
        return np.empty(0, dtype=float)
    pad = (w - 1) // 2
    full_len = n + w - 1
    full = np.zeros(full_len, dtype=float)
    for k in range(full_len):
        total = 0.0
        for j in range(w):
            ai = k - j
            if ai >= 0 and ai < n:
                total += s[ai]
        full[k] = total / w
    return full[pad : pad + n]

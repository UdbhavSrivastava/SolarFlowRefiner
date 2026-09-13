from __future__ import annotations

import numpy as np


def global_ssim(target: np.ndarray, pred: np.ndarray, data_range: float = 1000.0) -> float:
    target = target.astype(np.float64)
    pred = pred.astype(np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x = float(target.mean())
    mu_y = float(pred.mean())
    var_x = float(target.var())
    var_y = float(pred.var())
    cov = float(((target - mu_x) * (pred - mu_y)).mean())
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def percent_decrease_to_lfc(percent):
    values = np.asarray(percent, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values >= 100):
        raise ValueError("Percent decrease must be finite and in [0, 100).")
    return np.log1p(-values / 100.0)


def percent_change_to_lfc(percent: float, direction: str) -> float:
    if not np.isfinite(percent) or not 0 < percent < 100 or direction not in {"increase", "decrease"}:
        raise ValueError("effect must be in (0,100) and direction increase/decrease")
    return float(np.log1p(percent / 100 if direction == "increase" else -percent / 100))


def nb_wald_power(
    baseline_mean,
    theta,
    cells_per_arm,
    lfc,
    depth_multiplier=1.0,
    alpha: float = 0.05,
    critical_z: float | None = None,
):
    if not (0 < float(alpha) < 1):
        raise ValueError("alpha must be between 0 and 1")

    mu, th, n, delta, depth = np.broadcast_arrays(
        np.asarray(baseline_mean, dtype=float),
        np.asarray(theta, dtype=float),
        np.asarray(cells_per_arm, dtype=float),
        np.asarray(lfc, dtype=float),
        np.asarray(depth_multiplier, dtype=float),
    )

    valid = (
        np.isfinite(mu)
        & np.isfinite(th)
        & np.isfinite(n)
        & np.isfinite(delta)
        & np.isfinite(depth)
        & (mu > 0)
        & (th > 0)
        & (n > 0)
        & (depth > 0)
    )

    out = np.full(mu.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out

    mu_depth = mu[valid] * depth[valid]
    per_arm_log_mean_var = 1.0 / (n[valid] * mu_depth) + 1.0 / (n[valid] * th[valid])
    se_delta = np.sqrt(2.0 * per_arm_log_mean_var)
    ncp = np.abs(delta[valid]) / se_delta
    if critical_z is not None and (not np.isfinite(critical_z) or critical_z <= 0):
        raise ValueError("critical_z must be finite and positive")
    zcrit = float(critical_z) if critical_z is not None else norm.ppf(1.0 - alpha / 2.0)
    power = norm.sf(zcrit - ncp) + norm.cdf(-zcrit - ncp)
    out[valid] = np.clip(power, 0.0, 1.0)
    return out

"""Normal-approximation safety stock formulas (assumptions are stated in the README)."""
from __future__ import annotations

import math
from statistics import NormalDist

_N = NormalDist()


def z_from_service_level(service_level: float) -> float:
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level must be in [0.5, 1.0)")
    return _N.inv_cdf(service_level)


def normal_cdf(x: float) -> float:
    return _N.cdf(x)


def demand_std_over(days: float, sigma_daily: float, mean_daily: float, sigma_lead_days: float = 0.0) -> float:
    """Std-dev of demand over a protection interval of `days` days with variable lead time:
    sqrt(days*sigma_d^2 + mean_d^2*sigma_L^2). Assumes independent daily forecast errors, which
    typically UNDERSTATES risk when errors are autocorrelated."""
    return math.sqrt(max(days, 0.0) * sigma_daily ** 2 + (mean_daily ** 2) * sigma_lead_days ** 2)


def safety_stock(z: float, sigma_protect: float) -> float:
    return max(0.0, z * sigma_protect)

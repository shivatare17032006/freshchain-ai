from __future__ import annotations

from src import config
from src.forecasting.lightgbm_model import GBMForecaster


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


class XGBoostForecaster(GBMForecaster):
    """Optional comparison model; same interface as GBMForecaster."""

    def __init__(self, horizon: int, params: dict | None = None, seed: int = config.SEED, **kw):
        if not xgboost_available():
            raise ImportError("xgboost is not installed. `pip install xgboost` to enable the comparison model.")
        super().__init__(horizon, backend="xgboost", params=params, seed=seed, **kw)

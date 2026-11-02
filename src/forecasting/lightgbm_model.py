"""Global gradient-boosting forecaster. Primary backend: LightGBM. If LightGBM is not installed
the model falls back to scikit-learn's HistGradientBoostingRegressor and says so in `backend`."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

DEFAULT_PARAMS = {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50,
                  "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0}
CAT_COLS = ("store_id", "category")


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except Exception:
        return False


class GBMForecaster:
    def __init__(self, horizon: int, backend: str = "auto", params: dict | None = None,
                 seed: int = config.SEED, stockout_exclude_hours: float | None = config.STOCKOUT_EXCLUDE_HOURS):
        if backend == "auto":
            backend = "lightgbm" if lightgbm_available() else "sklearn_hgb"
        if backend not in ("lightgbm", "xgboost", "sklearn_hgb"):
            raise ValueError(f"Unknown backend '{backend}'")
        self.horizon = horizon
        self.backend = backend
        self.params = dict(params or {})
        self.seed = seed
        self.stockout_exclude_hours = stockout_exclude_hours
        self.base_features_: list[str] = []
        self.feature_cols_: list[str] = []
        self.cat_maps_: dict = {}
        self.estimator_ = None
        self.best_iteration_ = None
        self.interval_ = None
        self.meta_: dict = {}

    # ---- matrix --------------------------------------------------------------------------
    def _matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.base_features_].astype(float).copy()
        for c in CAT_COLS:
            if c in df.columns:
                X[f"{c}_code"] = df[c].astype(str).map(self.cat_maps_[c]).astype(float)
        return X[self.feature_cols_]

    def _train_rows(self, df: pd.DataFrame, mask) -> pd.DataFrame:
        sub = df[mask]
        warm = [c for c in self.base_features_ if c.startswith("roll_mean_")]
        sub = sub[sub[warm].notna().all(axis=1)] if warm else sub
        if self.stockout_exclude_hours is not None and "stockout_hours" in sub.columns:
            sub = sub[sub["stockout_hours"] < self.stockout_exclude_hours]
        return sub

    def _make_estimator(self, n_estimators: int, early: bool):
        p = {**DEFAULT_PARAMS, **self.params}
        n_estimators = int(p.pop("n_estimators") if n_estimators is None else n_estimators)
        if self.backend == "lightgbm":
            import lightgbm as lgb
            return lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=p["learning_rate"],
                                     num_leaves=p["num_leaves"], min_child_samples=p["min_child_samples"],
                                     subsample=p["subsample"], subsample_freq=p["subsample_freq"],
                                     colsample_bytree=p["colsample_bytree"], reg_lambda=p["reg_lambda"],
                                     random_state=self.seed, n_jobs=-1, verbose=-1)
        if self.backend == "xgboost":
            import xgboost as xgb
            return xgb.XGBRegressor(n_estimators=n_estimators, learning_rate=p["learning_rate"], max_depth=8,
                                    subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                                    tree_method="hist", random_state=self.seed, n_jobs=-1,
                                    early_stopping_rounds=30 if early else None)
        from sklearn.ensemble import HistGradientBoostingRegressor
        # early_stopping=False on purpose: sklearn's internal early stopping uses a RANDOM split,
        # which would violate chronological validation.
        return HistGradientBoostingRegressor(max_iter=min(n_estimators, 300), learning_rate=max(p["learning_rate"], 0.08),
                                             max_leaf_nodes=p["num_leaves"], min_samples_leaf=p["min_child_samples"],
                                             early_stopping=False, random_state=self.seed)

    # ---- fit / predict ---------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, feature_cols: list[str], train_mask, valid_mask=None):
        self.base_features_ = list(feature_cols)
        tr = self._train_rows(df, train_mask)
        if tr.empty:
            raise ValueError("No training rows remain after warm-up / stock-out filtering.")
        self.cat_maps_ = {c: {v: i for i, v in enumerate(sorted(tr[c].astype(str).unique()))}
                          for c in CAT_COLS if c in tr.columns}
        self.feature_cols_ = self.base_features_ + [f"{c}_code" for c in self.cat_maps_]
        Xtr, ytr = self._matrix(tr), tr["demand"].to_numpy(float)
        va = self._train_rows(df, valid_mask) if valid_mask is not None else None
        has_val = va is not None and len(va) > 0
        est = self._make_estimator(self.params.get("n_estimators"), early=has_val)
        cats = [f"{c}_code" for c in self.cat_maps_]
        if self.backend == "lightgbm":
            import lightgbm as lgb
            kw = {"categorical_feature": cats}
            if has_val:
                kw.update(eval_set=[(self._matrix(va), va["demand"].to_numpy(float))],
                          callbacks=[lgb.early_stopping(30, verbose=False)])
            est.fit(Xtr, ytr, **kw)
            self.best_iteration_ = int(est.best_iteration_) if getattr(est, "best_iteration_", None) else None
        elif self.backend == "xgboost":
            kw = {"verbose": False}
            if has_val:
                kw["eval_set"] = [(self._matrix(va), va["demand"].to_numpy(float))]
            est.fit(Xtr, ytr, **kw)
            self.best_iteration_ = int(est.best_iteration) + 1 if has_val and getattr(est, "best_iteration", None) is not None else None
        else:
            est.fit(Xtr, ytr)
            self.best_iteration_ = None
        self.estimator_ = est
        self.meta_.update(n_train_rows=int(len(tr)), n_valid_rows=int(len(va)) if has_val else 0,
                          train_end=str(tr["date"].max().date()))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("Model is not fitted.")
        return np.clip(self.estimator_.predict(self._matrix(df)), 0, None)

    # ---- empirical prediction intervals -----------------------------------------------------
    def calibrate_intervals(self, df: pd.DataFrame, mask, coverage: float = config.INTERVAL_COVERAGE):
        """Empirical-residual intervals: quantiles of (actual - prediction) on a chronological
        calibration window (non-stock-out days). Coverage is then checked on held-out data."""
        sub = df[mask]
        if self.stockout_exclude_hours is not None and "stockout_hours" in sub.columns:
            sub = sub[sub["stockout_hours"] < self.stockout_exclude_hours]
        sub = sub[sub[[c for c in self.base_features_ if c.startswith("roll_mean_")]].notna().all(axis=1)]
        if len(sub) < 30:
            self.interval_ = None
            return None
        res = sub["demand"].to_numpy(float) - self.predict(sub)
        a = (1 - coverage) / 2
        self.interval_ = {"lo": float(np.quantile(res, a)), "hi": float(np.quantile(res, 1 - a)),
                          "nominal_coverage": coverage, "n_calibration": int(len(sub))}
        return self.interval_

    def predict_interval(self, df: pd.DataFrame, pred: np.ndarray | None = None):
        if self.interval_ is None:
            return None, None
        pred = self.predict(df) if pred is None else pred
        return np.clip(pred + self.interval_["lo"], 0, None), np.clip(pred + self.interval_["hi"], 0, None)

    # ---- explainability ---------------------------------------------------------------------
    def feature_importance(self, df: pd.DataFrame | None = None, mask=None, max_rows: int = 5000) -> pd.DataFrame:
        if self.backend == "lightgbm":
            imp = self.estimator_.booster_.feature_importance(importance_type="gain")
        elif self.backend == "xgboost":
            imp = self.estimator_.feature_importances_
        else:
            if df is None:
                return pd.DataFrame(columns=["feature", "importance"])
            from sklearn.inspection import permutation_importance
            sub = self._train_rows(df, mask if mask is not None else np.ones(len(df), bool))
            sub = sub.sample(min(max_rows, len(sub)), random_state=self.seed)
            r = permutation_importance(self.estimator_, self._matrix(sub), sub["demand"], n_repeats=3,
                                       random_state=self.seed, n_jobs=1)
            imp = r.importances_mean
        out = pd.DataFrame({"feature": self.feature_cols_, "importance": np.asarray(imp, dtype=float)})
        tot = out["importance"].clip(lower=0).sum()
        out["importance_share"] = out["importance"].clip(lower=0) / tot if tot > 0 else 0.0
        return out.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "GBMForecaster":
        """Only load artifacts produced locally by this project (joblib uses pickle)."""
        import joblib
        return joblib.load(path)

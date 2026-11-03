"""Central configuration. Everything can be overridden via environment variables or a local .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path | None = None) -> None:
    """Minimal .env reader (no extra dependency). Existing environment variables win."""
    p = path or ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

DATA_DIR = Path(os.environ.get("FRESHCHAIN_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = Path(os.environ.get("FRESHCHAIN_ARTIFACT_DIR", ROOT / "artifacts"))
DB_PATH = Path(os.environ.get("FRESHCHAIN_DB_PATH", DATA_DIR / "freshchain.sqlite"))

HISTORY_FILE = "demand_daily.parquet"
META_FILE = "dataset_meta.json"
INVENTORY_FILE = "inventory.parquet"
RUN_SUMMARY_FILE = "run_summary.json"

# ---- Canonical schema -------------------------------------------------------------------
CANONICAL_REQUIRED = ("date", "store_id", "product_id", "demand")
CANONICAL_OPTIONAL = ("category", "price", "discount", "promo", "holiday", "stockout_hours")

# ---- Forecasting ------------------------------------------------------------------------
HORIZONS = (7, 14, 28)
SEASON = 7
LAG_OFFSETS = (0, 1, 7)          # lags are horizon + offset  (direct strategy, leakage-safe)
ROLL_WINDOWS = (7, 14, 28)
MIN_TRAIN_DAYS = 28              # usable training days after feature warm-up
VAL_DAYS = 7
SEED = 42
INTERVAL_COVERAGE = 0.80
# Days with at least this many stock-out hours are treated as censored and excluded from
# model *training* (they are still reported in evaluation). Heuristic - tune per dataset.
STOCKOUT_EXCLUDE_HOURS = 3


def required_history_days(horizon: int, windows=ROLL_WINDOWS) -> int:
    """Minimum number of calendar days needed to train+validate+test a horizon-h model."""
    warmup = horizon + max(windows)
    return warmup + MIN_TRAIN_DAYS + VAL_DAYS + horizon


# ---- Upload limits ----------------------------------------------------------------------
ALLOWED_UPLOAD_EXTENSIONS = (".csv", ".parquet")
MAX_UPLOAD_MB = int(os.environ.get("FRESHCHAIN_MAX_UPLOAD_MB", "200"))

# ---- Inventory defaults (ASSUMPTIONS - always labelled as such in the UI) -----------------
DEFAULT_INVENTORY_ASSUMPTIONS = {
    "lead_time_days": 2,
    "lead_time_std_days": 0.0,
    "review_period_days": 1,
    "service_level": 0.95,
    "moq": 0,
    "pack_size": 1,
}

# ---- Waste risk -------------------------------------------------------------------------
WASTE_HIGH = 60.0
WASTE_MEDIUM = 30.0
DEFAULT_COVER_REFERENCE_DAYS = 14.0
WASTE_WEIGHTS = {"coverage": 0.40, "decline": 0.25, "variability": 0.20, "perishable": 0.15}

# ---- Alerts -----------------------------------------------------------------------------
DEFAULT_ALERT_THRESHOLDS = {
    "stockout_risk": 0.20,         # P(stock-out during lead time)
    "excess_days_cover": 14.0,     # days of cover on hand above which excess is flagged
    "spike_z": 3.0,                # recent-7d mean vs prior-21d distribution
    "decline_pct": 0.40,           # fractional decline recent-7d vs prior-21d mean
    "wape": 0.60,                  # per-series hold-out WAPE
    "min_volume": 20.0,            # min hold-out actual volume to evaluate WAPE
    "freshness_days": 7,           # days between latest data point and "today"
}


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    base_url: str
    model: str
    timeout: float

    @property
    def is_local(self) -> bool:
        return any(h in self.base_url for h in ("://localhost", "://127.0.0.1", "://[::1]"))

    @property
    def enabled(self) -> bool:
        return self.provider != "none" and (bool(self.api_key) or (self.provider == "openai_compatible" and self.is_local))


def llm_settings() -> LLMSettings:
    """Read LLM settings at call time so they can change without restarting."""
    provider = os.environ.get("LLM_PROVIDER", "none").strip().lower() or "none"
    default_url = "https://api.anthropic.com" if provider == "anthropic" else "https://api.openai.com/v1"
    default_model = "claude-sonnet-5-5" if provider == "anthropic" else ""
    return LLMSettings(
        provider=provider,
        api_key=os.environ.get("LLM_API_KEY", "").strip(),
        base_url=(os.environ.get("LLM_BASE_URL", "").strip() or default_url).rstrip("/"),
        model=os.environ.get("LLM_MODEL", "").strip() or default_model,
        timeout=float(os.environ.get("LLM_TIMEOUT_SECONDS", "20")),
    )

"""Builds a small end-to-end context from SYNTHETIC data (test fixture only)."""
import tempfile
from pathlib import Path

from src.analytics.context import load_context
from src.data.loaders import save_processed
from src.data.preprocessing import clean_canonical
from src.storage.repository import Repository
from synthetic import make_raw
from scripts.train import run as train_run

_CACHE = {}


def build_ctx(with_inventory: bool = False):
    key = with_inventory
    if key in _CACHE:
        return _CACHE[key]
    tmp = Path(tempfile.mkdtemp(prefix="fc_test_"))
    pdir, adir = tmp / "processed", tmp / "artifacts"
    clean, rep = clean_canonical(make_raw(n_stores=3, n_products=5, days=100, seed=3))
    save_processed(clean, {"dataset": "SYNTHETIC test fixture", "synthetic": True, "cleaning": rep}, pdir)
    repo = Repository(tmp / "t.sqlite")
    train_run(horizons=(7, 14), backend="auto", folds=2, data_dir=pdir, artifact_dir=adir, repo=repo)
    if with_inventory:
        ids = clean.drop_duplicates("series_id")[["store_id", "product_id"]].reset_index(drop=True)
        ids["on_hand"] = [5.0 if i % 2 == 0 else 400.0 for i in range(len(ids))]
        ids["unit_cost"] = 2.0
        ids.to_parquet(pdir / "inventory.parquet")
    ctx = load_context(pdir, adir, tmp / "t.sqlite")
    _CACHE[key] = ctx
    return ctx

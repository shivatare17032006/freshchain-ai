"""Create SYNTHETIC smoke-test data so the app can be exercised without downloading anything.
The processed table is flagged synthetic=True and the UI shows a banner. NEVER use for conclusions.

    python -m scripts.make_synthetic
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from src.data.loaders import save_processed  # noqa: E402
from src.data.preprocessing import clean_canonical  # noqa: E402
from synthetic import make_raw  # noqa: E402


def main() -> int:
    clean, rep = clean_canonical(make_raw(n_stores=5, n_products=12, days=120, seed=1))
    save_processed(clean, {"dataset": "SYNTHETIC smoke-test data", "synthetic": True, "sampling": "n/a (generated)",
                           "source_file": "tests/synthetic.py", "cleaning": rep})
    print(f"Wrote synthetic data: {rep['rows_out']:,} rows, {rep['series']} series. Now run: python -m scripts.train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

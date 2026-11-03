"""Create SYNTHETIC demo data (+ demo inventory) so the app can be used without downloading anything.

    python -m scripts.make_synthetic            # data only
    python -m scripts.make_synthetic --train    # data + train all supported horizons
"""
from __future__ import annotations

import sys

from src.demo import build_demo


def main() -> int:
    out = build_demo(train="--train" in sys.argv, progress=lambda m, f: print(f"[{f:4.0%}] {m}"))
    print(f"Wrote synthetic data: {out['rows']:,} rows, {out['series']} series.")
    if "--train" not in sys.argv:
        print("Now run: python -m scripts.train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

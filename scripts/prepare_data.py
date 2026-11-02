"""Raw dataset -> cleaned canonical daily table (data/processed/demand_daily.parquet).

    python -m scripts.prepare_data --raw data/raw/freshretailnet_train.parquet --max-series 2000   # dev sample
    python -m scripts.prepare_data --raw data/raw/freshretailnet_train.parquet --full              # all series
"""
from __future__ import annotations

import argparse

from src import config
from src.data.dataset_adapters import get_adapter
from src.data.loaders import load_raw_canonical, save_processed
from src.data.preprocessing import clean_canonical
from src.data.validation import validate_canonical


def run(raw, adapter="freshretailnet", max_series=None, seed=config.SEED, date_from=None, date_to=None, out_dir=None) -> dict:
    ad = get_adapter(adapter)
    canon, info = load_raw_canonical(raw, ad, max_series=max_series, seed=seed, date_from=date_from, date_to=date_to)
    rep = validate_canonical(canon)
    if not rep.ok:
        raise ValueError("Validation failed: " + "; ".join(rep.errors))
    clean, crep = clean_canonical(canon)
    meta = {"dataset": ad.name, "description": ad.description, "adapter_notes": list(ad.notes), "synthetic": False,
            "sampling": info["sampling"], "source_file": info["source_file"], "validation_warnings": rep.warnings,
            "cleaning": crep, "seed": seed}
    save_processed(clean, meta, out_dir)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--adapter", default="freshretailnet")
    ap.add_argument("--max-series", type=int, default=2000, help="development sample size (series); ignored with --full")
    ap.add_argument("--full", action="store_true", help="process all series (needs more RAM)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--date-from")
    ap.add_argument("--date-to")
    a = ap.parse_args(argv)
    meta = run(a.raw, a.adapter, None if a.full else a.max_series, a.seed, a.date_from, a.date_to)
    c = meta["cleaning"]
    print(f"Prepared {c['rows_out']:,} rows / {c['series']:,} series, {c['date_min']} .. {c['date_max']}. Sampling: {meta['sampling']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

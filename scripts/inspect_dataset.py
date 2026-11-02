# -------------------------------------------------
# scripts/inspect_dataset.py
# -------------------------------------------------
import argparse
from pathlib import Path
import pyarrow.parquet as pq

def main():
    parser = argparse.ArgumentParser(
        description="Print the schema and basic stats of a FreshRetailNet parquet file"
    )
    parser.add_argument("parquet_path", type=Path, help="Path to *.parquet")
    args = parser.parse_args()
    if not args.parquet_path.exists():
        raise FileNotFoundError(f"{args.parquet_path} not found")

    table = pq.read_table(args.parquet_path)
    print("=== Parquet schema ===")
    for name, typ in zip(table.column_names, table.schema.types):
        print(f"{name}: {typ}")

    # Basic stats
    print("\n=== File stats ===")
    print(f"Rows   : {table.num_rows:,}")
    print(f"Columns: {len(table.column_names)}")
    size_mb = args.parquet_path.stat().st_size / (1024**2)
    print(f"Size   : {size_mb:.2f} MiB")
    if "date" in table.column_names:
        dates = table.column("date").to_pylist()
        print(f"Date range: {min(dates)} → {max(dates)}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convert every .parquet in a folder to a single JSON array per file,
using Polars + the standard json module.

Usage:
    python parquet2json_array.py <parquet_dir> <output_dir>
"""

from pathlib import Path
import sys
import polars as pl    # pip install -U polars
import json

def parquet_to_json_array(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)

    for pq_path in src_dir.glob("*.parquet"):
        # 1) Read the Parquet file into a Polars DataFrame
        df = pl.read_parquet(pq_path)

        # 2) Convert to a list of dicts (one dict per row)
        records = df.to_dicts()

        # 3) Write out as one big JSON array, letting json.dumps call str() on datetimes
        out_path = dst_dir / f"{pq_path.stem}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(
                records,
                f,
                ensure_ascii=False,
                indent=2,
                default=str       # ← this makes datetime → ISO string
            )

        print(f"✔ {pq_path.name} → {out_path.name}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Parquet files to JSON arrays")
    parser.add_argument("src", type=str, help="Source directory containing .parquet files")
    parser.add_argument("dst", type=str, help="Destination directory for output .json files")
    args = parser.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not src.is_dir():
        print(f"Error: {src} is not a directory")
        sys.exit(1)

    parquet_to_json_array(src, dst)

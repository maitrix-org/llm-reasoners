#!/usr/bin/env python3
"""
Convert all JSON files in a directory into Parquet using Polars,
flattening only the `extra_info` field to top-level columns to avoid
nested chunked arrays.
"""
import sys
from pathlib import Path
import json
import os

try:
    import polars as pl  # pip install polars
except ImportError:
    print("Error: polars is required. Install via `pip install polars`.")
    sys.exit(1)


def main():
    # Input and output directories
    input_dir = Path(
        "/mnt/weka/home/shibo.hao/yutao/Reasoning360/release/data/norm_json"
    ).expanduser().resolve()
    out_parquet = Path(
        "/mnt/weka/home/shibo.hao/yutao/Reasoning360/release/data/parquet"
    ).expanduser().resolve()
    os.makedirs(out_parquet, exist_ok=True)

    for path in input_dir.glob("*.json"):
        if "livecodebench" in path.stem:
            continue
        # Load normalized JSON list
        with path.open("r", encoding="utf-8") as f:
            records = json.load(f)

        if not records:
            print("No records found in", path)
            continue

        # Create Polars DataFrame
        df = pl.DataFrame(records)

        print(df.head())

        # Write out to Parquet with PyArrow backend
        out_path = out_parquet / f"{path.stem}.parquet"
        df.write_parquet(
            out_path,
            use_pyarrow=True,
            row_group_size=len(records)  
        ) 

        print(f"Saved {df.height} rows to {out_path}")


if __name__ == "__main__":
    main()

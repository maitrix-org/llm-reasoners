#!/usr/bin/env python3
"""
Filter a Parquet dataset by pass-rate thresholds with rich output.

Example
-------
python filter_by_pass_rate.py \
    --parquet_in  webinstruct_le30.parquet \
    --parquet_out_dir ./filtered_parquets \
    --json_out    webinstruct_le30_by_idx.json \
    --low_json    low_pass_rates.json \
    --high_json   high_pass_rates.json \
    --low_thresh  0.1 \
    --high_thresh 0.9
"""
import argparse
import json
import os
from pathlib import Path

from datasets import Dataset, load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

def build_idx_map(ds: Dataset) -> dict[str, dict]:
    """Return a mapping {idx: row_dict} where idx comes from row['extra_info'] keys."""
    idx_map: dict[str, dict] = {}
    for row in ds:
        extra = row.get("extra_info", {})
        for field in ("idx", "index", "id"):
            if field in extra:
                idx_map[str(extra[field])] = row
                break
    return idx_map


def load_idx_set(json_path: Path, low_thresh: float = None, high_thresh: float = None) -> set[str]:
    """Return the set of idx keys to drop based on thresholds in the JSON mapping."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    drop_set: set[str] = set()
    for key, rate in data.items():
        if low_thresh is not None and rate <= low_thresh:
            drop_set.add(str(key))
        if high_thresh is not None and rate >= high_thresh:
            drop_set.add(str(key))
    return drop_set


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter Parquet by pass-rate thresholds with rich output"
    )
    parser.add_argument("--parquet_in",      required=True, help="Input Parquet file")
    parser.add_argument("--parquet_out_dir", required=True, help="Directory to save filtered Parquets")
    parser.add_argument("--json_out",               help="Optional: dump idx→row JSON for inspection")
    parser.add_argument("--low_json",       required=True, help="JSON idx→pass_rate for low-pass filters")
    parser.add_argument("--high_json",      required=True, help="JSON idx→pass_rate for high-pass filters")
    parser.add_argument("--low_thresh",     type=float, required=True, help="Drop if pass_rate <= this")
    parser.add_argument("--high_thresh",    type=float, required=True, help="Drop if pass_rate >= this")
    args = parser.parse_args()

    console.rule("[bold blue] Loading Parquet Dataset")
    ds = load_dataset("parquet", data_files=args.parquet_in, split="train")
    total = len(ds)
    console.print(f"Loaded [bold]{total:,}[/bold] rows from [cyan]{args.parquet_in}[/cyan]")

    console.rule("[bold blue] Building Index Map")
    idx2row = build_idx_map(ds)
    console.print(f"Built map with [yellow]{len(idx2row):,}[/yellow] entries")

    if args.json_out:
        console.print(f"Writing JSON output to [cyan]{args.json_out}[/cyan]")
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(idx2row, f, ensure_ascii=False, indent=2)

    console.rule("[bold blue] Loading Pass-Rate Filters")
    low_drop  = load_idx_set(Path(args.low_json),  low_thresh=args.low_thresh)
    high_drop = load_idx_set(Path(args.high_json), high_thresh=args.high_thresh)
    drop_set  = low_drop | high_drop

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Filter")
    table.add_column("Threshold", justify="right")
    table.add_column("# IDs to Drop", justify="right")
    table.add_row("Low Pass", f"<= {args.low_thresh}", f"{len(low_drop):,}")
    table.add_row("High Pass", f">= {args.high_thresh}", f"{len(high_drop):,}")
    table.add_row("Total Drop", "-", f"{len(drop_set):,}")
    console.print(table)

    console.rule("[bold blue] Filtering Rows")
    kept_rows = [row for idx, row in idx2row.items() if idx not in drop_set]
    console.print(f"Kept [bold green]{len(kept_rows):,}[/bold green] rows out of [bold]{len(idx2row):,}[/bold]")

    console.rule("[bold blue] Saving Filtered Dataset")
    # construct output file name based on input name and thresholds
    base = Path(args.parquet_in).stem
    out_name = f"{base}_l{args.low_thresh}_h{args.high_thresh}.parquet"
    out_path = Path(args.parquet_out_dir) / out_name
    os.makedirs(args.parquet_out_dir, exist_ok=True)

    Dataset.from_list(kept_rows).to_parquet(str(out_path))
    console.print(f"[bold green]Saved[/bold green] filtered dataset to [cyan]{out_path}[/cyan]")

if __name__ == "__main__":
    main()

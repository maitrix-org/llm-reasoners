#!/usr/bin/env python3
"""
Annotate a Parquet dataset by adding two columns per row:
- `low_pass_rate`: pass rate from the low_json mapping (default 0)
- `high_pass_rate`: pass rate from the high_json mapping (default 0)

Example
-------
python annotate_pass_rates.py \
    --parquet_in  webinstruct_le30.parquet \
    --low_json     low_pass_rates.json \
    --high_json    high_pass_rates.json \
    --parquet_out  webinstruct_with_rates.parquet
"""
import argparse
import json
from pathlib import Path
import os
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.table import Table

console = Console()

def load_pass_rates(json_path: Path) -> dict[str, float]:
    """
    Load a JSON mapping of idx→pass_rate.
    Returns an empty dict on missing file.
    """
    if not json_path.exists():
        console.print(f"[yellow]Warning:[/] Pass-rate file not found: {json_path}")
        return {}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # ensure keys are strings
        return {str(k): float(v) for k, v in data.items()}
    except Exception as e:
        console.print(f"[red]Error loading {json_path}:[/] {e}")
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate Parquet with low/high pass rates"
    )
    parser.add_argument(
        "--parquet_in",  required=True, help="Input Parquet file"
    )
    parser.add_argument("--parquet_out_dir", required=True, help="Directory to save filtered Parquets")
    parser.add_argument(
        "--low_json",     required=True, help="JSON mapping idx→pass_rate for lows"
    )
    parser.add_argument(
        "--high_json",    required=True, help="JSON mapping idx→pass_rate for highs"
    )
    args = parser.parse_args()

    console.rule("[bold blue] Loading dataset")
    ds = load_dataset("parquet", data_files=args.parquet_in, split="train")
    console.print(f"Loaded [bold]{len(ds):,}[/bold] rows from [cyan]{args.parquet_in}[/cyan]")

    console.rule("[bold blue] Loading pass-rate JSONs")
    low_rates  = load_pass_rates(Path(args.low_json))
    high_rates = load_pass_rates(Path(args.high_json))
    console.print(f"Low rates: {len(low_rates):,} entries; High rates: {len(high_rates):,} entries")

    console.rule("[bold blue] Annotating rows")
    def annotate(example: dict) -> dict:
        extra = example.get("extra_info", {})

        for field in ("idx", "index", "id"):
            if field in extra:
                idx = field
                break

        idx = str(extra.get(idx, ""))
        # default to 0.0 if not found
        example["low_pass_rate"]  = low_rates.get(idx, 0.0)
        example["high_pass_rate"] = high_rates.get(idx, 0.0)
        return example

    annotated = ds.map(annotate)
    console.print("Annotation complete.")

    console.rule("[bold blue] Saving annotated dataset")
    base = Path(args.parquet_in).stem
    out_name = f"{base}_pr.parquet"
    out_path = Path(args.parquet_out_dir) / out_name
    os.makedirs(args.parquet_out_dir, exist_ok=True)

    annotated.to_parquet(str(out_path))
    print(annotate)
    console.print(f"[bold green]Saved[/bold green] annotated dataset to [cyan]{out_path}[/cyan]")

if __name__ == "__main__":
    main()

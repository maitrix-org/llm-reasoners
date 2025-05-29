#!/usr/bin/env python3
import os
import json
import glob
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from collections import defaultdict
import torch
from datetime import datetime
import argparse
from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.utils.data.dataloader import default_collate

# --------------------------------------------------------------------------- #
# Rich console setup                                                          #
# --------------------------------------------------------------------------- #
custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "highlight": "bold magenta",
    "metric": "bold cyan",
    "time": "bold blue",
})
console = Console(theme=custom_theme)

# --------------------------------------------------------------------------- #
# Helper: make anything JSON serialisable                                     #
# --------------------------------------------------------------------------- #
def json_default(obj):
    """Fallback encoder for json.dump."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.dim() == 0 else obj.tolist()
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.item() if np.ndim(obj) == 0 else obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")

# --------------------------------------------------------------------------- #
# Preserves dictionary and list formats without converting to tensors         #
# --------------------------------------------------------------------------- #
def custom_collate_fn(batch):
    """
    Custom collate function that preserves dictionary and list formats without converting to tensors
    """
    elem = batch[0]
    if isinstance(elem, dict):
        # For dictionaries, process each key separately
        result = {}
        for key in elem:
            values = [d[key] for d in batch]
            # Recursively process values for each key
            result[key] = custom_collate_fn(values)
        return result
    elif isinstance(elem, list):
        # For lists, return original list directly
        return batch
    elif isinstance(elem, tuple):
        # For tuples, process each element separately
        transposed = zip(*batch)
        result = []
        for samples in transposed:
            result.append(custom_collate_fn(samples))
        return tuple(result)
    else:
        # For other types, use default collate
        try:
            return default_collate(batch)
        except:
            # If default collate fails, return original data
            return batch

# --------------------------------------------------------------------------- #
# Data loading, concatenation, and analysis                                   #
# --------------------------------------------------------------------------- #
def read_json(file_path: str) -> Dict:
    """Read a single JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[error]Error reading {file_path}: {e}[/error]")
        return {}

# --------------------------------------------------------------------------- #
# Extract idx to pass_rate mapping                                            #
# --------------------------------------------------------------------------- #
def extract_idx_to_passrate(output_dir: str, dataset_name: str, model_name: str) -> Dict[str, float]:
    """
    Extract mapping from idx to pass_rate for a specific model and dataset.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        model_name: Name of the model
        
    Returns:
        Dictionary mapping unique idx to pass_rate
    """
    model_dir = os.path.join(output_dir, dataset_name, model_name)
    if not os.path.exists(model_dir):
        console.print(f"[error]Directory not found: {model_dir}[/error]")
        return {}

    idx_to_pass_rate = {}
    seen_idx_set = set()
    dp_dirs = glob.glob(os.path.join(model_dir, "dp*"))
    
    if not dp_dirs:
        console.print(f"[warning]No DP directories found in {model_dir}[/warning]")
        return {}
        
    console.print(f"Found {len(dp_dirs)} DP directories in {model_dir}")
    
    for dp_dir in dp_dirs:
        # Extract dp_rank from directory name
        try:
            dp_rank = int(os.path.basename(dp_dir).replace('dp', ''))
        except ValueError:
            console.print(f"[warning]Could not extract dp_rank from {dp_dir}[/warning]")
            continue

        final_results_path = os.path.join(dp_dir, "final_results.json")
        if not os.path.exists(final_results_path):
            console.print(f"[warning]No final_results.json found in {dp_dir}[/warning]")
            continue
            
        data = read_json(final_results_path)
        if not data or "results" not in data:
            console.print(f"[warning]Invalid results format in {final_results_path}[/warning]")
            continue
            
        # Extract idx to pass_rate mapping
        for key, value in data["results"].items():
            idx = None
            if "extra_info" in value:
                # Check if extra_info contains an index field
                for index_field in ["index", "idx", "id"]:
                    if index_field in value["extra_info"]:
                        idx = value["extra_info"][index_field]
                        break
                
                # Assert that we found an index
                assert idx is not None, f"No index field found in extra_info for sample {key}"
                
                # Assert that source exists in the value
                assert "source" in value, f"No 'source' field found in value for sample {key}"
                
                # Create combined id with source and index
                idx = f"{value['source']}_{idx}"
            
            if idx is not None:
                # Check for duplicate idx
                if idx in seen_idx_set:
                    raise ValueError(f"Duplicate idx '{idx}' found in dataset {dataset_name}, model {model_name}")
                
                seen_idx_set.add(idx)
                pass_rate = value["pass_rate"]
                idx_to_pass_rate[idx] = pass_rate
            else:
                console.print(f"[warning]Missing index in extra_info for sample {key} in {final_results_path}[/warning]")
    
    return idx_to_pass_rate

# --------------------------------------------------------------------------- #
# Generate idx to pass_rate mapping for a dataset                             #
# --------------------------------------------------------------------------- #
def generate_idx_mapping(output_dir: str, dataset_name: str, model_name: str) -> Dict[str, float]:
    """
    Generate idx to pass_rate mapping for a single model in a dataset.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        model_name: Name of the model to process
        
    Returns:
        Dictionary mapping unique idx to pass_rate
    """
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        console.print(f"[error]Dataset directory not found: {dataset_dir}[/error]")
        return {}

    console.print(f"[bold]Processing Model: {model_name}[/bold]")
    
    # Extract idx to pass_rate mapping for this model
    idx_to_pass_rate = extract_idx_to_passrate(output_dir, dataset_name, model_name)
    
    if not idx_to_pass_rate:
        console.print(f"[warning]No valid idx to pass_rate mapping found for {model_name}[/warning]")
        return {}
    
    # Save individual model mapping
    model_dir = os.path.join(dataset_dir, model_name)
    model_mapping_path = os.path.join(model_dir, "idx_to_passrate.json")
    try:
        with open(model_mapping_path, 'w') as f:
            json.dump(idx_to_pass_rate, f, indent=2, default=json_default)
        console.print(f"[success]Saved idx to pass_rate mapping for {model_name} ({len(idx_to_pass_rate)} samples)[/success]")
    except Exception as e:
        console.print(f"[error]Failed to save idx to pass_rate mapping for {model_name}: {e}[/error]")
    
    return idx_to_pass_rate

# --------------------------------------------------------------------------- #
# Combine mappings from multiple datasets                                     #
# --------------------------------------------------------------------------- #
def combine_mappings(output_dir: str, dataset_names: List[str], model_name: str, regenerate: bool = False) -> Dict[str, float]:
    """
    Combine idx to pass_rate mappings from multiple datasets for a single model.
    
    Args:
        output_dir: Base output directory
        dataset_names: List of dataset names to combine
        model_name: Model name to use for all datasets
        regenerate: Whether to regenerate mappings
        
    Returns:
        Combined mapping of unique idx to pass_rate
    """
    combined_mapping = {}
    all_seen_idx = set()
    
    for dataset_name in dataset_names:
        console.print(f"\n[bold]Processing dataset: {dataset_name}[/bold]")
        model_dir = os.path.join(output_dir, dataset_name, model_name)
        
        if not os.path.exists(model_dir):
            console.print(f"[warning]Directory {model_dir} does not exist, skipping[/warning]")
            continue
        
        mapping_path = os.path.join(model_dir, "idx_to_passrate.json")
        idx_to_pass_rate = {}
        
        # Generate or load mapping
        if regenerate:
            console.print(f"[bold]Regenerating mapping for {dataset_name} with {model_name}[/bold]")
            idx_to_pass_rate = generate_idx_mapping(output_dir, dataset_name, model_name)
        elif not os.path.exists(mapping_path):
            console.print(f"[error]Mapping file not found: {mapping_path}[/error]")
            console.print(f"[info]Please run the 'map' command first[/info]")
            return
        else:
            idx_to_pass_rate = read_json(mapping_path)
            console.print(f"[info]Loaded existing mapping for {dataset_name} with {len(idx_to_pass_rate)} samples[/info]")
        
        # Check for duplicate idx across datasets
        for idx in idx_to_pass_rate:
            if idx in all_seen_idx:
                raise ValueError(f"Duplicate idx '{idx}' found across multiple datasets")
            all_seen_idx.add(idx)
        
        # Add to combined mapping
        combined_mapping.update(idx_to_pass_rate)
    
    # Save combined mapping
    combined_dir = os.path.join(output_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    combined_path = os.path.join(combined_dir, f"{model_name}_combined.json")
    try:
        with open(combined_path, 'w') as f:
            json.dump(combined_mapping, f, indent=2, default=json_default)
        console.print(f"[success]Saved combined mapping for {model_name} with {len(combined_mapping)} samples from {len(dataset_names)} datasets[/success]")
    except Exception as e:
        console.print(f"[error]Failed to save combined mapping: {e}[/error]")
    
    return combined_mapping

# --------------------------------------------------------------------------- #
# Analyze dataset difficulty based on idx_to_passrate mapping                 #
# --------------------------------------------------------------------------- #
def analyze_dataset_difficulty(idx_to_pass_rate: Dict[str, float]) -> None:
    """
    Analyze difficulty distribution based on idx_to_passrate mapping.
    
    Args:
        idx_to_pass_rate: Dictionary mapping unique idx to pass_rate
    """
    if not idx_to_pass_rate:
        console.print(f"[error]Empty mapping provided for analysis[/error]")
        return

    # Extract pass rates
    pass_rates = list(idx_to_pass_rate.values())
    
    # Calculate statistics
    mean_pass_rate = float(np.mean(pass_rates))
    median_pass_rate = float(np.median(pass_rates))
    std_pass_rate = float(np.std(pass_rates))
    
    # Print basic statistics
    stats_table = Table(show_header=True, header_style="bold magenta")
    stats_table.add_column("Metric")
    stats_table.add_column("Value")
    
    stats = {
        "Total samples": len(pass_rates),
        "Mean pass rate": f"{mean_pass_rate:.3f}",
        "Median pass rate": f"{median_pass_rate:.3f}",
        "Std pass rate": f"{std_pass_rate:.3f}",
    }
    
    for metric, value in stats.items():
        stats_table.add_row(metric, str(value))
    
    console.print(stats_table)

    # Print difficulty distribution
    dist_table = Table(show_header=True, header_style="bold magenta")
    dist_table.add_column("Difficulty")
    dist_table.add_column("Pass Rate Range")
    dist_table.add_column("Count")
    dist_table.add_column("Percentage")
    
    # Count exact 0s and 1s
    exact_zeros = sum(1 for r in pass_rates if r == 0.0)
    exact_ones = sum(1 for r in pass_rates if r == 1.0)
    
    # Add special categories for exactly 0 and exactly 1
    dist_table.add_row(
        "Impossible",
        "Exactly 0.0",
        str(exact_zeros),
        f"{(exact_zeros / len(pass_rates)) * 100:.1f}%"
    )
    
    # Update bins to exclude exact 0s and 1s
    bins = [(0, 0.2, "Very Hard", True, False), 
           (0.2, 0.4, "Hard", False, False),
           (0.4, 0.6, "Medium", False, False),
           (0.6, 0.8, "Easy", False, False),
           (0.8, 1.0, "Very Easy", False, True)]
    
    for bin_start, bin_end, difficulty, exclude_start, exclude_end in bins:
        # Count items in this bin, respecting exclusions
        count = sum(1 for r in pass_rates if 
                   (bin_start < r if exclude_start else bin_start <= r) and
                   (r < bin_end if exclude_end else r <= bin_end))
        
        percentage = (count / len(pass_rates)) * 100
        
        # Show range notation with appropriate brackets
        range_str = f"{'(' if exclude_start else '['}{bin_start:.1f}-{bin_end:.1f}{')' if exclude_end else ']'}"
        
        dist_table.add_row(
            difficulty,
            range_str,
            str(count),
            f"{percentage:.1f}%"
        )
    
    # Add special category for exactly 1
    dist_table.add_row(
        "Perfect",
        "Exactly 1.0",
        str(exact_ones),
        f"{(exact_ones / len(pass_rates)) * 100:.1f}%"
    )
    
    console.print(dist_table)

def main():
    parser = argparse.ArgumentParser(description="Dataset difficulty analysis tools")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Map command to generate idx to pass_rate mapping
    map_parser = subparsers.add_parser('map', help='Generate idx to pass_rate mapping')
    map_parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')
    map_parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    map_parser.add_argument('--model', type=str, required=True, help='Model name to process')

    # Analyze command with support for multiple datasets
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset difficulty')
    analyze_parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')
    analyze_parser.add_argument('--datasets', type=str, nargs='+', required=True, help='Dataset name(s) to analyze')
    analyze_parser.add_argument('--model', type=str, required=True, help='Model name to analyze')
    analyze_parser.add_argument('--regenerate', action='store_true', help='Regenerate idx to pass_rate mapping')
    analyze_parser.add_argument('--save_combined', action='store_true', help='Save combined mapping (default: False)')

    args = parser.parse_args()

    if args.command == 'map':
        generate_idx_mapping(args.output_dir, args.dataset, args.model)
    elif args.command == 'analyze':
        # Handle multiple datasets
        combined_mapping = {}
        all_seen_idx = set()
        
        for dataset_name in args.datasets:
            console.print(f"\n[bold]Processing dataset: {dataset_name}[/bold]")
            model_dir = os.path.join(args.output_dir, dataset_name, args.model)
            
            if not os.path.exists(model_dir):
                console.print(f"[warning]Directory {model_dir} does not exist, skipping[/warning]")
                continue
            
            mapping_path = os.path.join(model_dir, "idx_to_passrate.json")
            idx_to_pass_rate = {}
            
            # Generate or load mapping
            if args.regenerate:
                console.print(f"[bold]Regenerating mapping for {dataset_name} with {args.model}[/bold]")
                idx_to_pass_rate = generate_idx_mapping(args.output_dir, dataset_name, args.model)
            elif not os.path.exists(mapping_path):
                console.print(f"[error]Mapping file not found: {mapping_path}[/error]")
                console.print(f"[info]Generating mapping for {dataset_name} with {args.model}[/info]")
                idx_to_pass_rate = generate_idx_mapping(args.output_dir, dataset_name, args.model)
            else:
                idx_to_pass_rate = read_json(mapping_path)
                console.print(f"[info]Loaded existing mapping for {dataset_name} with {len(idx_to_pass_rate)} samples[/info]")
            
            # Check for duplicate idx across datasets
            for idx in idx_to_pass_rate:
                if idx in all_seen_idx:
                    console.print(f"[warning]Duplicate idx '{idx}' found across multiple datasets, last occurrence will be used[/warning]")
                all_seen_idx.add(idx)
            
            # Add to combined mapping
            combined_mapping.update(idx_to_pass_rate)
        
        # Save combined mapping if requested
        if args.save_combined and len(args.datasets) > 1:
            # Determine base dataset name by removing "_chunk_XX" from dataset names
            dataset_bases = set()
            for dataset_name in args.datasets:
                if "_chunk" in dataset_name:
                    base_name = dataset_name.split("_chunk")[0]
                    dataset_bases.add(base_name)
                else:
                    dataset_bases.add(dataset_name)
            
            # Use the base name if all datasets have the same base, otherwise use "combined"
            if len(dataset_bases) == 1:
                combined_name = list(dataset_bases)[0]
            else:
                combined_name = "combined"
            
            # Create directory if it doesn't exist
            combined_dir = os.path.join(args.output_dir, combined_name)
            os.makedirs(combined_dir, exist_ok=True)
            
            combined_path = os.path.join(combined_dir, f"{args.model}_combined.json")
            try:
                with open(combined_path, 'w') as f:
                    json.dump(combined_mapping, f, indent=2, default=json_default)
                console.print(f"[success]Saved combined mapping for {args.model} with {len(combined_mapping)} samples to {combined_path}[/success]")
            except Exception as e:
                console.print(f"[error]Failed to save combined mapping: {e}[/error]")
        # Always analyze the mapping (whether single dataset or combined)
        dataset_label = args.datasets[0] if len(args.datasets) == 1 else f"{len(args.datasets)} combined datasets"
        console.print(f"\n[bold]Analyzing {dataset_label} for {args.model}[/bold]")
        analyze_dataset_difficulty(combined_mapping)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
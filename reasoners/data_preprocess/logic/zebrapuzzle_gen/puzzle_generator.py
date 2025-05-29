import threading
import queue
import time
import random
from typing import List, Literal
import concurrent.futures
from generator import generate_puzzle, generate_table, table_to_dict
import multiprocessing
import argparse


# diverse set of configs, rows = num_attributes, cols = num_objects
# The level is a number from 1 to 20, where 1 is the easiest and 20 is the hardest
# minimal_conditions is a boolean indicating whether to use minimal conditions
PUZZLE_CONFIGS = [
    {"rows": 2, "cols": 2, "level": 5, "minimal_conditions": True},
    {"rows": 2, "cols": 4, "level": 12, "minimal_conditions": True},
    {"rows": 2, "cols": 5, "level": 3, "minimal_conditions": False},
    {"rows": 2, "cols": 6, "level": 15, "minimal_conditions": True},
    {"rows": 2, "cols": 8, "level": 7, "minimal_conditions": False},
    {"rows": 2, "cols": 10, "level": 18, "minimal_conditions": True},
    {"rows": 2, "cols": 12, "level": 10, "minimal_conditions": False},
    {"rows": 2, "cols": 15, "level": 20, "minimal_conditions": True},
    
    {"rows": 3, "cols": 2, "level": 14, "minimal_conditions": False},
    {"rows": 3, "cols": 3, "level": 6, "minimal_conditions": True},
    {"rows": 3, "cols": 5, "level": 19, "minimal_conditions": False},
    {"rows": 3, "cols": 7, "level": 11, "minimal_conditions": True},
    {"rows": 3, "cols": 9, "level": 4, "minimal_conditions": False},
    {"rows": 3, "cols": 12, "level": 16, "minimal_conditions": True},
    {"rows": 3, "cols": 15, "level": 9, "minimal_conditions": False},
    
    {"rows": 4, "cols": 2, "level": 17, "minimal_conditions": True},
    {"rows": 4, "cols": 4, "level": 1, "minimal_conditions": False},
    {"rows": 4, "cols": 6, "level": 13, "minimal_conditions": True},
    {"rows": 4, "cols": 8, "level": 20, "minimal_conditions": False},
    {"rows": 4, "cols": 10, "level": 5, "minimal_conditions": True},
    {"rows": 4, "cols": 13, "level": 15, "minimal_conditions": False},
    
    {"rows": 5, "cols": 3, "level": 8, "minimal_conditions": True},
    {"rows": 5, "cols": 5, "level": 18, "minimal_conditions": False},
    {"rows": 5, "cols": 7, "level": 3, "minimal_conditions": True},
    {"rows": 5, "cols": 9, "level": 12, "minimal_conditions": False},
    {"rows": 5, "cols": 11, "level": 7, "minimal_conditions": True},
    {"rows": 5, "cols": 14, "level": 19, "minimal_conditions": False},
    
    {"rows": 6, "cols": 2, "level": 10, "minimal_conditions": False},
    {"rows": 6, "cols": 4, "level": 6, "minimal_conditions": True},
    {"rows": 6, "cols": 6, "level": 16, "minimal_conditions": False},
    {"rows": 6, "cols": 8, "level": 2, "minimal_conditions": True},
    {"rows": 6, "cols": 10, "level": 14, "minimal_conditions": False},
    {"rows": 6, "cols": 12, "level": 9, "minimal_conditions": True},
    {"rows": 6, "cols": 15, "level": 4, "minimal_conditions": False},
    
    {"rows": 7, "cols": 3, "level": 11, "minimal_conditions": True},
    {"rows": 7, "cols": 5, "level": 5, "minimal_conditions": False},
    {"rows": 7, "cols": 7, "level": 17, "minimal_conditions": True},
    {"rows": 7, "cols": 9, "level": 8, "minimal_conditions": False},
    {"rows": 7, "cols": 11, "level": 20, "minimal_conditions": True},
    {"rows": 7, "cols": 13, "level": 3, "minimal_conditions": False},
    
    {"rows": 8, "cols": 2, "level": 15, "minimal_conditions": True},
    {"rows": 8, "cols": 4, "level": 7, "minimal_conditions": False},
    {"rows": 8, "cols": 6, "level": 19, "minimal_conditions": True},
    {"rows": 8, "cols": 8, "level": 10, "minimal_conditions": False},
    {"rows": 8, "cols": 10, "level": 1, "minimal_conditions": True},
    {"rows": 8, "cols": 12, "level": 13, "minimal_conditions": False},
    {"rows": 8, "cols": 14, "level": 6, "minimal_conditions": True},
    
    {"rows": 9, "cols": 3, "level": 18, "minimal_conditions": False},
    {"rows": 9, "cols": 5, "level": 4, "minimal_conditions": True},
    {"rows": 9, "cols": 7, "level": 16, "minimal_conditions": False},
    {"rows": 9, "cols": 9, "level": 9, "minimal_conditions": True},
    {"rows": 9, "cols": 11, "level": 2, "minimal_conditions": False},
    {"rows": 9, "cols": 13, "level": 14, "minimal_conditions": True},
    
    {"rows": 10, "cols": 2, "level": 8, "minimal_conditions": False},
    {"rows": 10, "cols": 4, "level": 20, "minimal_conditions": True},
    {"rows": 10, "cols": 6, "level": 5, "minimal_conditions": False},
    {"rows": 10, "cols": 8, "level": 17, "minimal_conditions": True},
    {"rows": 10, "cols": 10, "level": 10, "minimal_conditions": False},
    {"rows": 10, "cols": 12, "level": 3, "minimal_conditions": True},
    {"rows": 10, "cols": 14, "level": 18, "minimal_conditions": False},
    {"rows": 10, "cols": 15, "level": 20, "minimal_conditions": True},
]

INSTRUCTION = """Solve the following puzzle where you are given {{n_objects}} objects and their attributes like job, nationality etc. 
The list of attributes and their unique values are: {{table}}. The goal is to determine the correct attribute for each object based on the provided clues."""


def worker_function(config_and_id):
    config, worker_id = config_and_id
    rows, cols = config["rows"], config["cols"]
    level = config["level"]
    minimal_conditions = config["minimal_conditions"]
    instance_id = config.get("instance_id", 0)

    print(f"Worker {worker_id}: Starting puzzle generation with {rows}x{cols} table, level {level}, instance {instance_id}")
    
    # Create a table for the puzzle
    table = generate_table(rows, cols)
    
    try:
        # Call the generate_puzzle function with our table
        premises = generate_puzzle(
            table, 
            level=level, 
            minimal_conditions=minimal_conditions,
            max_seconds_for_minimizing=30.0,
            tries=3
        )
        
        ground_truth = table_to_dict(table)
        # zebralogic convention and ground truth in format of n_objects x n_attributes
        puzzle_key = f"puzzle_{config['cols']}x{config['rows']}_level{config['level']}_instance{instance_id}"
        list_attributes = ""
        for row in table:
            list_attributes += f"\n{row[0]}: " + ', '.join(sorted(row[1:]))
        instruction = INSTRUCTION.replace("{{table}}", list_attributes)
        instruction = instruction.replace("{{n_objects}}", str(cols))
        result = {
            "worker_id": worker_id,
            "puzzle_id": puzzle_key,
            "config": config,
            "instruction": instruction,
            "clues": premises,
            "ground_truth": ground_truth,
        }
        print(f"Worker {worker_id}: Successfully generated puzzle")
        return result
        
    except Exception as e:
        print(f"Worker {worker_id}: Error generating puzzle: {e}")
        return None

def run_parallel_generation(total_puzzles = None, num_processes=None):
    """Run puzzle generation in parallel using multiple threads"""
    num_configs = len(PUZZLE_CONFIGS)

    if total_puzzles is None:
        total_puzzles = num_configs

    puzzles_per_config = total_puzzles // num_configs

    extra_puzzles = total_puzzles % num_configs
    expanded_configs = []
    for i, config in enumerate(PUZZLE_CONFIGS):
        repeats = puzzles_per_config + (1 if i < extra_puzzles else 0)
        for j in range(repeats):
            # Create a copy with a unique instance ID
            config_copy = config.copy()
            config_copy['instance_id'] = j
            expanded_configs.append(config_copy)

    if num_processes is None:
        num_processes = min(len(expanded_configs), 
                          concurrent.futures.cpu_count())

    print(f"Generating {total_puzzles} puzzles across {num_configs} configurations using {num_processes} processes")
    
    worker_args = [(config, i) for i, config in enumerate(expanded_configs)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        results = list(executor.map(worker_function, worker_args))
    
    # Collect all results from the queue
    results = [r for r in results if r is not None]
    
    return results

def save_puzzles(results, output_dir="./data/raw", output_file="zebra_puzzles.json"):
    import os
    import json
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate puzzles in parallel.")
    parser.add_argument("--num_puzzles", type=int, default=100, help="Total number of puzzles to generate")
    parser.add_argument("--num_processes", type=int, default=16, help="Number of processes to use for generation")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save generated puzzles")
    parser.add_argument("--output_file", type=str, default="zebra_puzzles.json", help="Filename for the generated puzzles")
    args = parser.parse_args()
    multiprocessing.freeze_support()
    start_time = time.time()
    print("Starting parallel puzzle generation...")
    
    # Run parallel generation with 4 threads (or adjust as needed)
    results = run_parallel_generation(total_puzzles = args.num_puzzles, num_processes=args.num_processes)
    
    # Save the generated puzzles
    save_puzzles(results, output_dir=args.output_dir, output_file=args.output_file)
    
    end_time = time.time()
    print(f"Completed in {end_time - start_time:.2f} seconds")
    print(f"Generated {len(results)} puzzles")
"""Downloads, processes, and saves PrimeIntellect datasets."""

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter
from verl.utils.reward_score.coder1 import code_exec, remote_check_stdio, fuzzy_equal, extract_code_from_string

# Define a constant for the empty/failed example
EMPTY_EXAMPLE = {
    "data_source": None,
    "prompt": None,
    "apply_chat_template": False,
    "ability": None,
    "reward_model": None,
    "extra_info": None
}

def get_datasets(cache_dir: str):
    """
    Loads the PrimeIntellect dataset.
    """
    try:
        dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect", split="train", cache_dir=cache_dir)
        
        # TODO: Remove this line before production - only selecting 100 examples for debugging
        # dataset = dataset.select(range(100))
        
        print(f"Dataset: {len(dataset)} examples")
        return dataset, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def make_map_fn(split: str, data_source: str, verbose: bool) -> callable:
    def process_fn(example, idx):
        # Get the problem description
        prompt = example["problem"]
        
        # Get the solution code
        solution = example["solutions"][0] if example["solutions"] else ""
        solution = extract_code_from_string(solution)
        
        # Process tests
        tests = json.loads(example["tests"])
        
        # Now let's do some filtering
        # 1. Remove examples with no tests
        if not tests:
            print(f"No tests found for example {idx}")
            return EMPTY_EXAMPLE
    
        # 2. Remove examples with "image" in prompt, image typically will be in the following format:
        # <image> or [image]
        if "<image>" in prompt.lower() or "[image]" in prompt.lower():
            print(f"Image found in prompt for example {idx}")
            return EMPTY_EXAMPLE
        
        # 3. Remove examples with no problem description
        # Check if prompt starts with unwanted patterns after removing common prefix
        check_prompt = prompt
        if "Solve the following coding problem using the programming language python:" in check_prompt:
            check_prompt = check_prompt.split("Solve the following coding problem using the programming language python:")[1].lstrip()
            
        if (check_prompt.lower().startswith("example") or 
            check_prompt.lower().startswith("input") or 
            check_prompt.lower().startswith("-----input-----")):
            print(f"Example starts with unwanted pattern for example {idx}")
            return EMPTY_EXAMPLE
        
        # Handle different test types
        if tests[0]["type"] == "function_call":
            # Function call tests
            fn_name = tests[0]["fn_name"]

            inputs = []
            outputs = []
            for test in tests:
                inputs.append(test["input"])
                outputs.append(test["output"])
            
            test_code = f"""\
_inputs = {inputs}
_outputs = {outputs}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
    assert _deep_eq({fn_name}(*i), o[0] if len(o) == 1 else tuple(o))
"""
            
            # Validate the solution
            full_code = f"{solution}\n{test_code}"        
            succ, err = code_exec(full_code)
            if not succ:
                print(f"Test code failed for example {idx}")
                print(f"Error: {err}")
                return EMPTY_EXAMPLE
                
            oracle = json.dumps({"functional": test_code})
            
        elif tests[0]["type"] == "stdin_stdout":
            # STDIN/STDOUT tests
            stdin_list = []
            stdout_list = []
            for test in tests:
                stdin_list.append(test["input"])
                stdout_list.append(test["output"])
            
            # Validate the solution
            with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
                futures = []
                for stdin, stdout in zip(stdin_list, stdout_list):
                    futures.append(
                        executor.submit(
                            remote_check_stdio,
                            solution,
                            stdin,
                            stdout,
                        )
                    )
                for future in as_completed(futures):
                    exec_succ, output, stdin, stdout = future.result()
                    pass_test = exec_succ and fuzzy_equal(output.strip(), stdout.strip(), verbose=verbose)
                    if not pass_test:
                        print(f"Test code failed for example {idx}")
                        if verbose:
                            print(f"Input: {stdin}")
                            print(f"Expected output: {stdout}")
                            print(f"Actual output: {output}")
                        return EMPTY_EXAMPLE
            
            oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
        else:
            print(f"Unknown test type: {tests[0]['type']} for example {idx}")
            return EMPTY_EXAMPLE       
        
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": prompt}
            ],
            "ability": "codegen",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": oracle,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": solution,
                "original_prompt": prompt,
                "dataset": "PrimeIntellect",
                "function_name": fn_name if tests[0]["type"] == "function_call" else ""
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save PrimeIntellect datasets.")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory to save the processed datasets.')
    parser.add_argument('--domain', default="codegen",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="primeintellect",
                        help='Name of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=None,
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print verbose output.')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    train_output_dir = os.path.join(args.data_dir, 'train')
    os.makedirs(train_output_dir, exist_ok=True)
    
    # Load the dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    dataset, _ = get_datasets(cache_dir)

    # Process the dataset
    process_fn = make_map_fn('train', data_source, args.verbose)
        
    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=64)

    # Filter out examples where processing failed
    dataset = dataset.filter(lambda x: x["data_source"] == data_source)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        dataset = dataset.filter(lambda x: length_filter.check(x),)
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Sample the dataset using utility function
    train_dataset = sample_dataset(dataset, args.train_sample_size)

    # Save the dataset using utility function
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=data_source,
        sample_size=len(train_dataset)
    )

    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Train data saved to {train_output_path} ({len(train_dataset)} samples)")
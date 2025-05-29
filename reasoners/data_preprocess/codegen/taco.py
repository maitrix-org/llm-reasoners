"""Downloads, processes, and saves TACO dataset."""

import os
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import datasets
import transformers
from datasets import load_dataset, Dataset

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter
from verl.utils.reward_score.coder1 import (
    code_exec,
    remote_check_stdio,
    fuzzy_equal
)

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
    Loads the TACO dataset.
    """
    try:
        dataset = load_dataset("likaixin/TACO-verified", trust_remote_code=True, split="train", cache_dir=cache_dir)
        print(f"TACO dataset: {len(dataset)} examples")
        return dataset, None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return Dataset.from_list([]), None


def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):
        # Create a default "skip" response with all required fields
        oracle = json.loads(example["input_output"])
        source = example["source"]

        # Skip poorly formatted examples
        if source in ["geeksforgeeks", "leetcode"]:
            return EMPTY_EXAMPLE

        # Skip examples with too short descriptions
        if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
            return EMPTY_EXAMPLE

        # Skip examples with images
        if "image" in example["question"].lower() or "\n![" in example["question"]:
            return EMPTY_EXAMPLE

        # Build prompt
        prompt_pieces = [
            "Solve the programming task below in a Python markdown code block.",
            example["question"].strip(),
        ]
        if example["starter_code"].strip():
            prompt_pieces.append(
                "You will use the following starter code to write the solution to the problem and enclose your code within ```python delimiters."
            )
            prompt_pieces.append(
                f"```python\n{example['starter_code'].strip()}\n```"
            )
        else:
            prompt_pieces.append(
                "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within ```python delimiters."
            )

        # Process oracle based on format
        if "fn_name" in oracle:  # Function-based tests
            fn_name = oracle["fn_name"]
            if source == "leetcode":
                fn_name = "Solution()." + fn_name

            test_code = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""

            if source in ["leetcode", "hackerrank"]:
                test_code += f"    assert _deep_eq({fn_name}(*i), o)"
            elif source == "codewars":
                test_code += f"    assert _deep_eq({fn_name}(*i), o[0])"
            else:
                print(f"Unknown source: {source}")
                return EMPTY_EXAMPLE

            # Verify the solution passes tests
            _check_test = example["solutions"][-1] + "\n" + test_code
            succ, err = code_exec(_check_test)
            if not succ:
                print(f"Test code failed for {source}")
                return EMPTY_EXAMPLE
            
            oracle_json = json.dumps({"functional": test_code})
            
        elif "inputs" in oracle and "outputs" in oracle:  # STDIN/STDOUT tests
            stdin_list, stdout_list = oracle["inputs"], oracle["outputs"]
            if len(stdin_list) == 0:
                return EMPTY_EXAMPLE
            
            # handle list inputs and normalize line endings
            stdin_list = [
                "\n".join(stdin) if isinstance(stdin, list) else stdin 
                for stdin in stdin_list
            ]
            stdout_list = [
                ("\n".join(stdout) if isinstance(stdout, list) else stdout).replace("\r\n", "\n")
                for stdout in stdout_list
            ]

            # Verify the solution passes tests
            with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
                futures = []
                for stdin, stdout in zip(stdin_list, stdout_list):
                    futures.append(
                        executor.submit(
                            remote_check_stdio,
                            example["solutions"][-1],
                            stdin,
                            stdout,
                        )
                    )
                for future in as_completed(futures):
                    exec_succ, output, stdin, stdout = future.result()
                    pass_test = exec_succ and fuzzy_equal(output.strip(), stdout.strip())
                    if not pass_test:
                        print(f"Test code failed for {source}")
                        return EMPTY_EXAMPLE

            oracle_json = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
        else:
            print(f"Unknown ground truth format: {oracle}")
            return EMPTY_EXAMPLE

        # Format the final prompt
        prompt = "\n".join(prompt_pieces)
        
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": prompt}
            ],
            "ability": "codegen",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": oracle_json,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": example["solutions"][0] if example["solutions"] else "",
                "original_prompt": prompt,
                "dataset": "likaixin/TACO-verified",
                "source": source,
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save TACO dataset.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--name', default="taco",
                        help='Name of the dataset.')
    parser.add_argument('--domain', default="codegen",
                        help='Domain of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=None,
                        help='Number of samples to use for training. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    train_output_dir = os.path.join(args.data_dir, 'train')
    test_output_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Load the dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    dataset, _ = get_datasets(cache_dir=cache_dir)

    # Process the dataset
    process_fn = make_map_fn('train', data_source)
    
    dataset = dataset.map(function=process_fn, with_indices=True, num_proc=64)

    # Filter out examples where processing failed
    dataset = dataset.filter(lambda x: x["data_source"] == data_source)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        dataset = dataset.filter(lambda x: length_filter.check(x), num_proc=64)
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Sample the dataset
    dataset = sample_dataset(dataset, args.train_sample_size)

    # Save the datasets
    train_output_path = save_dataset(
        dataset=dataset,
        output_dir=train_output_dir,
        filename_prefix=data_source,
        sample_size=args.train_sample_size
    )

    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Train data saved to {train_output_path} ({len(dataset)} samples)\n")
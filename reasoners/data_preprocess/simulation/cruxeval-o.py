"""Downloads, processes, and saves HumanEval dataset."""

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter
from verl.utils.reward_score.coder1 import code_exec


def get_datasets(cache_dir: str):
    """
    Loads the CruxEval dataset.
    """
    try:
        dataset = load_dataset("cruxeval-org/cruxeval", cache_dir=cache_dir)["test"]
        print(f"CruxEval dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):

        # Extract task ID and prompt
        task_id = example["id"]

        prompt = (
            "You will be given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in ```python and ``` tags, following the examples.\n\n"
            "```python\n"
            f"{example['code'].strip()}\n"
            f"assert f({example['input']}) == ??\n"
            "```\n"
        )

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "simulation",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps({"functional": example["code"].strip()}),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": example[
                    "output"
                ],  # Include the canonical solution as reference
                "original_prompt": prompt,
                "dataset": "cruxeval-o",
                "task_id": task_id,
            },
        }

        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)

        return data

    return process_fn


if __name__ == "__main__":
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save HumanEval dataset.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory to save the processed data files.",
    )
    parser.add_argument("--domain", default="simulation", help="Domain of the dataset.")
    parser.add_argument("--name", default="cruxeval-o", help="Name of the dataset.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to use from dataset. If None, use all samples.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    test_output_dir = os.path.join(args.data_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load the dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    _, dataset = get_datasets(cache_dir=cache_dir)

    # Process the dataset
    process_fn = make_map_fn("test", data_source)

    dataset = dataset.map(function=process_fn, with_indices=True)

    # Filter out examples where processing failed
    dataset = dataset.filter(lambda x: x["data_source"] == data_source)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        dataset = dataset.filter(lambda x: length_filter.check(x))
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Sample the dataset
    dataset = sample_dataset(dataset, args.sample_size)

    # Save the dataset to test directory
    test_output_path = save_dataset(
        dataset=dataset,
        output_dir=test_output_dir,
        filename_prefix=data_source,
        sample_size=len(dataset),
    )

    print(
        f"\nDone! \n"
        f"Data source: {data_source}\n"
        f"Test data saved to {test_output_path} ({len(dataset)} samples)"
    )

import argparse
import json
import os

import datasets
import transformers
from datasets import load_dataset
from tqdm import tqdm

from verl.utils.data_process.filter import LengthFilter
from verl.utils.data_process.utils import (sample_dataset, save_dataset,
                                           set_seed)

"""
python data_preprocess/ood/ifeval.py
"""


def get_datasets(cache_dir: str):
    """
    Loads the IFEval dataset.
    """
    try:
        dataset = load_dataset("google/IFEval", cache_dir=cache_dir)["train"] # they updated the dataset under train, but its still called test
        print(f"IFEval dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


non_numeric_answer = 0
non_arithmetic_question = 0

PromptTemplate = """{{context}}"""


def make_map_fn(split: str, data_source: str) -> callable:

    def process_fn(example, idx):
        prompt = example["prompt"]
        instruction_id_list = example["instruction_id_list"]
        kwargs = example["kwargs"]

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": PromptTemplate.replace("{{context}}", prompt)
                }
            ],
            "ability": "ood",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": kwargs,
            },
            "extra_info": {
                "instruction_id_list": instruction_id_list,
                "prompt": prompt,
            }
        }

        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)

        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, process, and save OOD datasets."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory to save the processed data files.",
    )
    parser.add_argument("--domain", default="ood", help="Domain of the dataset.")
    parser.add_argument("--name", default="ifeval", help="Name of the dataset.")
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

    # Config
    set_seed(args.seed)
    data_source = f"{args.domain}__{args.name}"
    test_output_dir = os.path.join(args.data_dir, "test")

    # Download dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    _, dataset = get_datasets(cache_dir)

    # Process datasets
    process_fn = make_map_fn("test", data_source)

    dataset = dataset.map(function=process_fn, with_indices=True)

    # Filter dataset
    try:
        # length filter
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        dataset = dataset.filter(lambda x: length_filter.check(x))

        # non-numeric answer filter
        dataset = dataset.filter(lambda x: x["reward_model"]["ground_truth"] is not None)
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

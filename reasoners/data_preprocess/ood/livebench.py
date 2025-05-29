import argparse
import json
import os
import datetime

import datasets
import transformers
from datasets import load_dataset
from tqdm import tqdm

from verl.utils.data_process.filter import LengthFilter
from verl.utils.data_process.utils import (sample_dataset, save_dataset,
                                           set_seed)

"""
python data_preprocess/ood/livebench.py
"""


def get_datasets(cache_dir: str):
    """
    Loads the LiveBench dataset.
    """

    test_data_sources = [
        "livebench/reasoning",
        "livebench/data_analysis",
        # "livebench/coding",
        # "livebench/instruction_following",
        # "livebench/math",
        "livebench/language",
    ]

    print(f"Loading the test datasets...")
    test_datasets = {
        os.path.basename(test_data_source.lower()):
            datasets.load_dataset(test_data_source, trust_remote_code=True, split="test", cache_dir=cache_dir)
        for test_data_source in test_data_sources
    }

    return test_datasets
    


non_numeric_answer = 0
non_arithmetic_question = 0

PromptTemplate = """{{context}}"""


def make_map_fn(split: str, data_source: str) -> callable:

    def process_fn(example, idx):
        turns = example["turns"]
        if len(turns) != 1:
            print(f"⚠️ Warning: {len(turns)} turns found in {example['task']}")
        ground_truth = example["ground_truth"]
        date = example["livebench_release_date"]

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": PromptTemplate.replace("{{context}}", turns[0])
                }
            ],
            "ability": "ood",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": ground_truth,
            },
            "extra_info": {
                "category": example["category"],
                "task": example["task"],
                "prompt": turns,
                "date": date,
            }
        }

        if idx == 0 or idx == 1:
            print(data["prompt"][0]["content"])
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
    parser.add_argument("--name", default="livebench", help="Name of the dataset.")
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
    test_datasets = get_datasets(cache_dir)

    # Process datasets
    process_fn = make_map_fn("test", data_source)
    for test_data_source, test_data in test_datasets.items():
        dataset = test_data.map(function=process_fn, with_indices=True)
        # filter date before 2024-6 and after 2024-07
        dataset = dataset.filter(lambda x: x["extra_info"]["date"] >= datetime.datetime(2024, 6, 1) and x["extra_info"]["date"] <= datetime.datetime(2024, 7, 31))
        # Filter dataset
        try:
            # length filter
            tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
            length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
            dataset = dataset.filter(lambda x: length_filter.check(x))

            # null answer filter
            dataset = dataset.filter(lambda x: x["reward_model"]["ground_truth"] is not None)
        except Exception as e:
            print(f"Warning: Could not perform length filtering. Error: {e}")
            print("Proceeding without length filtering.")

        # Sample the dataset
        dataset = sample_dataset(dataset, args.sample_size)

        # Save the dataset to test directory
        print(test_data_source, data_source)
        file_prefix = test_data_source.replace("/", "_")
        test_output_path = save_dataset(
            dataset=dataset,
            output_dir=test_output_dir,
            filename_prefix=f"{data_source}_{file_prefix}",
            sample_size=len(dataset),
        )

        print(
            f"\nDone! \n"
            f"Data source: {data_source}\n"
            f"Test data saved to {test_output_path} ({len(dataset)} samples)"
        )

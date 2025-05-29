"""Downloads, processes, and saves math datasets."""

import os
import datasets
from typing import Dict, List, Optional, Any, Union
import enum
import argparse
import pandas as pd
import json
import random
import transformers

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.data_process.prompt import build_zero_style_prompt
from verl.utils.data_process.utils import add_suffix, set_seed, sample_dataset, save_dataset

def get_datasets(cache_dir: str):
    """
    Load the math datasets from Hugging Face Hub.
    
    Args:
        train_data_source (str): Source for training data
        test_data_sources (list): List of sources for test data
        
    Returns:
        tuple: (train_dataset, test_datasets) as Dataset objects
    """
    train_data_source = "SDSB/merged_deduped_dapo_or1_dataset"
    test_data_sources = [
        "nanoverl/minerva",
        "SDSB/aime_repeated_8x",
        "SDSB/amc_repeated_4x",
        "nanoverl/olympiad_bench",
        "nanoverl/math",
        "nanoverl/aime2025_repeated_8x",
    ]

    print(f"Loading the {train_data_source} dataset...")
    train_dataset = datasets.load_dataset(
        train_data_source, trust_remote_code=True, split="train", cache_dir=cache_dir
    )
    
    print(f"Loading the test datasets...")
    test_datasets = {
        os.path.basename(test_data_source.lower()):
            datasets.load_dataset(test_data_source, trust_remote_code=True, split="test", cache_dir=cache_dir)
        for test_data_source in test_data_sources
    }
    
    return train_dataset, test_datasets


def make_map_fn(split: str, data_source: str, reward_metric: str='default') -> callable:
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        # if original_question in example, use it as the question
        question = example.pop("problem")
        # question = example.pop("problem")
        answer = example.pop("answer")
        if isinstance(answer, list):
            answer = answer[0]
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": question + " Please output the final answer within \\boxed{}."},
            ],
            "ability": "math",
            "apply_chat_template": True,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "reward_metric": reward_metric,
                "original_question": question,
            },
        }

        if idx == 0 or idx == 1:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data

    return process_fn
    
    
if __name__ == "__main__":
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to save the processed data files. Will be modified based on other parameters.",
    )
    parser.add_argument(
        "--domain",
        default="math",
        type=str,
        help="Data domain",
    )
    parser.add_argument(
        "--name",
        default="merged_deduped_dapo_or1_dataset",
        type=str,
        help="Dataset name",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=None,
        help="Number of samples to use from the training dataset. If None, use all samples.",
    )
    parser.add_argument(
        "--train-reward-metric",
        type=str,
        default="default",
        help="Reward metric to use for training. If None, use the naive_dapo.compute_score.",
    )
    parser.add_argument(
        "--test-reward-metric",
        type=str,
        default="default",
        help="Reward metric to use for testing. If None, use the naive_dapo.compute_score.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        help="Suffix to add to the dataset name.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when sampling data.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    # Download the datasets from Hugging Face Hub
    cache_dir = datasets.config.HF_DATASETS_CACHE
    train_dataset, test_datasets = get_datasets(cache_dir)
    data_source = f"{args.domain}__{args.name}"

    # Process train dataset
    process_train_fn = make_map_fn("train", data_source, args.train_reward_metric)
    train_data = train_dataset.map(function=process_train_fn, with_indices=True)
    # Sample
    train_data = sample_dataset(train_data, args.train_sample_size)
    # Save
    train_output_dir = os.path.join(args.data_dir, "train")
    args.train_sample_size = len(train_dataset) if args.train_sample_size is None else args.train_sample_size
    train_output_path = save_dataset(
        dataset=train_data,
        output_dir=train_output_dir,
        filename_prefix=data_source + args.suffix if args.suffix else data_source,
        sample_size=args.train_sample_size
    )

    # Process test datasets
    test_output_dir = os.path.join(args.data_dir, "test")
    test_output_paths = []
    for test_data_source, test_data in test_datasets.items():
        test_data_source = f"{args.domain}__{test_data_source}"
        process_fn = make_map_fn("test", test_data_source, args.test_reward_metric)
        test_data = test_data.map(process_fn, with_indices=True)
        test_output_path = save_dataset(
            dataset=test_data,
            output_dir=test_output_dir,
            filename_prefix=test_data_source,
            sample_size=None
        )
        test_output_paths.append(test_output_path)
        print(f"test_data_source: {test_data_source}")
        print(f"Test data saved to {test_output_path}")

    print(f"Done! \n"
          f"Train data saved to {train_output_path}\n"
          f"Test data saved to {test_output_paths}")

    # python dapo_or1_merge_dedup_apr30.py
    
    # with llm judge
    # python dapo_or1_merge_dedup_apr30.py --train-reward-metric math_llm_judge --suffix _llm_judge
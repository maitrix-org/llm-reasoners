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
    train_data_source = "SDSB/big_math_partial_mar21_filtered_basic"
    test_data_sources = [ # hard-coded for now as only bigmath handles all the test datasets
        "nanoverl/minerva",
        "SDSB/aime_repeated_8x",
        "SDSB/amc_repeated_4x",
        "nanoverl/olympiad_bench",
        "nanoverl/math",
    ]

    print(f"Loading the {train_data_source} dataset...")
    train_dataset = datasets.load_dataset(
        train_data_source, trust_remote_code=True, split="train", cache_dir=cache_dir
    )
    
    print(f"Loading the test datasets...")
    test_datasets = [
        datasets.load_dataset(test_data_source, trust_remote_code=True, split="test", cache_dir=cache_dir)
        for test_data_source in test_data_sources
    ]
    
    return train_dataset, test_datasets


def make_map_fn(split: str, data_source: str, prompt_style: str="zero_style", reward_metric: str=None) -> callable:
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question = example.pop("problem")
        answer = example.pop("answer")
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
        default="bigmath_preview_filtered_mar21",
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
        default=None,
        help="Reward metric to use for training. If None, use the naive_dapo.compute_score.",
    )
    parser.add_argument(
        "--test-reward-metric",
        type=str,
        default=None,
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
    train_dataset, _ = get_datasets(cache_dir)
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

    print(f"Done! \n"
          f"Train data saved to {train_output_path}")

    # python bigmath_preview_filtered_mar21.py
    
    # with llm judge
    # python bigmath_preview_filtered_mar21.py --train-reward-metric math_llm_judge --suffix _llm_judge
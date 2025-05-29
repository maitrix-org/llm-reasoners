# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset to parquet format
Code referring to https://github.com/agentica-project/deepscaler/blob/main/scripts/data/deepscaler_dataset.py
"""

import os
import datasets
from typing import Dict, Any
import argparse

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.prompt import build_zero_style_prompt
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

def get_datasets(cache_dir: str):
    """
    Load the math datasets from Hugging Face Hub.
    
    Args:
        train_data_source (str): Source for training data
        test_data_sources (list): List of sources for test data
        
    Returns:
        tuple: (train_dataset, test_datasets) as Dataset objects
    """
    train_data_source = "agentica-org/DeepScaleR-Preview-Dataset"
    print(f"Loading the {train_data_source} dataset...")
    train_dataset = datasets.load_dataset(
        train_data_source, trust_remote_code=True, split="train", cache_dir=cache_dir
    )

    return train_dataset, None


def make_map_fn(split: str, data_source: str) -> callable:
    """
    Create a mapping function for processing dataset examples.
    
    Args:
        split (str): Dataset split ('train' or 'test')
        data_source (str): Source of the data
        
    Returns:
        callable: Function to process individual examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question = example.pop("problem")
        answer = example.pop("answer")
        
        if isinstance(answer, list):
            answer = answer[0]  # minerva, olympiad_bench
            
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": question + " Please output the final answer within \\boxed{}."},
            ],
            "ability": "math",
            "apply_chat_template": True,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, "index": idx, "original_question": question},
        }
        if idx == 0 or idx == 1:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
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
        default="deepscaler_preview",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when sampling data.",
    )
    args = parser.parse_args()
    set_seed(args.seed)
    data_source = f"{args.domain}__{args.name}"

    # Configure data sources
    cache_dir = datasets.config.HF_DATASETS_CACHE
    # Download the datasets from Hugging Face Hub
    train_dataset, test_datasets = get_datasets(cache_dir)

    # Process train dataset
    process_train_fn = make_map_fn("train", data_source)
    train_data = train_dataset.map(function=process_train_fn, with_indices=True)
    
    # Sample
    train_data = sample_dataset(train_data, args.train_sample_size)
    
    # Save
    train_output_dir = os.path.join(args.data_dir, "train")
    args.train_sample_size = len(train_dataset) if args.train_sample_size is None else args.train_sample_size
    train_output_path = save_dataset(
        dataset=train_data,
        output_dir=train_output_dir,
        filename_prefix=data_source,
        sample_size=args.train_sample_size
    )
    print(f"Done! \n"
          f"Train data saved to {train_output_path}")
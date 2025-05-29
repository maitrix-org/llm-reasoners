"""
Preprocess the GPQA dataset to parquet format
"""

import os
import argparse
import random
from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset


def get_datasets():
    """
    Loads the SuperGPQA dataset.
    """
    try:
        dataset = load_dataset("m-a-p/SuperGPQA")["train"]
        print(f"SuperGPQA dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None



def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):        
        def form_options(options: list):
            option_str = 'Options are:\n'
            opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
            for opt, o in zip(options, opts):
                option_str += f'({o}): {opt}\n'
            return option_str
        
        question = example["question"].strip()
        options = form_options(example["options"])
        query = question + '\n' + options + '\n'
        
        # prompt format is adopted from "General Reasoner" in https://github.com/TIGER-AI-Lab/General-Reasoner/blob/main/evaluation/eval_supergpqa.py
        prompt = (
            f"{query}"
            "Please reason step by step, and put your final answer option within \\boxed{}. Only put the letter in the box, e.g. \\boxed{A}. There is only one correct answer."
        )

        
        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": prompt}
            ],
            "ability": "stem",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": example["answer_letter"],
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "original_prompt": prompt,
                "dataset": "m-a-p/SuperGPQA",
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            print(f'\none prompt example is \n{prompt}')
            
        return data

    return process_fn

if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save GPQA dataset.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="stem",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="supergpqa",
                        help='Name of the dataset.')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Number of samples to use from dataset. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    test_output_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Load the dataset
    _, dataset = get_datasets()

    # Process the dataset
    process_fn = make_map_fn('test', data_source)
    
    dataset = dataset.map(function=process_fn, with_indices=True)

    # Sample the dataset
    dataset = sample_dataset(dataset, args.sample_size)
    
    # Save the dataset to test directory
    test_output_path = save_dataset(
        dataset=dataset,
        output_dir=test_output_dir,
        filename_prefix=data_source,
        sample_size=len(dataset)
    )

    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Test data saved to {test_output_path} ({len(dataset)} samples)")
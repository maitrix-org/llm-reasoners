"""Downloads, processes, and saves LeetCode2K datasets."""

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter
from verl.utils.reward_score.coder1 import code_exec

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
    Loads the LeetCode2K dataset.
    """
    try:
        commit_hash = "34803eb64eab1979069ba1f80e7ea474282e28f3"
        train_dataset = load_dataset("newfacade/LeetCodeDataset", 
                                     revision=commit_hash, 
                                     cache_dir=cache_dir)["train"]
        test_dataset = load_dataset("newfacade/LeetCodeDataset", 
                                    revision=commit_hash, 
                                    cache_dir=cache_dir)["test"]
        print(f"Train set: {len(train_dataset)} examples")
        print(f"Test set: {len(test_dataset)} examples")
        return train_dataset, test_dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):
        prefix = example["prompt"]
        
        # Clean up the prompt
        prompt = example["query"]
        prompt = prompt.replace("### Answer: (use the provided format with backticks)", "").strip()
        prompt = prompt.replace("### Format: ", "### Format:\n")

        # Build test code
        test_code = f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
        
        # Extract the candidate solution
        solution = example["completion"]
    
        # Combine all code pieces into a single file to execute
        full_code = f"{prefix}\n{solution}\n{test_code}"
        
        # Validate that the candidate solution passes the tests
        # Skip examples where the test code fails
        succ, err = code_exec(full_code, timeout=20)
        if not succ:
            print(f"Test code failed for example {idx}: {example['meta']['question_title']}")
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
                "ground_truth": json.dumps({
                    "functional": test_code
                }),
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": solution,
                "original_prompt": prompt,
                "prefix": prefix,
                "dataset": "LeetCodeDataset",
                "question_title": example["meta"]["question_title"],
                "difficulty": example["meta"]["difficulty"],
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save LeetCode2K datasets.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="codegen",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="leetcode2k",
                        help='Name of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=None,
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--test-sample-size', type=int, default=None,
                        help='Number of samples to use from test dataset. If None, use all samples.')
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
    train_dataset, test_dataset = get_datasets(cache_dir=cache_dir)

    # Process the dataset
    process_train_fn = make_map_fn('train', data_source)
    process_test_fn = make_map_fn('test', data_source)
    
    train_dataset = train_dataset.map(function=process_train_fn, with_indices=True, num_proc=64)
    test_dataset = test_dataset.map(function=process_test_fn, with_indices=True, num_proc=64)

    # Filter out examples where processing failed
    train_dataset = train_dataset.filter(lambda x: x["data_source"] == data_source)
    test_dataset = test_dataset.filter(lambda x: x["data_source"] == data_source)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        train_dataset = train_dataset.filter(lambda x: length_filter.check(x))
        test_dataset = test_dataset.filter(lambda x: length_filter.check(x))
    except Exception as e:
        print(f"Warning: Could not perform length filtering. Error: {e}")
        print("Proceeding without length filtering.")

    # Sample the datasets using utility function
    train_dataset = sample_dataset(train_dataset, args.train_sample_size)
    test_dataset = sample_dataset(test_dataset, args.test_sample_size)

    # Save the datasets using utility function
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=data_source,
        sample_size=len(train_dataset)
    )
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=data_source,
        sample_size=len(test_dataset)
    )

    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Train data saved to {train_output_path} ({len(train_dataset)} samples)\n"
          f"Test data saved to {test_output_path} ({len(test_dataset)} samples)") 
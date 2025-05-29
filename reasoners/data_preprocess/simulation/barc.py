"""Downloads, processes, and saves BARC (Bootstrapping ARC: Synthetic Problem Generation for ARC Visual Reasoning Tasks) datasets."""

import os
import argparse
import datasets
import numpy as np
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter

import transformers


RawARCPrompt = """You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions.
Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays. Here are the input and output grids for the reference examples:
----------------------------------------
{{training_data}}
----------------------------------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.
----------------------------------------
{{input_test_data}}
----------------------------------------
What is the output grid? Please put your answer within <answer> and </answer> tags, your final answer should be only the output grid (2d array).
"""


def get_datasets(cache_dir, N_samples=10000):
    """
    Downloads (if specified) and loads the ARC-AGI-1 and ARC-AGI-2 datasets.
    """
    # Load the dataset
    ds = datasets.load_dataset("barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems", 
                  split="train", cache_dir=cache_dir) # only have train split, no test split, max 200k samples
    
    ds = ds.select(range(min(N_samples, len(ds))))
    split_dataset = ds.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]

    return train_dataset, test_dataset




def make_map_fn(split: str, data_source: str, prompt_style: str="zero_style") -> callable:
    """
    Creates a mapping function to process individual examples of the dataset.

    Args:
        split: The dataset split ('train' or 'test').
        data_source: Identifier for the data source.
        prompt_style: The style of prompt to use (e.g., 'zero_style').

    Returns:
        A callable function that takes an example and index, and returns the processed data.
    """
    def process_fn(example, idx):
        """
        BARC dataset contains multiple examples, we only use the maximum 6 examples as the training examples.
        """
        max_examples = 6 # max number of examples to show in the prompt
        train_data = example.pop("examples")
        N_pairs = min(len(train_data)-1, max_examples)
        training_data_prompt = ""
        for i in range(N_pairs):
            pair = train_data[i]
            training_data_prompt += f"Example {i+1}\n"
            input_data = [[int(float(x)) for x in row] for row in pair[0]]
            output_data = [[int(float(x)) for x in row] for row in pair[1]]
            training_data_prompt += f"Input: {input_data}\nOutput: {output_data}\n\n"
        raw_prompt = RawARCPrompt.replace("{{training_data}}", training_data_prompt)
        test_pair = train_data[N_pairs]
        input_test_data = [[int(float(x)) for x in row] for row in test_pair[0]]
        output_test_data = [[int(float(x)) for x in row] for row in test_pair[1]]
        raw_prompt = raw_prompt.replace("{{input_test_data}}", str(input_test_data))
        answer = np.array(output_test_data)
        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": raw_prompt,
            }],
            "ability": "reasoning",
            "apply_chat_template": True,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {"split": split, 
                            "index": idx,
                           },
        }
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data

    return process_fn





if __name__ == "__main__":
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Download, process, and save ARC-AGI-2 datasets.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="simulation", help='Domain of the dataset.')
    parser.add_argument('--name', default="barc", help='Name of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=None,
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--test-sample-size', type=int, default=None,
                        help='Number of samples to use from test dataset. If None, use all samples.')
    parser.add_argument('--prompt-style', type=str, choices=['zero_style'], default='zero_style',
                        help='Prompt style to use (currently only zero_style supported).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    set_seed(args.seed)

    data_source = f"{args.domain}__{args.name}"
    train_output_dir = os.path.join(args.data_dir, 'train')
    test_output_dir = os.path.join(args.data_dir, 'test')
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Download the dataset from Github, but still saving to HF_datasets_cache
    cache_dir = datasets.config.HF_DATASETS_CACHE
    train_dataset, test_dataset = get_datasets(cache_dir)

     # Process the dataset
    process_train_fn = make_map_fn('train', data_source, args.prompt_style)
    process_test_fn = make_map_fn('test', data_source, args.prompt_style)
    train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)
    test_dataset = test_dataset.map(function=process_test_fn, with_indices=True)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=4096)
        train_dataset = train_dataset.filter(lambda x: length_filter.check(x), num_proc=64)
        test_dataset = test_dataset.filter(lambda x: length_filter.check(x), num_proc=64)
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

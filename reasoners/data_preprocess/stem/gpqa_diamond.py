"""
Preprocess the GPQA - Diamond dataset to parquet format
"""

import os
import re
import argparse
import random
from datasets import load_dataset, Dataset

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset


def get_datasets():
    """
    Loads the GPQA - Diamond dataset.
    """
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        print(f"GPQA - Diamond dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


# adopted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/zeroshot/utils.py
def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):
        # Create a default "skip" response with all required fields
        question = example["Question"].strip()
        correct = preprocess(example["Correct Answer"])
        incorrect1 = preprocess(example["Incorrect Answer 1"])
        incorrect2 = preprocess(example["Incorrect Answer 2"])
        incorrect3 = preprocess(example["Incorrect Answer 3"])

        all_choices = [incorrect1, incorrect2, incorrect3, correct]
        random.shuffle(all_choices)

        correct_index = all_choices.index(correct)
        correct_letter = chr(65 + correct_index)

        formatted_choices = ""
        for i, choice in enumerate(all_choices):
            letter = chr(65 + i)
            formatted_choices += f"{letter}) {choice}\n"
        
        # # deepseek uses OpenAI's simple-eval for GPQA-Diamond, so we adopt prompts from here: https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
        # prompt = (
        #     f"Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."
        #     f"\n{question}\n"
        #     f"{formatted_choices}"
        # )
        prompt = (
            f"{question}\n"
            f"{formatted_choices}"
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
                "ground_truth": correct_letter,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "original_prompt": prompt,
                "dataset": "Idavidrein/gpqa",
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data["prompt"][0]["content"])
            # print(f'\none prompt example is \n{prompt}')
            
        return data

    return process_fn

if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save GPQA - Diamond dataset.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="stem",
                        help='Domain of the dataset.')
    parser.add_argument('--name', default="gpqa_diamond_no_box",
                        help='Name of the dataset.')
    parser.add_argument('--sample-size', type=int, default=None,
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
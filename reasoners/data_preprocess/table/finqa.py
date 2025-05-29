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
python data_preprocess/table/finqa.py
"""


def get_datasets(cache_dir: str):
    """
    Loads the FinQA dataset.
    """
    try:
        dataset = load_dataset("nanoverl/finqa", cache_dir=cache_dir)["test"]
        print(f"FinQA dataset: {len(dataset)} examples")
        return None, dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


non_numeric_answer = 0
non_arithmetic_question = 0

InstructionFollow = "Please output the final numeric answer within \\boxed{}."
PromptTemplate = """You are given some text and tables. Use the information in the given context to answer the following question.
{{context}}
The question is:
{{question}}
"""


def make_map_fn(split: str, data_source: str) -> callable:

    def process_fn(example, idx):
        pre_text = example["pre_text"]
        post_text = example["post_text"]
        table = example["table"]
        question = example["question"]
        answer = example["answer"]

        # custom table
        tablerows = []
        for idx, row in enumerate(table):
            if idx == 0:
                tablerows.append("| " + " | ".join(row) + " |")
                tablerows.append("| " + " | ".join(["---"] * len(row)) + " |")
            else:
                tablerows.append("| " + " | ".join(row) + " |")

        table_str = "\n".join(tablerows)
        context_str = (
            "\n".join(pre_text) + "\n" + table_str + "\n" + "\n".join(post_text)
        )

        # filter answer which is not numeric
        nanswer = (
            answer.replace(",", "")
            .replace("%", " / 100")
            .replace("$", "")
            .replace(":", "/")
        )

        try:
            nanswer = float(eval(nanswer))
            # print(nanswer)
        except:
            nanswer = None

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": PromptTemplate.replace("{{context}}", context_str).replace("{{question}}", question) + InstructionFollow
                }
            ],
            "ability": "table",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": nanswer,
            },
        }

        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)

        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download, process, and save MultiHierTT table reasoning datasets."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Base directory to save the processed data files.",
    )
    parser.add_argument("--domain", default="table", help="Domain of the dataset.")
    parser.add_argument("--name", default="finqa", help="Name of the dataset.")
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

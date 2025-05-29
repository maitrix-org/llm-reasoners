"""Downloads, processes, and saves HiTab table reasoning datasets."""

import os
import json
import argparse
import random
import transformers
from datasets import Dataset
import datasets
import requests
import zipfile

from verl.utils.data_process.filter import LengthFilter
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.prompt import build_zero_style_prompt


InstructionFollow = "Please output the final answer within \\boxed{}. If there are multiple answers, please output them separated by |."
PromptTemplate = """You are given one or more tables. Use the information in the table to answer the following question.
{{tables}}
The question is:
{{question}}
"""


class HiTabFilter(LengthFilter):
    """Filter for HiTab dataset to keep only arithmetic questions without text evidence."""
    
    def __init__(self, tokenizer, max_length=2048):
        super().__init__(tokenizer=tokenizer, max_length=max_length)

    def check(self, example):
        # Only keep questions requiring operations on the table
        aggregation = example["aggregation"]
        # TODO: Now comment to include non-operation questions to enlarge the dataset
        # if 'none' in aggregation:
        #     return False

        # Ensure the prompt length is within the specified range
        length_check = super().check(example)
        if not length_check:
            return False
        
        return True


def get_datasets(cache_dir, download=False):
    """
    Load the HiTab dataset from cache or download it if necessary.
    
    Args:
        cache_dir (str): Directory to cache the dataset
        download (bool): Whether to download the dataset if not found
        
    Returns:
        tuple: (train_dataset, test_dataset) as Dataset objects
    """
    data_path = os.path.join(cache_dir, "hitab/")
    if download: 
        if os.path.exists(data_path):
            pass
        else:
            # Base URL for raw content
            def download_file(url, output_path):
                """Download a file from GitHub raw content URL."""
                response = requests.get(url)
                if response.status_code == 200:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"Successfully downloaded: {output_path}")
                else:
                    print(f"Failed to download: {url}")
            base_url = "https://raw.githubusercontent.com/microsoft/HiTab/main/data"

            # Files to download
            files = [
                "tables.zip",
                "train_samples.jsonl",
                "test_samples.jsonl"
            ]

            # Download each file
            for file in files:
                url = f"{base_url}/{file}"
                output_path = os.path.join(data_path, file)
                download_file(url, output_path)
            # Unzip the tables.zip file
            zip_path = os.path.join(data_path, "tables.zip")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
    
    train_path = os.path.join(data_path, "train_samples.jsonl")
    test_path = os.path.join(data_path, "test_samples.jsonl")
    
    def _preprocess_hitab_entry(entry):
        new_entry = {}
        new_entry['id'] = entry['id']
        new_entry['question'] = entry['question']
        new_entry['answer'] = '|'.join([str(answer) for answer in entry['answer']])
        new_entry['aggregation'] = entry['aggregation']
        table_id = entry['table_id']
        table_path = os.path.join(data_path, "tables/raw", f"{table_id}.json")
        with open(table_path, "r") as f:
            table = json.load(f)
        new_entry['table'] = table["texts"]
        return new_entry
    
    with open(train_path, "r") as f:
        train_dataset = [json.loads(line) for line in f]
        train_dataset = [_preprocess_hitab_entry(entry) for entry in train_dataset]
        for idx, entry in enumerate(train_dataset):
            print(entry)
            if idx > 5:
                break

    with open(test_path, "r") as f:
        test_dataset = [json.loads(line) for line in f]
        test_dataset = [_preprocess_hitab_entry(entry) for entry in test_dataset]

    print("Total train samples:", len(train_dataset))
    print("Total test samples:", len(test_dataset))
    
    return Dataset.from_list(train_dataset), Dataset.from_list(test_dataset)


def table_matrix_to_markdown(table_matrix: list[list[str]]) -> str:
    """
    Convert table matrix to markdown format.
    
    Args:
        table_matrix (list[list[str]]): Table matrix
        
    Returns:
        str: Markdown representation of the table
    """
    if not table_matrix or not table_matrix[0]:
        return ""
    
    rows = []
    
    # Header row
    header = "| " + " | ".join(str(cell) for cell in table_matrix[0]) + " |"
    rows.append(header)
    
    # Separator row
    separator = "|" + "|".join("-" * 3 for _ in table_matrix[0]) + "|"
    rows.append(separator)
    
    # Data rows
    for row in table_matrix[1:]:
        formatted_row = "| " + " | ".join(str(cell) for cell in row) + " |"
        rows.append(formatted_row)
    
    return "\n".join(rows)


def make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        try:
            table = example.pop("table")
            table_string = table_matrix_to_markdown(table)
            question = example["question"]
            answer = example["answer"]
            
            prompt = PromptTemplate.replace("{{tables}}", table_string).replace("{{question}}", question) + InstructionFollow
                
        except Exception as e:
            print(e)
            print(table)
            exit()
            
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "ability": "table",
            "apply_chat_template": True,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split, 
                "index": idx,
            },
        }
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download, process, and save Hitab table reasoning datasets.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="table", help='Domain of the dataset.')
    parser.add_argument('--name', default="hitab", help='Name of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=200, 
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--test-sample-size', type=int, default=None,
                        help='Number of samples to use from test dataset. If None, use all samples.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Config
    set_seed(args.seed)
    data_source = f"{args.domain}__{args.name}"
    train_output_dir = os.path.join(args.data_dir, 'train')
    test_output_dir = os.path.join(args.data_dir, 'test')

    # Download dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    train_dataset, test_dataset = get_datasets(cache_dir, download=True)

    # Process datasets
    process_train_fn = make_map_fn('train', data_source)
    process_test_fn = make_map_fn('test', data_source)
    train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)
    test_dataset = test_dataset.map(function=process_test_fn, with_indices=True)
    
    print("Total train samples:", len(train_dataset))
    print("Total test samples:", len(test_dataset))
    for idx, example in enumerate(train_dataset):
        print(example)
        if idx > 5:
            break

    # Filter dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    table_filter = HiTabFilter(tokenizer=tokenizer, max_length=4096)
    train_dataset = train_dataset.filter(lambda x: table_filter.check(x), num_proc=64)
    test_dataset = test_dataset.filter(lambda x: table_filter.check(x), num_proc=64)
    args.train_sample_size = len(train_dataset) if args.train_sample_size is None else args.train_sample_size
    args.test_sample_size = len(test_dataset) if args.test_sample_size is None else args.test_sample_size
    
    # Sample datasets if specified
    train_dataset = sample_dataset(train_dataset, args.train_sample_size)
    test_dataset = sample_dataset(test_dataset, args.test_sample_size)
    
    # Save datasets
    train_output_path = save_dataset(
        dataset=train_dataset,
        output_dir=train_output_dir,
        filename_prefix=data_source,
        sample_size=args.train_sample_size
    )
    test_output_path = save_dataset(
        dataset=test_dataset,
        output_dir=test_output_dir,
        filename_prefix=data_source,
        sample_size=args.test_sample_size
    )
    
    print(f"\nDone! \n"
          f"Data source: {data_source}\n"
          f"Train data saved to {train_output_path} ({len(train_dataset)} samples)\n"
          f"Test data saved to {test_output_path} ({len(test_dataset)} samples)")

"""Downloads, processes, and saves MultiHierTT table reasoning datasets."""

import os
import json
import argparse
import random
import transformers
from datasets import Dataset
import datasets

from verl.utils.data_process.filter import LengthFilter
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.prompt import build_zero_style_prompt


InstructionFollow = "Please output the final answer within \\boxed{}."
PromptTemplate = """You are given one or more tables. Use the information in the tables to answer the following question.
{{tables}}
The question is:
{{question}}
"""


class MultiHierTTFilter(LengthFilter):
    """Filter for MultiHierTT dataset to keep only arithmetic questions without text evidence."""
    
    def __init__(self, tokenizer, max_length=2048):
        super().__init__(tokenizer=tokenizer, max_length=max_length)

    def check(self, example):
        # Only keep arithmetic questions
        # TODO: Now comment to include non-operation questions to enlarge the dataset
        # question_type = example["qa"]["question_type"]
        # if question_type != "arithmetic":
        #     return False
        # Filter out questions that need text evidence
        text_evidence = example["qa"]["text_evidence"]
        if text_evidence != []:
            return False
        # Ensure the prompt length is within the specified range
        length_check = super().check(example)
        if not length_check:
            return False
        
        return True


def get_datasets(cache_dir, download=False):
    """
    Load the MultiHierTT dataset from cache or download it if necessary.
    
    Args:
        cache_dir (str): Directory to cache the dataset
        download (bool): Whether to download the dataset if not found
        
    Returns:
        tuple: (train_dataset, test_dataset) as Dataset objects
    """
    data_path = os.path.join(cache_dir, "multihier/")
    if download: 
        if os.path.exists(data_path):
            pass
        else:
            import gdown

            # Replace with your folder ID or full URL
            url = "https://drive.google.com/drive/folders/1ituEWZ5F7G9T9AZ0kzZZLrHNhRigHCZJ"

            # Download the folder to the cache directory
            gdown.download_folder(url, output=data_path, quiet=False, use_cookies=False)
    
    train_path = os.path.join(data_path, "MultiHiertt_dataset/train.json")
    test_path = os.path.join(data_path, "MultiHiertt_dataset/dev.json")
    
    def _align_format(dataset):
        for entry in dataset:
            entry["qa"]["answer"] = str(entry["qa"]["answer"])
        return dataset
    
    with open(train_path, "r") as f:
        train_dataset = json.load(f)
        train_dataset = _align_format(train_dataset)
    with open(test_path, "r") as f:
        test_dataset = json.load(f)
        test_dataset = _align_format(test_dataset)
    print("Total train samples:", len(train_dataset))
    print("Total test samples:", len(test_dataset))
    
    return Dataset.from_list(train_dataset), Dataset.from_list(test_dataset)


def html_table_to_markdown(table):
    """
    Convert HTML table to markdown format.
    Handles hierarchical tables with colspan attributes.
    
    Args:
        table (str): HTML table string
    
    Returns:
        str: Markdown representation of the table
    """
    import re
    from bs4 import BeautifulSoup
    
    # Parse HTML
    soup = BeautifulSoup(table, 'html.parser')
    
    # Get all rows
    rows = soup.find_all('tr')
    if not rows:
        return ""
    
    # Process each row to determine column structure
    table_data = []
    max_cols = 0
    
    for row in rows:
        row_data = []
        col_idx = 0
        
        for cell in row.find_all(['td', 'th']):
            # Get cell content
            content = cell.get_text().strip()
            
            # Get colspan (default to 1)
            colspan = int(cell.get('colspan', 1))
            
            # Add cell with colspan info
            row_data.append({
                'content': content,
                'colspan': colspan,
                'col_idx': col_idx
            })
            
            col_idx += colspan
        
        max_cols = max(max_cols, col_idx)
        table_data.append(row_data)
    
    # Create markdown table
    markdown_rows = []
    
    # Process each row
    for i, row in enumerate(table_data):
        md_row = [''] * max_cols
        
        # Fill in cells
        for cell in row:
            content = cell['content']
            col_idx = cell['col_idx']
            colspan = cell.get('colspan', 1)
            
            # For cells with colspan > 1, center the content
            if colspan > 1:
                md_row[col_idx] = content
                # Fill in empty cells for the span
                for j in range(1, colspan):
                    if col_idx + j < max_cols:
                        md_row[col_idx + j] = ''
            else:
                md_row[col_idx] = content
        
        # Join cells with pipe separator
        markdown_rows.append('| ' + ' | '.join(md_row) + ' |')
        
        # Add header separator after first row
        if i == 0:
            separator = '| ' + ' | '.join(['---'] * max_cols) + ' |'
            markdown_rows.append(separator)
    
    return '\n'.join(markdown_rows)


def make_map_fn(split: str, data_source: str):
    def process_fn(example, idx):
        try:
            tables = example.pop("tables")
            paragraphs = example.pop("paragraphs")
            question = example["qa"]["question"]
            answer = example["qa"]["answer"]
            table_string = "\n".join([html_table_to_markdown(table) for table in tables])
            
            prompt = PromptTemplate.replace("{{tables}}", table_string).replace("{{question}}", question) + InstructionFollow
                
        except Exception as e:
            print(e)
            print(tables)
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
    parser = argparse.ArgumentParser(description="Download, process, and save MultiHierTT table reasoning datasets.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="table", help='Domain of the dataset.')
    parser.add_argument('--name', default="multihier", help='Name of the dataset.')
    parser.add_argument('--train-sample-size', type=int, default=None, 
                        help='Number of samples to use from training dataset. If None, use all samples.')
    parser.add_argument('--test-sample-size', type=int, default=500,
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
    
    # Filter dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
    table_filter = MultiHierTTFilter(tokenizer=tokenizer, max_length=4096)
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

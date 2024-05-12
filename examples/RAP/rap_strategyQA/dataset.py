import json
import os
import re
import torch as th


def get_examples(folder, split):
    path = os.path.join(folder, f"strategyqa_{split}.json")
    with open(path) as f:
        examples = json.load(f)

    print(f"{len(examples)} {split} examples")
    
    return examples


def get_prompt_examples(path):
    with open(path, 'r') as file:
        examples = file.read()
    return examples


def extract_golden_answer(example):
    return example['answer'].strip()

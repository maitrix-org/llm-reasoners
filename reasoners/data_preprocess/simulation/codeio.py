"""Downloads, processes, and saves CodeIO PyEdu Reasoning datasets."""

import os
import argparse
import json
import random
import time
import transformers
import datasets
from datasets import Dataset

from verl.utils.data_process.prompt import build_zero_style_prompt
from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter

InstructionFollow = "Please output the final answer within ```json```"
RawInputPredictionPrompt = """You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: "input": <your input>, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables' names as specified. Please put your answer in \\boxed{} tags.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}

"""
RawOutputPredictionPrompt = """You are given a question that requires some input and output variables as follows:
{{problem_description}}
The input and output requirements are as follows:
{{io_requirements}}
Given the following {{given_type}}: 
{{given}}
Can you predict the output without writing any code? Please reason and put your final answer in the following json format: "output": <your output>, where <your output> should strictly match the the output requirement as specified. Please put your answer in \\boxed{} tags.
Tip: Here is a reference code snippet for this question. You can refer to this code tov guide your reasoning but not copy spans of code directly.
{{refcode}}

"""
AnswerTemplate = """"{{predict_type}}": {{sol}}"""


def get_datasets(cache_dir, download=False):
    """
    Downloads (if specified) and loads the CodeIO PyEdu Reasoning dataset.
    """
    # Download the dataset
    data_path = os.path.join(cache_dir, "PythonEdu-Reasoning.jsonl")
    if download: 
        if os.path.exists(data_path):
            pass
        else:
            url = "https://huggingface.co/datasets/hkust-nlp/CodeIO-PyEdu-Reasoning-Raw/resolve/main/0_368500_filtered_v2_ds25.sced.jsonl"
            os.system(f'wget -O {data_path} {url}')
    dataset = []
    N_code = 0
    if not os.path.exists(data_path):
        time.sleep(5)
    
    # Build the dataset. Only keep the first "input/output" pair for each code sample for diversity.
    max_entry_to_load = 200000  # hard-code to save processing time
    with open(data_path, "r") as f:
        for line in f:
            if N_code >= max_entry_to_load:
                break
            N_code += 1
            data = json.loads(line)
            common_fields = {k: v for k, v in data.items() if k != "ios"} 
            if data["ios"]:
                io = data["ios"][0]
                if random.random() < 0.5:
                    dataset.append({**common_fields, "input": json.dumps(io["input"]), "output": json.dumps(io["output"]), "given_type": "input", "predict_type": "output"})
                else:
                    dataset.append({**common_fields, "input": json.dumps(io["input"]), "output": json.dumps(io["output"]),  "given_type": "output", "predict_type": "input"})

    N = len(dataset)
    N_train = int(N * 0.1)
    train_dataset = dataset[:N_train]
    test_dataset = dataset[N_train:]
    print(f"Total {N_code} code samples, {N} I/O samples.")
    return Dataset.from_list(train_dataset), Dataset.from_list(test_dataset)


def make_map_fn(split: str, data_source: str) -> callable:
    """
    Creates a mapping function to process individual examples of the dataset.

    Args:
        split: The dataset split ('train' or 'test').
        data_source: Identifier for the data source.

    Returns:
        A callable function that takes an example and index, and returns the processed data.
    """
    def process_fn(example, idx):
        given_type = example.pop("given_type")
        predict_type = example.pop("predict_type")
        if predict_type == "input":
            PromptTemplate = RawInputPredictionPrompt
        else:
            PromptTemplate = RawOutputPredictionPrompt
        prompt = PromptTemplate.replace("{{given_type}}", given_type) + InstructionFollow
        for key in ["problem_description", "io_requirements", given_type, "refcode"]:
            feature = example.pop(key)
            if key in ["input", "output"]:
                prompt = prompt.replace("{{given}}", str(feature))
            else:
                prompt = prompt.replace(f"{{{{{key}}}}}", str(feature))
    
        sol = example.pop(predict_type)
        answer = AnswerTemplate.replace("{{predict_type}}", predict_type)
        answer = answer.replace("{{sol}}", sol)
        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt}],
            "ability": "coding-inference",
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

if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Download, process, and save CodeIO PyEdu Reasoning datasets.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', default="simulation", help='Domain of the dataset.')
    parser.add_argument('--name', default="codeio", help='Name of the dataset.')
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

    # Download the dataset
    cache_dir = datasets.config.HF_DATASETS_CACHE
    train_dataset, test_dataset = get_datasets(cache_dir, download=True)

    # Process the dataset
    process_train_fn = make_map_fn('train', data_source)
    process_test_fn = make_map_fn('test', data_source)
    train_dataset = train_dataset.map(function=process_train_fn, with_indices=True)
    test_dataset = test_dataset.map(function=process_test_fn, with_indices=True)

    # Length filter
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        length_filter = LengthFilter(tokenizer=tokenizer, max_length=2048)
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

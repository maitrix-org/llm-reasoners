"""Downloads, processes, and saves MBPP datasets."""

import os
import argparse
import json
import transformers
import datasets
from datasets import load_dataset, Dataset

from verl.utils.data_process.utils import set_seed, sample_dataset, save_dataset
from verl.utils.data_process.filter import LengthFilter


def get_datasets(cache_dir: str):
    """
    Loads the LiveCodeBench dataset.
    """
    try:
        dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5", cache_dir=cache_dir)
        print(f"Train set: {len(dataset['train'])} examples")
        print(f"Test set: {len(dataset['test'])} examples")
        return dataset["train"], dataset["test"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None


def make_map_fn(split: str, data_source: str) -> callable:
    def process_fn(example, idx):
        # Get the problem description
        problem_desc = example["problem"]
        starter_code = example["starter_code"]
        
        # Process tests
        try:
            tests = json.loads(example["tests"])
            if not tests:
                return {
                    "data_source": None,
                    "prompt": None,
                    "ability": None,
                    "reward_model": None,
                    "extra_info": None
                }
            
            # Process metadata to get function name
            metadata = example["metadata"]
            function_name = metadata.get("func_name")
                
            # Handle different test types
            if tests[0]["testtype"] == "functional":
                if not function_name:
                    return {
                        "data_source": None,
                        "prompt": None,
                        "raw_prompt": None,
                        "ability": None,
                        "reward_model": None,
                        "extra_info": None
                    }
                
                # create the prompt with the starter code 
                # prompt adopted from livecodebench official implementation
                # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
                prompt = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Below is the question:

{problem_desc}

You will use the following starter code to write the solution to the problem and enclose your code within ```python delimiters.
```python
{starter_code}
```"""
                
                # Function call tests
                test_code = f"""\
def check_{function_name}():
"""
                for test in tests:
                    # Parse input string by splitting on '\n'
                    input_parts = test["input"].split('\n')
                    # Create proper comma-separated arguments for the function call
                    input_args = ', '.join(input_parts)
                    # Get the output value
                    output_val = test["output"]
                    
                    test_code += f"""    assert Solution().{function_name}({input_args}) == {output_val}
"""
                    
                test_code += f"""
check_{function_name}()
"""
                
                oracle = json.dumps({"functional": test_code})
                
            elif tests[0]["testtype"] == "stdin":
                # create the prompt, and there will be no starter code
                # prompt adopted from livecodebench official implementation
                # https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py
                prompt = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.

Below is the question:

{problem_desc}

Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."""
                
                # STDIN/STDOUT tests
                stdin_list = []
                stdout_list = []
                for test in tests:
                    stdin_list.append(test["input"])
                    stdout_list.append(test["output"])
                
                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown test type: {tests[0]['testtype']}")
        except Exception as e:
            print(f"Error processing tests for example {idx}: {e}")
            return {
                "data_source": None,
                "prompt": None,
                "ability": None,
                "reward_model": None,
                "extra_info": None
            }

        data = {
            "data_source": data_source,
            "prompt": [
                {"role": "user", "content": prompt}
            ],
            "ability": "codegen",
            "apply_chat_template": True,
            "reward_model": {
                "style": "rule",
                "ground_truth": oracle,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "reference": "",  # No solution data in LiveCodeBench
                "dataset": "LiveCodeBench",
                "function_name": function_name if function_name else "",
                "original_prompt": prompt,
            },
        }
        
        if idx == 0 or idx == 1:
            print("\n" + "=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
            
        return data

    return process_fn


if __name__ == '__main__':
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser(description="Process and save LiveCodeBench datasets.")
    parser.add_argument('--data-dir', default='data',
                        help='Base directory to save the processed data files.')
    parser.add_argument('--domain', type=str, default='codegen',
                        help='Domain of the dataset.')
    parser.add_argument('--name', type=str, default='livecodebench',
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
    train_dataset, test_dataset = get_datasets(cache_dir)

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
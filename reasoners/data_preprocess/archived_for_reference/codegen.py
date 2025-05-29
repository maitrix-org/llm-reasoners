"""
Preprocess LeetCode problems (newfacade/LeetCodeDataset) to parquet format.
Thanks to https://github.com/ganler/code-r1/blob/main/examples/data_preprocess/coder1.py
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets
from rich.rule import Rule

import rich
import matplotlib.pyplot as plt
import datasets

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.coder1 import (
    code_exec,
    remote_check_stdio,
    _ERROR_MSG_PREFIX,
    extract_code_from_string,
    fuzzy_equal
)

# from examples.data_preprocess.code.code_utils import *

WORKDING_DIR = os.path.join(os.environ.get("HOME"), "Reasoning360")

SYSTEM_PROMPT = """You are a helpful programming assistant. \
The user will ask you a question and you as the assistant solve it. \
The assistant first thinks how to solve the task through reasoning and then provides the user with the final answer. \
The reasoning process and answer are enclosed within <think>...</think> and <answer>...</answer> tags, respectively."""
EMPTY_RETURN = {
    "data_source": None,
    "prompt": None,
    "ability": None,
    "reward_model": None,
    "extra_info": None
}
N_TESTSET_PER_DATASET = 0

def minimize_stdio(inputs, outputs, max_n_tests=8):
    stdin_list = []
    stdout_list = []
    for stdin, stdout in zip(inputs, outputs):
        if isinstance(stdin, list):
            stdin = "\n".join(stdin)
        if isinstance(stdout, list):
            stdout = "\n".join(stdout)
        if sys.getsizeof(stdin) > 4 * 1024:
            continue
        stdout.replace("\r\n", "\n")
        stdin_list.append(stdin)
        stdout_list.append(stdout)

    zipped = sorted(zip(stdin_list, stdout_list), key=lambda x: sys.getsizeof(x[0]))

    if not zipped:
        print("No tests found!")
        return [], []

    sorted_stdin, sorted_stdout = zip(*zipped)
    return list(sorted_stdin[:max_n_tests]), list(sorted_stdout[:max_n_tests])


def kodcode():  # Thanks!!! to Zhangchen and Yueqin
    # library requirements?
    rich.print(Rule("Loading KodCode/KodCode-Light-RL-10K..."))
    dataset = load_dataset("KodCode/KodCode-Light-RL-10K")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct")

    def make_map_fn(split):

        def process_fn(example, idx):
            reference_solution = example["solution"]
            test_code = "from solution import *\n" + example["test"].strip()
            # Filtering...
            # + block libs are used in reference solution and test code
            # + block usages are used in reference solution or test code
            # + filter out too long prompts
            # + filter out easy problems
            # + filter out failed unittests
            filter_info = {}
            for lib in BLOCK_LIBS:
                if (
                    f"import {lib}"
                    in reference_solution  # naive import detection; ast then detect would be better
                    or f"from {lib}" in reference_solution
                ):
                    print("===========Blocked lib in solution===========")
                    print(f"reference_solution:")
                    print(reference_solution)
                    print(f"lib: {lib}")
                    print(f"question_id: {example['question_id']}")
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_lib",
                            "detail": lib,
                        },
                    }
            for lib in BLOCK_LIBS:
                if f"import {lib}" in test_code or f"from {lib}" in test_code:
                    print("===========Blocked lib in test===========")
                    print(f"test_code:")
                    print(test_code)
                    print(f"lib: {lib}")
                    print(f"question_id: {example['question_id']}")
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_lib",
                            "detail": lib,
                        },
                    }
            for usage in BLOCK_USAGES:
                if usage in reference_solution:
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_usage",
                            "detail": usage,
                        },
                    }
            for usage in BLOCK_USAGES:
                if usage in test_code:
                    return {
                        **EMPTY_RETURN,
                        "filter_info": {
                            "type": f"blocked_usage",
                            "detail": usage,
                        },
                    }
            if (
                len(tokenizer.encode(example["question"])) > MAX_PROMPT_LENGTH - 200
            ):  # -200 for (approximately) the prompt template extra tokens
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "prompt_too_long", "detail": None},
                }
            if example["gpt_difficulty"] == "easy":
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "easy_problem", "detail": None},
                }
            succ, err = code_exec(code=reference_solution, pytest=test_code)
            if not succ:
                # The above code is using the `rich` library in Python to print a formatted message in the console.
                # The message is in red color and includes the value of `example['conversation_id']`.
                # rich.print(
                #     f"[bold red]Test code failed for {example['question_id']}"
                # )
                print("===========Unittest failed===========")
                print(f"reference_solution:")
                print(reference_solution)
                print(f"test_code:")
                print(test_code)
                print(f"err:")
                print(err)
                return {
                    **EMPTY_RETURN,
                    "filter_info": {"type": "failed_unittests", "detail": None},
                }

            prompt = f"Please solve the programming task below in Python. Code should wrapped in a markdown code block.\n\n{example['question'].strip()}"
            if example["test_info"]:
                prompt += f"\n\nNote that the output function should be {str(example['test_info']).strip()}."

            return {
                "data_source": "codegen-kodcode",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "codegen",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps({"pytest": test_code}),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": reference_solution,
                    "prompt": prompt,
                    "dataset": "KodCode/KodCode-Light-RL-10K",
                    "question_subset": example["subset"],
                    "question_id": example["question_id"],
                    "gpt_difficulty": example["gpt_difficulty"],
                },
                "filter_info": None,
            }

        return process_fn

    dataset = dataset["train"].shuffle(seed=666)

    # Preprocess the dataset
    print("Executing tests to ensure correctness...")
    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        load_from_cache_file=False,
    )
    # Analyze the filter reasons
    filter_counts, filter_counts_fine_block_libs = {}, {}
    for entry in dataset:
        if entry["filter_info"] is None:
            continue
        filter_type = entry["filter_info"]["type"]
        if filter_type is not None:
            # Filter type distribution
            if filter_type not in filter_counts:
                filter_counts[filter_type] = 0
            filter_counts[filter_type] += 1
            # Filter detail distribution
            filter_detail = entry["filter_info"].get("detail", None)
            if filter_detail is not None:
                if filter_detail not in filter_counts_fine_block_libs:
                    filter_counts_fine_block_libs[filter_detail] = 0
                filter_counts_fine_block_libs[filter_detail] += 1
        # entry["filter_info"] = None

    print(f"Filtered samples from KodCode: {filter_counts}")

    plot_hist(
        filter_counts,
        file_path=os.path.join(WORKDING_DIR, "artifacts", "filter_counts.png"),
        title="Filter Sample Distribution from KodCode",
        xlabel="Filter Reason",
        ylabel="Count",
    )
    plot_hist(
        filter_counts_fine_block_libs,
        file_path=os.path.join(
            WORKDING_DIR, "artifacts", "filter_counts_fine_block_libs.png"
        ),
        title="Blocked Library Distribution from KodCode",
        xlabel="Blocked Library",
        ylabel="Count",
    )
    
    print(f"Before filtering, KodCode dataset size: {len(dataset)}")

    dataset = dataset.filter(lambda x: x["data_source"] is not None)
    print(f"Remaining samples from KodCode: {len(dataset)}")

    # pick random 50k examples for RL, otherwise it's too large
    # dataset = dataset.select(range(50000 + N_TESTSET_PER_DATASET))

    # Split into train and test
    # splits = dataset["train"].train_test_split(
    #     test_size=N_TESTSET_PER_DATASET, seed=666
    # )
    splits = dataset.train_test_split(test_size=N_TESTSET_PER_DATASET, seed=666)
    train_dataset = splits["train"].shuffle(seed=666)
    test_dataset = splits["test"]
    return train_dataset, test_dataset


# this dataset is super noisy and needs code execution to verify the tasks
def taco():
    rich.print(Rule("Loading likaixin/TACO-verified..."))
    dataset = load_dataset("likaixin/TACO-verified")["train"]
    

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            oracle = json.loads(example["input_output"])
            source = example["source"]

            # skip poorly formatted examples
            if source in ["geeksforgeeks", "leetcode"]:
                return EMPTY_RETURN

            # too short description
            if len("".join([c for c in example["question"] if c.isalnum()])) < 100:
                return EMPTY_RETURN

            # no image
            if "image" in example["question"].lower() or "\n![" in example["question"]:
                return EMPTY_RETURN

            prompt_pieces = [
                "Solve the programming task below in a Python markdown code block.",
                example["question"].strip(),
            ]
            if example["starter_code"].strip():
                prompt_pieces.append(
                    "Also feel free to reuse/extend the following starter code:"
                )
                prompt_pieces.append(
                    f"```python\n{example['starter_code'].strip()}\n```"
                )

            ##
            ## Customization
            ##
            if "fn_name" in oracle:  # the dataset is too noisy
                fn_name = oracle["fn_name"]
                if source == "leetcode":
                    fn_name = "Solution()." + fn_name

                test_code = f"""\
_inputs = {oracle["inputs"]}
_outputs = {oracle["outputs"]}
import math
def _deep_eq(a, b, tol=1e-5):
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b): return False
        return all(_deep_eq(x, y, tol) for x, y in zip(a, b))
    return a == b

for i, o in zip(_inputs, _outputs):
"""

                if source in ["leetcode", "hackerrank"]:
                    test_code += f"    assert _deep_eq({fn_name}(*i), o)"
                elif source == "codewars":
                    test_code += f"    assert _deep_eq({fn_name}(*i), o[0])"
                else:
                    raise ValueError(f"Unknown source: {source}")

                _check_test = example["solutions"][-1] + "\n" + test_code
                if source in ["leetcode"]:
                    _check_test = PY_IMPORTS + _check_test

                succ, err = code_exec(_check_test)
                if not succ:
                    rich.print(f"[bold red]Test code failed for {source}")
                    print(_check_test)
                    print(err)
                    return EMPTY_RETURN
                oracle = json.dumps({"functional": test_code})
                assert example["starter_code"].strip() != ""
            elif "inputs" in oracle and "outputs" in oracle:
                stdin_list, stdout_list = minimize_stdio(
                    oracle["inputs"], oracle["outputs"]
                )
                if len(stdin_list) == 0:
                    return EMPTY_RETURN

                with ThreadPoolExecutor(
                    max_workers=min(len(stdin_list), 8)
                ) as executor:
                    futures = []
                    for stdin, stdout in zip(stdin_list, stdout_list):
                        futures.append(
                            executor.submit(
                                remote_check_stdio,
                                example["solutions"][-1],
                                stdin,
                                stdout,
                            )
                        )
                    for future in as_completed(futures):
                        exec_succ, output, stdin, stdout = future.result()
                        pass_test = exec_succ and fuzzy_equal(output.strip(), stdout.strip())
                        if not pass_test:
                            rich.print(f"[bold red]Test code failed for {source}")
                            print(example["solutions"][-1])
                            print(f"{exec_succ = }")
                            print(f"{stdin = }", f"{stdout = }")
                            if output.startswith(_ERROR_MSG_PREFIX):
                                print("output = \n", output)
                            else:
                                print(f"{output = }")
                            return EMPTY_RETURN

                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown ground truth format: {oracle}")

            prompt = "\n".join(prompt_pieces)
            return {
                "data_source": "codegen-taco",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "codegen",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": (
                        example["solutions"][0] if example["solutions"] else ""
                    ),
                    "dataset": "likaixin/TACO-verified",
                },
            }

        return process_fn

    dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)
    
    if N_TESTSET_PER_DATASET == 0:
        # Return no test samples if N_TESTSET_PER_DATASET is 0
        test_dataset = datasets.Dataset.from_dict({
            "data_source": [],
            "prompt": [],
            "ability": [],
            "reward_model": [],
            "extra_info": []
        })
        train_dataset = dataset
    else:
        splits = dataset.train_test_split(
            test_size=min(N_TESTSET_PER_DATASET, len(dataset) * 0.1), seed=666
        )
        train_dataset = splits["train"]
        test_dataset = splits["test"]

    print(f"Taco train set: {train_dataset}")
    print(f"Taco test set: {test_dataset}")

    return train_dataset, test_dataset


def leetcode2k():
    rich.print(Rule("Loading LeetCodeDataset..."))
    commit_hash = "34803eb64eab1979069ba1f80e7ea474282e28f3"
    train_dataset = load_dataset("newfacade/LeetCodeDataset", 
                                    revision=commit_hash, 
                                    cache_dir=cache_dir)["train"]
    test_dataset = load_dataset("newfacade/LeetCodeDataset", 
                                revision=commit_hash, 
                                cache_dir=cache_dir)["test"]
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)


    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            prefix = example["prompt"]
            
            prompt = example["query"]
            # remove the "### Answer: (use the provided format with backticks)" part
            prompt = prompt.replace("### Answer: (use the provided format with backticks)", "").strip()
            # adjust the "### Format: " part to be more readable
            prompt = prompt.replace("### Format: ", "### Format:\n")

            # Build test code (as before)
            test_code = f"{example['test']}\n\ncheck({example['entry_point'].strip()})"
            # Extract the candidate solution (the original completion)
            solution = example["completion"]
        
            # Combine all code pieces into a single file to execute.
            full_code = f"{prefix}\n{solution}\n{test_code}"
            
            # Validate that the candidate solution passes the tests
            # 20s timeout as some leetcode tests are slow
            succ, err = code_exec(full_code, timeout=20)
            
            if not succ:
                print("===========Test code failed for LeetCodeDataset===========")
                print("Question:", example["meta"]["question_title"])
                print("Error:", err)
                # Skip the example if the test code fails
                return EMPTY_RETURN

            return {
                "data_source": "codegen-leetcode2k",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "codegen",
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
                    "prompt": prompt,
                    "prefix": prefix,
                    "dataset": "LeetCodeDataset",
                },
            }

        return process_fn

    # filter out empty examples ("reward_model" is None)
    train_dataset = train_dataset.map(
        function=make_map_fn("train"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x["reward_model"] is not None)
    test_dataset = test_dataset.map(
        function=make_map_fn("test"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x["reward_model"] is not None)
    print(f"Leetcode2k train set: {train_dataset}")
    print(f"Leetcode2k test set: {test_dataset}")
    return train_dataset, test_dataset


def humaneval():
    rich.print(Rule("Loading OpenAI HumanEval..."))
    dataset = load_dataset("openai_humaneval")["test"]
    print("HumanEval dataset:", dataset)
    
    def process_fn(example, idx):
        # HumanEval's prompt already contains the function signature and docstring
        prompt = (
            "Write a complete, self-contained Python solution to the following problem. "
            "Your solution must include all necessary imports and the full function definition including "
            "the signature exactly as specified. Do not modify the function signature or docstring.\n\n"
            f"```python\n{example['prompt'].strip()}\n```"
        )
        
        # Extract test code
        test_code = example['test']
        entry_point = example['entry_point']
        
        # Validate that the canonical solution passes the tests
        solution = example['canonical_solution']
        
        # Combine the prompt code + solution + test code to verify it works
        full_code = f"{example['prompt']}\n{solution}\n{test_code}\n\ncheck({entry_point})"
        
        succ, err = code_exec(full_code)
        if not succ:
            print(f"Error in canonical solution for task {example['task_id']}: {err}")
            return EMPTY_RETURN
        
        return {
            "data_source": "codegen-humaneval",
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "ability": "codegen",
            "reward_model": {
                "style": "rule",
                "ground_truth": json.dumps(
                    {"functional": f"{test_code}\n\ncheck({entry_point})"}
                ),
            },
            "extra_info": {
                "split": "test",
                "index": idx,
                "reference": solution,
                "prompt": prompt,
                "dataset": "openai_humaneval",
                "task_id": str(example["task_id"]),
            },
        }
    
    test_dataset = dataset.map(
        function=process_fn, 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x["reward_model"] is not None)
    
    # Return empty train dataset and test dataset
    empty_train = datasets.Dataset.from_dict({
        "data_source": [],
        "prompt": [],
        "ability": [],
        "reward_model": [],
        "extra_info": []
    }) if len(test_dataset) > 0 else datasets.Dataset.from_dict({})
    
    print(f"HumanEval test set: {test_dataset}")
    return empty_train, test_dataset

def mbpp():
    rich.print(Rule("Loading MBPP dataset..."))
    dataset = load_dataset("google-research-datasets/mbpp")
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # rewrite the task_id as it is int
            example["task_id"] = "MBPP/" + str(example["task_id"])
            
            # Create prompt
            prompt = (
                f"{example['text']}\n\n"
                f"Your solution should be a complete, self-contained function in a markdown code block. "
                f"Make sure your solution passes the following test cases:\n"
            )
            
            # Construct test code
            test_code = ""
            if example.get('test_setup_code'):
                test_code += example['test_setup_code'] + "\n\n"
            
            # Add all test assertions
            for assertion in example['test_list'] + example.get('challenge_test_list', []):
                test_code += assertion + "\n"
            
            # Add test cases to prompt
            prompt += f"```python\n{test_code}```"
            prompt += "\n\nPlease do not include the test cases in your solution."
            
            # Validate that the canonical solution passes the tests
            solution = example['code']
            full_code = f"{solution}\n\n{test_code}"
            
            succ, err = code_exec(full_code)
            if not succ:
                print(f"Error in canonical solution for task {example['task_id']}: {err}")
                return EMPTY_RETURN
            
            return {
                "data_source": "codegen-mbpp",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "codegen",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.dumps(
                        {"functional": test_code}
                    ),
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "reference": solution,
                    "prompt": prompt,
                    "dataset": "mbpp",
                    "task_id": str(example["task_id"]),
                },
            }
        
        return process_fn
    
    # Process train and test splits
    train_dataset = dataset["train"].map(
        function=make_map_fn("train"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)
    
    test_dataset = dataset["test"].map(
        function=make_map_fn("test"), 
        with_indices=True,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)
    
    print(f"MBPP train set: {train_dataset}")
    print(f"MBPP test set: {test_dataset}")
    return train_dataset, test_dataset

def primeintellect():
    rich.print(Rule("Loading PrimeIntellect dataset..."))
    dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "primeintellect", split="train")
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # Get the problem description
            prompt = example["problem"]
            
            # Get the solution code
            solution = example["solutions"][0] if example["solutions"] else ""
            solution = extract_code_from_string(solution)
            
            # Process tests
            tests = json.loads(example["tests"])
            if not tests:
                return EMPTY_RETURN
                
            # Handle different test types
            if tests[0]["type"] == "function_call":
                # Function call tests
                fn_name = tests[0]["fn_name"]
                test_code = f"""\
def check_{fn_name}():
"""
                for test in tests:
                    input_args = ", ".join([
                        repr(arg) if isinstance(arg, (str, list, tuple, dict)) else str(arg)
                        for arg in test["input"]
                    ])
                    expected_output = repr(test["output"][0]) if isinstance(test["output"][0], (str, list, tuple, dict)) else test["output"][0]
                    test_code += f"""    assert {fn_name}({input_args}) == {expected_output}
"""
                test_code += f"""
check_{fn_name}()
"""
                
                # Validate the solution
                full_code = f"{solution}\n{test_code}"
                succ, err = code_exec(full_code)
                if not succ:
                    rich.print(f"[bold red]Test code failed for PrimeIntellect example {idx}")
                    print(f"===========Full code===========")
                    print(full_code)
                    print(f"===========Error===========")
                    print(err)
                    return EMPTY_RETURN
                    
                oracle = json.dumps({"functional": test_code})
                
            elif tests[0]["type"] == "stdin_stdout":
                # STDIN/STDOUT tests
                stdin_list = []
                stdout_list = []
                for test in tests:
                    stdin_list.append(test["input"])
                    stdout_list.append(test["output"])
                
                # Validate the solution
                with ThreadPoolExecutor(max_workers=min(len(stdin_list), 8)) as executor:
                    futures = []
                    for stdin, stdout in zip(stdin_list, stdout_list):
                        futures.append(
                            executor.submit(
                                remote_check_stdio,
                                solution,
                                stdin,
                                stdout,
                            )
                        )
                    for future in as_completed(futures):
                        exec_succ, output, stdin, stdout = future.result()
                        pass_test = exec_succ and fuzzy_equal(output.strip(), stdout.strip())
                        if not pass_test:
                            rich.print(f"[bold red]Test code failed for PrimeIntellect example {idx}")
                            print(f"===========Solution===========")
                            print(solution)
                            print(f"===========Input===========")
                            print(stdin)
                            print(f"===========Expected output===========")
                            print(stdout)
                            print(f"===========Actual output===========")
                            print(output)
                            return EMPTY_RETURN
                
                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown test type: {tests[0]['type']}")
            
            return {
                "data_source": "codegen-primeintellect",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "codegen",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": solution,
                    "dataset": "PrimeIntellect",
                    "function_name": fn_name if tests[0]["type"] == "function_call" else None,
                },
            }
        
        return process_fn
    
    # Process train and test splits
    train_dataset = dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)

    # Split into train and test sets
    if N_TESTSET_PER_DATASET == 0:
        # Return no test samples if N_TESTSET_PER_DATASET is 0
        test_dataset = datasets.Dataset.from_dict({
            "data_source": [],
            "prompt": [],
            "ability": [],
            "reward_model": [],
            "extra_info": []
        })
    else:
        splits = train_dataset.train_test_split(
            test_size=min(N_TESTSET_PER_DATASET, len(train_dataset) * 0.1), seed=666
        )
        train_dataset = splits["train"].shuffle(seed=666)
        test_dataset = splits["test"]

    print(f"PrimeIntellect train set: {train_dataset}")
    print(f"PrimeIntellect test set: {test_dataset}")
    return train_dataset, test_dataset

def livecodebench():
    rich.print(Rule("Loading LiveCodeBench dataset..."))
    # Load both train and test splits directly
    train_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5", split="train")
    test_dataset = load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5", split="test")
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # Get the problem description
            problem_desc = example["problem"]
            starter_code = example["starter_code"]
            
            # Create the prompt with the starter code 
            prompt = f"{problem_desc}\n\nComplete the implementation using the provided starter code:\n```python\n{starter_code}\n```\n\nYour solution should implement the method(s) in the Solution class."
            
            # Process tests
            tests = json.loads(example["tests"])
            if not tests:
                return EMPTY_RETURN
            
            # Process metadata to get function name
            metadata = example["metadata"]
            function_name = metadata.get("func_name")
                
            # Handle different test types
            if tests[0]["testtype"] == "functional":
                if not function_name:
                    return EMPTY_RETURN
                
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
                
                # For debugging - print a few examples
                if idx < 20:  
                    print(f"Generated test code for example {idx}:")
                    print(test_code)
                
                oracle = json.dumps({"functional": test_code})
                
            elif tests[0]["testtype"] == "stdin":
                # STDIN/STDOUT tests
                stdin_list = []
                stdout_list = []
                for test in tests:
                    stdin_list.append(test["input"])
                    stdout_list.append(test["output"])

                # For debugging - print a few examples
                if idx < 20: 
                    print(f"Generated test code for example {idx}:")
                    for i in range(len(stdin_list)):
                        print(f"Test {i+1}:")
                        print(f"Input: {stdin_list[i]}")
                        print(f"Output: {stdout_list[i]}")
                
                oracle = json.dumps({"inputs": stdin_list, "outputs": stdout_list})
            else:
                raise ValueError(f"Unknown test type: {tests[0]['testtype']}")
            
            return {
                "data_source": "codegen-livecodebench",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "ability": "codegen",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": oracle,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "prompt": prompt,
                    "reference": "",  # No solution data
                    "dataset": "LiveCodeBench",
                    "function_name": function_name,
                },
            }
        
        return process_fn
    
    # Process train and test datasets
    train_dataset = train_dataset.map(
        function=make_map_fn("train"),
        with_indices=True,
        num_proc=64,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)
    
    test_dataset = test_dataset.map(
        function=make_map_fn("test"),
        with_indices=True,
        num_proc=64,
        load_from_cache_file=False,
    ).filter(lambda x: x['reward_model'] is not None)

    print(f"LiveCodeBench train set: {train_dataset}")
    print(f"LiveCodeBench test set: {test_dataset}")
    return train_dataset, test_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=os.path.join(WORKDING_DIR, "data"))
    parser.add_argument(
        "--dataset_names", default="kodcode", help="comma separated dataset names"
    )

    args = parser.parse_args()

    root_dir = args.root_dir
    dataset_names = args.dataset_names.split(",")

    train_datasets = []
    test_datasets = []

    dataset_map = {
        "kodcode": kodcode,
        "taco": taco,
        "leetcode2k": leetcode2k,
        "humaneval": humaneval,
        "mbpp": mbpp,
        "primeintellect": primeintellect,
        "livecodebench": livecodebench, 
    }
    dataset_makes = [dataset_map[name] for name in dataset_names]
    names = "-".join([make.__name__ for make in dataset_makes])

    for make in dataset_makes:
        train, test = make()
        train_datasets.append(train)
        test_datasets.append(test)

    train_dataset = concatenate_datasets(train_datasets).shuffle(seed=666)
    test_dataset = concatenate_datasets(test_datasets)

    rich.print(Rule("Saving the final dataset"))
    print("Train set:", train_dataset)
    print("Test set:", test_dataset)

    local_dir = os.path.join(
        root_dir, f"codegen-{round(len(train_dataset) / 1000)}k-{names}"
    )
    rich.print(f"[bold green]Saving to {local_dir}...")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
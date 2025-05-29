# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset to parquet format
Code referring to https://github.com/agentica-project/deepscaler/blob/main/scripts/data/deepscaler_dataset.py
"""

import os
import datasets
from typing import Dict, List, Optional, Any, Union
import enum
import argparse
import pandas as pd
import json

from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


class TrainDataset(enum.Enum):
    """Enum for training datasets.

    Contains identifiers for various math problem datasets used during training.
    """

    AIME = "AIME"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    OMNI_MATH = "OMNI_MATH"  # Omni Math
    NUMINA_OLYMPIAD = "OLYMPIAD"  # Unique Olympiad problems from NUMINA
    MATH = "MATH"  # Dan Hendrycks Math Problems
    STILL = "STILL"  # STILL dataset
    DEEPSCALER = "DEEPSCALER"  # DeepScaler (AIME, AMC, OMNI_MATH, MATH, STILL)


class TestDataset(enum.Enum):
    """Enum for testing/evaluation datasets.

    Contains identifiers for datasets used to evaluate model performance.
    """

    AIME = "AIME"  # American Invitational Mathematics Examination
    AMC = "AMC"  # American Mathematics Competition
    MATH = "MATH"  # Math 500 problems
    MINERVA = "MINERVA"  # Minerva dataset
    OLYMPIAD_BENCH = "OLYMPIAD_BENCH"  # Olympiad benchmark problems


"""Type alias for either training or testing dataset types."""
Dataset = Union[TrainDataset, TestDataset]


def load_dataset(dataset: Dataset) -> List[Dict[str, Any]]:
    """Load a dataset from a JSON file.

    Loads and parses a JSON dataset file based on the provided dataset enum.
    The file path is constructed based on whether it's a training or testing dataset.

    Args:
        dataset: A Dataset enum value specifying which dataset to load.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the dataset records.
            Each dictionary represents one example in the dataset.

    Raises:
        ValueError: If the dataset file cannot be found, contains invalid JSON,
            or encounters other file access errors.

    Example:
        >>> load_dataset(TrainDataset.AIME)
        [{'problem': 'Find x...', 'solution': '42', ...}, ...]
    """
    dataset_name = dataset.value.lower()
    data_dir = "train" if isinstance(dataset, TrainDataset) else "test"

    current_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(data_dir, f"{dataset_name}.json")
    file_path = os.path.join(current_dir, file_path)

    if not os.path.exists(file_path):
        raise ValueError(f"Dataset file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in {file_path}")
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Error loading dataset: {exc}") from exc


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="data/deepscaler_preview_zero_style_mar21_filter")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    train_data_source = "SDSB/deepscale_partial_mar21_filtered_basic"
    
    print(f"Loading the {train_data_source} dataset from huggingface...", flush=True)
    test_data_sources = [
        "nanoverl/minerva",
        "SDSB/aime_repeated_8x",
        "SDSB/amc_repeated_4x",
        "nanoverl/olympiad_bench",
        "nanoverl/math",
    ]
    print(f"Loading the {test_data_sources} dataset from huggingface...", flush=True)
    train_dataset = datasets.load_dataset(
        train_data_source, trust_remote_code=True, split="train"
    )
    test_datasets = [
        datasets.load_dataset(test_data_source, trust_remote_code=True, split="test")
        for test_data_source in test_data_sources
    ]

    instruction_following = (
        "Let's think step by step and output the final answer within \\boxed{}."
    )


    prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here.

User: {{prompt}} Please put your answer in \\boxed{} tags.
Assistant: <think>
"""

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):

        def process_fn(example, idx):
            question = example.pop("problem")

            # question = question + " " + instruction_following

            answer = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [],
                "raw_prompt": prompt.replace("{{prompt}}", question),
                "ability": "math",
                "apply_chat_template": False,
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split,
                               "index": idx,
                               "question": question},
            }
            if idx == 0:
                print("=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
                print(data)
            return data

        return process_fn

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_data = train_dataset.map(
        function=make_map_fn("train", train_data_source), with_indices=True
    )
    train_data.to_parquet(os.path.join(local_dir, "train.parquet"))
    print(f"train data size:", len(train_data))
    print(train_data[0])
    for test_data_source, test_data in zip(test_data_sources, test_datasets):
        process_fn = make_map_fn("test", test_data_source)
        test_data = test_data.map(process_fn, with_indices=True)
        dataset_name = os.path.basename(test_data_source.lower())
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f"{dataset_name}.parquet"))
        print(f"test data size: ({dataset_name})", len(test_df))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

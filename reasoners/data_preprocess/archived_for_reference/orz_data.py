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
    parser.add_argument("--local_dir", default="data/orz")
    parser.add_argument("--hdfs_dir", default=None)
    args = parser.parse_args()

    # Load train data from local JSON file
    
    # https://raw.githubusercontent.com/Open-Reasoner-Zero/Open-Reasoner-Zero/refs/heads/main/data/orz_math_57k_collected.json
    train_data_source = "orz_math_57k_collected.json"
    print(f"Loading training data from {train_data_source}...", flush=True)
    
    train_dataset = json.load(open(train_data_source, "r"))
    
    # Convert to Dataset format that's compatible with the rest of the code
    train_data = datasets.Dataset.from_list([{
        "problem": item[0]["value"],
        "answer": item[1]["ground_truth"]["value"]
    } for item in train_dataset])

    print(train_data[0])

    instruction_following = (
        "Let's think step by step and output the final answer within \\boxed{}."
    )

    # Rest of the processing remains the same
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = example.pop("problem")
            question = question + " " + instruction_following
            answer = example.pop("answer")
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }
            if idx == 0:
                print("=" * 10 + f"{data_source} {split} {idx}" + "=" * 10)
                print(data)
            return data
        return process_fn

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # Process train data
    train_data = train_data.map(
        function=make_map_fn("train", train_data_source), with_indices=True
    )
    print(train_data[0])
    train_data.to_parquet(os.path.join(local_dir, "train.parquet"))
    print(f"train data size:", len(train_data))

    # Remove test data processing since we're only working with the training data
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

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
Preprocess the GSM8k dataset to parquet format
"""

import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def extract_solution(solution_str):
    solution = re.search(r"\\boxed\{(.*?)\}", solution_str)
    assert solution is not None
    final_solution = solution.group(1)
    return final_solution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/limo')
    parser.add_argument('--hdfs_dir', default=None)

    args = parser.parse_args()

    data_source = 'GAIR/LIMO'

    dataset = datasets.load_dataset(data_source)

    train_dataset = dataset['train']
    test_dataset = dataset['train']

    instruction_following = "Let's think step by step and output the final answer in \\boxed\{\}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_raw = example.pop('question')

            question = question_raw + ' ' + instruction_following

            solution = example.pop('solution')
            exact_answer = example.pop('answer')
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": exact_answer
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': solution,
                    "question": question, # TODO: use prompt \\boxed later
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train_w_prompt.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test_w_prompt.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)

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


import re

def extract_solution(solution_str, method='strict'):
    """
    Extract the final answer choice from an LLM's response to a multiple-choice GPQA question.
    
    Args:
        solution_str (str): The full text response from the LLM
        method (str): 'strict' for exact format matching, 'flexible' for more lenient matching
        
    Returns:
        str: The extracted answer choice (A, B, C, or D) or None if not found
    """
    assert method in ['strict', 'flexible']

    if method == 'strict':
        # Use regex from OpenAI simple-eval https://github.com/openai/simple-evals/blob/main/gpqa_eval.py
        solution = re.search(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(1)
    elif method == 'flexible':
        answer = re.findall(r"\(([A-D])\)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # No answer choices found in parentheses
            pass
        else:
            invalid_str = ['']
            # Find the last letter that is a valid answer choice
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    
    return final_answer


def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1., extra_info=None, compressed: bool = False):
    """The scoring function for GPQA.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    if compressed:
        from reward_score.utils import _deserialise_extra, _decompress_str
        extra_info = _deserialise_extra(_decompress_str(extra_info))
        ground_truth = _deserialise_extra(_decompress_str(ground_truth["compressed"]))["ground_truth"]
        print(f"ground_truth: {ground_truth}")
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return score
        else:
            return format_score
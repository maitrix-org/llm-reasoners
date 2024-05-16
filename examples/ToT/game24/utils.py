import json
import os
import re
import torch as th
import pandas as pd
from collections import Counter
import sympy
from prompts.game24 import *

"""
Input (x)   : a string of 4 numbers
Output (y)  : a trajectory of 3 steps to reach 24
Reward (r)  : 0 or 1, depending on whether the trajectory is correct
Input Example: 
    1 2 3 4
Output Example: 
    1 + 2 = 3 (left: 3 3 4)
    3 + 3 = 6 (left: 4 6)
    6 * 4 = 24 (left: 24)
    (1 + 2 + 3) * 4 = 24
"""


def read_data(file='24.csv'):
    """
    file: a csv file (fixed)
    """
    # path = os.path.join('prompts', file)
    data = list(pd.read_csv(file)['Puzzles'])
    return data


def get_input(self, idx: int) -> str:
    return self.data[idx]


# def test_output(question: str, output: str):
#     expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
#     numbers = re.findall(r'\d+', expression)
#     problem_numbers = re.findall(r'\d+', question)
#     if sorted(numbers) != sorted(problem_numbers):
#         return {'r': 0}
#     try:
#         # print(sympy.simplify(expression))
#         return {'r': int(sympy.simplify(expression) == 24)}
#     except Exception as e:
#         # print(e)
#         return {'r': 0}

def test_output(question: str, output: str):
    if output is None or '=' not in output:
        return False
    if output.split('=')[1].strip() != '24':
        return False
    expression = output.split('=')[0]
    numbers = re.findall(r'\d+', expression)
    question_numbers = re.findall(r'\d+', question)
    if sorted(numbers) != sorted(question_numbers):
        return False
    try:
        return abs(float(sympy.simplify(expression)) - 24) < 1e-6
    except ValueError:
        return False


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


def correct_left_numbers(x: str, y: str, action: str) -> str:
    ## find the actual left numbers
    original_nums = get_current_numbers(y if y else x).split(' ')
    # print(original_nums)
    original_cnt = Counter(original_nums)
    action = action.strip().lower().split(' (left')[0]
    if ' = ' in action:
        expression, new_num = action.split(' = ')[0], action.strip().lower().split(' = ')[1]
        used_nums = re.findall(r'\d+', expression)
        left_nums = [new_num]
        for num in used_nums:
            if num in original_cnt:
                original_cnt[num] -= 1
        for num in original_cnt:
            if original_cnt[num] > 0:
                for _ in range(original_cnt[num]):
                    left_nums.append(num)
    else:
        print(f'no equation in action: {action}')
        left_nums = re.findall(r'\d+', action)
    correct_action = action + ' (left: ' + ' '.join(left_nums) + ')'
    return correct_action


def propose_prompt_wrap(x: str, y: str = '', all_prompt: dict = {}) -> str:
    current_numbers = get_current_numbers(y if y else x)
    if current_numbers == '24':
        # prompt = all_prompt['cot_prompt'].format(input=x) + 'Steps:\n' + y
        prompt = output_prompt.format(input=x) + 'Steps:' + y
        # print(f"Final propose: {prompt}")
    else:
        # prompt = all_prompt['propose_prompt'].format(input=current_numbers)
        prompt = propose_prompt.format(input=current_numbers)
    return prompt


def value_prompt_wrap(x: str, y: str, all_prompt: dict = {}) -> str:
    last_line = y.strip().split('\n')[-1]
    if 'left: ' not in last_line and last_line != '':  # last step
        ans = last_line.lower().replace('answer: ', '')
        # print([value_last_step_prompt.format(input=x, answer=ans)])
        # return all_prompt['value_last_step_prompt'].format(input=x, answer=ans)
        return value_last_step_prompt.format(input=x, answer=ans)
    current_numbers = get_current_numbers(y)
    # return all_prompt['value_prompt'].format(input=current_numbers)
    return value_prompt.format(input=current_numbers)


def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
    if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
        print("not an answer at step 4")
        return 0
    value_names = [_.split('\n')[-1] for _ in value_outputs]
    value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
    value = sum(value * value_names.count(name) for name, value in value_map.items())
    return value

import json
import os
import re
import torch as th
import pandas as pd
import sympy

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
    path = os.path.join('prompts', file)
    data = list(pd.read_csv(path)['Puzzles'])
    return data

def get_input(self, idx: int) -> str:
    return self.data[idx]

def test_output(data, idx: int, output: str):
    expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
    numbers = re.findall(r'\d+', expression)
    problem_numbers = re.findall(r'\d+', data[idx])
    if sorted(numbers) != sorted(problem_numbers):
        return {'r': 0}
    try:
        # print(sympy.simplify(expression))
        return {'r': int(sympy.simplify(expression) == 24)}
    except Exception as e:
        # print(e)
        return {'r': 0}
    
def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

def propose_prompt_wrap(x: str, y: str='', all_prompt: dict={}) -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = all_prompt['cot_prompt'].format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = all_prompt['propose_prompt'].format(input=current_numbers)
        return prompt

def value_prompt_wrap(x: str, y: str, all_prompt: dict={}) -> str:
    last_line = y.strip().split('\n')[-1]
    if 'left: ' not in last_line:  # last step
        ans = last_line.lower().replace('answer: ', '')
        # print([value_last_step_prompt.format(input=x, answer=ans)])
        return all_prompt['value_last_step_prompt'].format(input=x, answer=ans)
    current_numbers = get_current_numbers(y)
    return all_prompt['value_last_step_prompt'].format(input=current_numbers)
    
def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
    if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
        return 0
    value_names = [_.split('\n')[-1] for _ in value_outputs]
    value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
    value = sum(value * value_names.count(name) for name, value in value_map.items())
    return value
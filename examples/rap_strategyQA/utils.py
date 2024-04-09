import re
from typing import Optional


import json
import os
import re
import torch

def extract_final_answer(sample):
    ans = ''
    if "So the answer is" in sample:
        ## extract answer directly
        # ans_idx = sample.find("So the answer is")
        # ans = re.findall(r'\d+', sample[ans_idx:])
        ans = sample.split('So the answer is')
        if ans:
            ans = ans[-1].strip().split('\n')[0].replace('.', '')
        else:
            ans = ''
    else:
        ## negative word list
        if ' not ' in sample or ' no ' in sample or 'Not ' in sample or 'No ' in sample:
            ans = 'no'
            # print(f"find no: {ans}, {sample}")
        else:
            ans = ''
    return ans

def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r'.*So the answer is (.*)\.$', output)
    if match is None:
        # print(f"No matching answer in output: {output}")
        return None
    answer = match[1]
    return answer

def extract_subquestions(subqs_lm):
    subqs_list = []
    ## keep only the subqs (remove follow up generated examples)
    subqs_str = subqs_lm.split('\n\n')[0].strip()
    if subqs_str[0] == '\n':
        subqs_str = subqs_str[1:]
    # print('\n<<<< sub-questions string >>>>\n{}'.format(subqs_str))
    ## 1. replace all change lines by ', 
    subqs_str = subqs_str.replace('\n', ', ')
    ## 2. try index pattern matching
    pattern = r'\d+\.\s*(.*?\?)'
    subqs_pattern = re.findall(pattern, subqs_str)
    if subqs_pattern:
        subqs_list = subqs_pattern
    else:
        subqs_list = subqs_str.split('", "')
        ## format questions
        for i, subq in enumerate(subqs_list):
            if subq[0] != '"':
                    subqs_list[i] = '"' + subqs_list[i]
            if subq[-1] != '"':
                subqs_list[i] = subqs_list[i] + '"'

    return subqs_list


def judge_answer(output, answer):
    if output is None:
        return False
    return output == answer

import re
from typing import Optional


import json
import os
import re
import torch

def extract_answer(sample):
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


def extract_feedback_answer(sample):
    ans = ''
    ## extract answer directly
    pattern = r'(?<=Therefore, the correct answer is ).*'
    ans = re.findall(pattern, sample)
    if ans:
        ans = re.findall(r'\d+', ans[-1])[-1]
    ## otherwise, check if there is equation
    elif '=' in sample:
        ans = sample.split('=')[-1]
        ans = ans.replace(',', '')
        num_pattern = r"[-+]?(?:\d*\.*\d+)"
        ans = re.findall(num_pattern, ans)
        if ans:
            ans = ans[-1]
        else:
            ans = ''
    return ans


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

def extract_followup_questions(subqs_lm):
    subqs_list = []
    subqs_str = subqs_lm.split(', we still need to know:')[-1].strip()
    if not subqs_str:
        return []
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
    output = re.findall('The answer is .*?([$ .0-9,\\-]+).*\\.', output)
    if len(output) == 0:
        output = ''
    else:
        output = output[-1].replace(',', '').replace('$', '').replace(' ', '')
    ret = output
    if '=' in output:
        output = output.split('=')[-1]
    try:
        output, answer = int(output), int(answer)
    except ValueError:
        try:
            output, answer = float(output), float(answer)
        except ValueError:
            pass
    return ret, output == answer

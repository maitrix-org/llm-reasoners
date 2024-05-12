import re
from collections import Counter
def extract_subquestions(subqs_lm):
    subqs_list = []
    ## keep only the subqs, find the last 'A:'
    subqs_idx = subqs_lm.rfind('A:')
    subqs_lm = subqs_lm[subqs_idx:].split('\n\n')[0]
    # print('\n<<<< all sub-questions >>>>\n{}'.format(subqs_lm))

    ## extract subquestions
    subqs_str = subqs_lm.split(', we need to know: ')[-1].strip()
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
            if len(subq)==0:
                subqs_list[i] = '"'+ '"'
                continue
            if subq[0] != '"':
                subqs_list[i] = '"' + subqs_list[i]
            if subq[-1] != '"':
                subqs_list[i] = subqs_list[i] + '"'

    return subqs_list

def majority_voting(outputs):
    # Return the most common output
    postive_expression = ['yes', 'true', 'correct', 'right']
    negative_expression = ['no', 'false', 'incorrect', 'wrong', 'not', 'none']

    for i, output in enumerate(outputs):
        if output.lower() in postive_expression:
            outputs[i] = 'yes'
        elif output.lower() in negative_expression:
            outputs[i] = 'no'

    return Counter(outputs).most_common(1)[0][0]

def judge_answer(pred, label):
    # pred is yes or no (string)
    # label is true or false (boolean)
    if pred == 'yes':
        return label
    elif pred == 'no':
        return not label

def parse_answer(text):
    match = re.search(r"the answer is (\w+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    else:
        if 'yes' in text.lower() or 'true' in text.lower():
            return 'yes'
        elif 'no' in text.lower() or 'false' in text.lower():
            return 'no'
        
    return ""
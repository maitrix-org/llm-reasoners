import json
import re

def remove_not(x):
    match_number = re.compile('[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?')
    result=re.findall(match_number, x)
    if len(result) !=0:
        return re.split(match_number, x)[-1]
    return None
def parse_not(inputs):
    if not inputs:
        return '',''
    if '\times' in inputs:
        x,ab=inputs.split('\times')
    elif '\\times' in inputs:
        x,ab=inputs.split('\\times')
    elif '*' in inputs:
        x,ab=inputs.split('*')
    else:
        return inputs
    return x,ab

def cal_not(inputs):
    
    try:
        x,ab=list(inputs)
        match_number = re.compile('10\^[{]?\ *-?[0-9]+\ *[}]?')
        ab=re.findall(match_number, ab)[0]
        ab=ab[ab.find('^')+1:]
        if '{' in ab:
            ab=ab[ab.find('{')+1:]
            if '}' in ab:
                ab=ab[:ab.find('}')]
        x=x.strip()
        out=float(x)*10**float(ab)
        # print(float(x)*10**float(ab))
        return str(out)
    except:
        print('error')
    return inputs

def remove_boxed(s):
        left = "oxed{" #change
        try:
            assert s[:len(left)] == left
            assert s[-1] == "}"
            answer = s[len(left):-1]
            if "=" in answer:
                answer = answer.split("=")[-1].lstrip(" ")
            return answer
        except:
            return None
def last_boxed_only_string(string):
        idx = string.rfind("oxed") #change
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx == None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval
def parse_math_answer(raw_string):
    return remove_boxed(last_boxed_only_string(raw_string))


def extract_formulae_reasoning(inp):

    formulae, reasoning = inp.split("\n\n**Reasoning/calculation process:**")[0], inp.split("\n\n**Reasoning/calculation process:**")[1]
    reasoning_final = "\n\n**Reasoning/calculation process:**" + reasoning.split("\n\n**Answer conclusion:**")[0]

    return formulae, reasoning_final

def judge_answer(output, answer, unit):
    if remove_not(unit):
        unit_prob=remove_not(unit)

    model_output = parse_math_answer(output)

    if unit_prob != unit:
        model_output=cal_not(parse_not(model_output))
        answer=cal_not((answer, unit))
    
    if answer is None or model_output is None:
        return False
    if answer in model_output:
        return True
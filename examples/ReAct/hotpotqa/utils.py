import re

def finished_check(buffered_action):
    pattern = r'Finish\[(.*?)\]'
    matches = re.findall(pattern, buffered_action)
    if matches:
        return True

def retrieve_answer(output):
    output = output[-1][-1][0]
    pattern = r'Finish\[(.*?)\]'
    matches = re.findall(pattern, output)
    if matches:
        return matches[0]
    else:
        return None
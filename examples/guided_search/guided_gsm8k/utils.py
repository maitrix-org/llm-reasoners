import re
from typing import Optional
import io
from collections import Counter

def majority_voting(outputs):
    # filter out non-numeric strings
    outputs = [output for output in outputs if isinstance(output, (int, float))]

    # get the output with the highest count
    count = Counter(outputs)
    max_count = max(count.values())
    for output in outputs:
        if count[output] == max_count:
            return output

def get_indent(line):
    match = re.match(r'^(\s*)', line)
    if match:
        return match.group(1)
    else:
        return ''

def construct_full_solution(state, execute=True):
    with io.StringIO() as f:
        f.write("def solution():\n")
        # iterate through the state
        for a, _, _, _, _ in state:
            f.write(f"{a}\n")

        full_output = f.getvalue()

    if execute:
        try:
            # Create a dictionary to serve as the global and local scope for the exec call
            exec_globals = {}

            # Execute the function definition
            exec(full_output, exec_globals)

            # Call the function and get the output
            output = exec_globals['solution']()
            return output
        except Exception as e:
            # return the error message
            return str(e)
    else:
        return full_output


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except:
        pass
    try:
        return output == answer
    except ValueError:
        return False


import re
from typing import Optional


def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


# def retrieve_answer_from_dataset(answer: str) -> str:
#     return re.match(r'[\S\s]*#### (.*)$', answer)[1]

def retrieve_answer_from_dataset(solution: str) -> str:
    # answer = re.search(r'\\boxed{(.*)}', solution)
    answer = re.search(r'\\boxed{(.*?)}\$', solution)
    if answer is not None:
        return answer[1]
    else:
        return "Answer pattern not found in the solution."



def judge_answer(output: Optional[str], answer: str) -> bool:
    if output is None:
        return False
    try:
        output = int(output)
        answer = int(answer)
        return output == answer
    except ValueError:
        pass
    try:
        output = float(output)
        answer = float(answer)
        return output == answer
    except ValueError:
        pass
    return output == answer

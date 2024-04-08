import re
from typing import Optional, Union
from reasoners.algorithm import BeamSearchResult

from reasoners.base import AlgorithmOutput


def retrieve_answer(output: Union[list, str, AlgorithmOutput]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, AlgorithmOutput):
        if (result := getattr(output, 'aggregated_result', None)) is not None:
            return result
        output = output.terminal_state
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*The answer is .*?([ $.0-9,\-=]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer

def retrieve_answer_bs(output: Union[list, str, BeamSearchResult]) -> Optional[str]:

    if isinstance(output, BeamSearchResult):
        output = output.terminal_node.state
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*The answer is .*?([ $.0-9,\-=]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer

def retrieve_answer_from_dataset(answer: Union[str, dict]) -> str:
    if isinstance(answer, dict):
        answer = answer['answer']
    return re.match(r'[\S\s]*#### (.*)$', answer)[1].replace(',', '').replace(' ', '')


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

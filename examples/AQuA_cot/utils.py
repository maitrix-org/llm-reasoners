import re
from typing import Optional, Union
from reasoners.base import AlgorithmOutput
from collections import Counter

def retrieve_answer(output: Union[list, str, AlgorithmOutput]) -> Optional[str]:
    '''
    output should be a world_model.MATHState if being a list
    '''
    print('output:', output)    
    answers = []  
    for output_i in output:
        match = re.match(r'.*[Tt]he answer is.*?([A-E]).*?$', output_i, re.DOTALL)
        
        if match is None:
            print('match:', match)
            continue
        answer = match[1].strip()
        answers.append(answer)
    counter = Counter(answers)
    if counter == {}:
        return None
    answer, _ = counter.most_common(1)[0]
    print(f'retrived answer: {answer}')
    print(f'answers: {answers}')
    return answer

def retrieve_answer_not_option(output: Union[list, str, AlgorithmOutput]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, AlgorithmOutput):
        if (result := getattr(output, 'aggregated_result', None)) is not None:
            return result
        output = output.terminal_state
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*[Tt]he answer is .*?([ $.0-9,\/\+\-A-Zx-z\u221a\u2013\u00d7\u2217\u2026\u2234\u2019\u2044\u21d2\u00f7\u00c3\u2014\u2018\u00b0]+).*\..*', output)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer

def retrieve_answer_from_dataset(answer: str) -> str:
    return answer.strip()

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

def rap_extractor(algo_output, aggregate=True):
    
    from reasoners.algorithm import MCTSAggregation
    if aggregate:
        aggregator = MCTSAggregation(retrieve_answer, weight_policy='edge_inverse_depth')
        output = aggregator(algo_output.tree_state)
    else:
        if algo_output.terminal_state is None:
            output = None
        else:
            output = retrieve_answer(algo_output.terminal_state)
    return output
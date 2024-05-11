import re
from typing import Optional, Union


def retrieve_answer(output):
    '''
    output should be a world_model.MATHState if being a list
    '''
    print('retrieve_answer:', output)
    if isinstance(output, list):
        output = output[-1].sub_answer
    match = re.match(r'.*[Tt]he answer is.*?([A-E]).*?$', output, re.DOTALL)
    
    if match is None:
        print('match:', match)
        return None
    print('match:', match[1].strip())
    answer = match[1].strip()
    
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
import re
from typing import Optional, Union
from collections import Counter

def retrieve_answer(output: Union[list, str]) -> Optional[str]:
    '''
    output should be a world_model.GSM8kState if being a list
    '''
    if isinstance(output, list):
        output = output[-1].sub_answer
    # print("output:", output)
    match = re.match(r'.*[Tt]he answer is .*?([ $.0-9,\-]+).*\..*', output, re.DOTALL)
    if match is None:
        return None
    answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
    # print("answer:", answer)
    if '=' in answer:
        answer = answer[answer.rindex('=') + 1:]
    return answer


def retrieve_answer_from_dataset(answer: str) -> str:
    return re.match(r'[\S\s]*#### (.*)$', answer)[1]


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

def cot_sc_extractor(algo_output, sc=True):
    # aggregate the results from multiple reasoning chains with majority vote
    answers = [retrieve_answer(x) for x in algo_output]
    answers = [x for x in answers if x is not None]
    counter = Counter(answers)
    if counter == {}:
        return None
    return counter.most_common(1)[0][0]




class Reward_CoT_Utils():

    def __init__(self, reward_model) -> None:
        self.reward_model = reward_model.model
        self.tokenizer = reward_model.tokenizer

    
    def cot_reward_sc_extractor(self, algo_and_question_output):

        algo_output, question = algo_and_question_output
        non_none_ans = [x for x in algo_output if retrieve_answer(x)!=None]
        
        ans_reward_pairs = []
        for ans in non_none_ans:

            reward_input = self.tokenizer(question + ans, return_tensors = "pt")
            reward_input = {k:reward_input[k].to("cuda:0") for k in reward_input.keys()}
            reward = self.reward_model(**reward_input).item()
            ans_reward_pairs.append((reward,ans))

        if (len(ans_reward_pairs)==0):
            return None
        
        ans_reward_pairs.sort(key = lambda x:x[0])
        ans_with_max_reward = ans_reward_pairs[-1][1]
        return retrieve_answer(ans_with_max_reward)

        



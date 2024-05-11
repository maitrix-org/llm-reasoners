import pickle
from typing import Optional
import glob
import os
import json

from tqdm import tqdm
from datasets import load_dataset

from reasoners.algorithm import MCTSAggregation, MCTSResult

import utils
from world_model import StrategyQAState
from dataset import get_examples, extract_golden_answer 


def retrieve_answer(state: StrategyQAState):
    return utils.extract_final_answer(state[-1].sub_answer)


def aggregate_rap_strategyqa(log_dir: str,
                        start: int = 0):
    aggregator = MCTSAggregation(retrieve_answer, weight_policy='edge_inverse_depth')
    files = glob.glob(f'{log_dir}/algo_output/*.pkl')
    indices = sorted(filter(lambda index: index >= start, (int(os.path.basename(f)[:-4]) for f in files)))
    dataset = get_examples(folder='examples/rap_strategyQA/data/', split='test')
    correct_count = 0
    answer_dict = {}
    for i, index in enumerate(tqdm(indices)):
        with open(f'{log_dir}/algo_output/{index}.pkl', 'rb') as f:
            result: MCTSResult = pickle.load(f)
        output = aggregator(result.tree_state)
        output = True if output == 'yes' else False
        # answer = extract_golden_answer(dataset[index - 1])
        # answer = dataset[index - 1]['answer']
        # correct = utils.judge_answer(output, answer)

        # correct_count += correct
        # accuracy = correct_count / (i + 1)
        # log_str = f'Case #{i + 1}({index}): {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{i+1})'
        # tqdm.write(log_str)
        answer_dict[dataset[index - 1]['qid']] = {"answer": output, "decomposition": [], "paragraphs": []}
    with open(os.path.join(log_dir, 'all_answers-agg.json'), 'w') as f:
        json.dump(answer_dict, f, indent=2)


if __name__ == '__main__':
    import fire
    fire.Fire(aggregate_rap_strategyqa)

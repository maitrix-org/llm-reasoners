import pickle
from typing import Optional
import glob
import os

from tqdm import tqdm
from datasets import load_dataset

from reasoners.algorithm import MCTSAggregation, MCTSResult

import utils


def aggregate_rap_gsm8k(log_dir: str,
                        start: int = 0):
    aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy='edge_inverse_depth')
    files = glob.glob(f'{log_dir}/algo_output/*.pkl')
    indices = sorted(filter(lambda index: index >= start, (int(os.path.basename(f)[:-4]) for f in files)))
    dataset = load_dataset("gsm8k", "main", split=f'test')
    correct_count = 0
    for i, index in enumerate(tqdm(indices)):
        with open(f'{log_dir}/algo_output/{index}.pkl', 'rb') as f:
            result: MCTSResult = pickle.load(f)
        output = aggregator(result.tree_state)
        answer = utils.retrieve_answer_from_dataset(dataset[index - 1]['answer'])
        correct = utils.judge_answer(output, answer)

        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{i + 1}({index}): {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{i+1})'
        tqdm.write(log_str)


if __name__ == '__main__':
    import fire
    fire.Fire(aggregate_rap_gsm8k)

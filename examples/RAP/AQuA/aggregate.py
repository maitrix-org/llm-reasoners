import pickle
from typing import Optional
import glob
import os

from tqdm import tqdm
from datasets import load_dataset, Dataset

from reasoners.algorithm import MCTSAggregation, MCTSResult

import utils
def data_reader(filename, split=None, sample_size=100):
    questions = []
    answers = []
    options = []
    lines = Dataset.from_json(filename)
    if split is not None:
        start, end = split
        lines = lines[start:end]
    for i in range(len(lines)):
        data = lines[i]
        if isinstance(data, dict):
            options_list = data['options']
            question_with_options = data['question'] + "\n" + "Options: " + ", ".join(data['options']) + "."
            questions.append(question_with_options)
            answers.append(data['correct'])
            options.append(options_list)
        else:
            raise ValueError("Unexpected data format")
    return Dataset.from_dict({"question": questions, "answer": answers, "options":options})

def aggregate_AQuA(log_dir: str,
                        start: int = 0):
    print(log_dir)
    aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy='edge_inverse_depth')
    files = glob.glob(f'{log_dir}/algo_output/*.pkl')
    indices = sorted(filter(lambda index: index >= start, (int(os.path.basename(f)[:-4]) for f in files)))
    dataset = data_reader("examples/AQuA_rap/dataset/AQuA/AQuA_clean.json")
    correct_count = 0
    for i, index in enumerate(tqdm(indices)):
        with open(f'{log_dir}/algo_output/{index}.pkl', 'rb') as f:
            result: MCTSResult = pickle.load(f)
        output = aggregator(result.tree_state)
        answer = utils.retrieve_answer_from_dataset(dataset[index - 1]['answer'])
        print(answer)
        correct = utils.judge_answer(output, answer)

        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{i + 1}({index}): {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{i+1})'
        tqdm.write(log_str)

#It is not used.
if __name__ == '__main__':
    import fire
    fire.Fire(aggregate_AQuA)

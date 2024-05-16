import pickle
# add path
import sys
sys.path.append('..')
import os
# print(os.cwd())
from reasoners.visualization import visualize
from reasoners.visualization.tree_snapshot import NodeData
from reasoners.algorithm.mcts import MCTSNode
from datasets import load_dataset
import pandas as pd
import fire
df = pd.DataFrame(columns=['question', 'cot'])
# data = load_dataset('gsm8k','main','test')
from datasets import Dataset
def data_reader(dataset, dataset_path, split=None, sample_size=None):
    questions = []
    answers = []
    options = []
    filename = os.path.join(dataset_path, f'{dataset}.json')
    lines = Dataset.from_json(filename)
    if split is not None:
        start, end = split
        lines = lines[start:end]
    for i in range(len(lines)):
        data = lines[i]
        if isinstance(data, dict):
            options_list = data['options']
            question_with_options = data['question'] + " Options: " + (" ".join(data['options'])).replace('A)','A) ').replace('B)','B) ').replace('C)','C) ').replace('D)','D) ').replace('E)','E) ') + "."
            questions.append(question_with_options)
            answers.append(data['correct'])
            options.append(options_list)
        else:
            raise ValueError("Unexpected data format")
    return Dataset.from_dict({"question": questions, "answer": answers, "options":options})


def get_trace_gsm8k():
    data = load_dataset('gsm8k','main','test')
    for i in range(1,len(data['test'])+1): 
        mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02292024-025642/algo_output/{i}.pkl', 'rb'))
        question = data['test'][i-1]['question']
        cot = mcts_result[0]
        cot = cot.split('Q:')[0]
        # cot = cot.split('\n')[0]#for weak model
        cot_steps = cot.split('. ')
        print(cot)
        cot_final = ""
        # cot_final = cot
        for j in range(len(cot_steps)):
            cot_final += f'Step {j+1}: ' + cot_steps[j] + ".\n"
        cot_final = cot_final.rstrip('\n')
        df.loc[i-1] = [question, cot_final]

    df.to_json('/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/02292024-025642/cot1.json')

def get_trace_sq():
    # data = data_reader('AQuA','/data/haotian/RAP_tune/llm-reasoners/dataset/AQuA')
    import json
    with open('/data/haotian/RAP_tune/llm-reasoners/examples/rap_strategyQA/data/strategyqa_test.json', 'r') as f:
        data = json.load(f)
    # data = load_dataset('gsm8k','main','test')
    for i in range(1,len(data)+1): 
        mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/strategyqa_cot/03062024-052230_anthropic/algo_output/{i}.pkl', 'rb'))
        question = data[i-1]['question']
        cot = mcts_result
        cot = cot.split('Q:')[0]
        cot = cot.split('\n')[0]
        cot_steps = cot.split('. ')
        
        print(cot)
        cot_final = ""
        # cot_final = cot
        for j in range(len(cot_steps)):
            cot_final += f'Step {j+1}: ' + cot_steps[j] + ".\n"
        cot_final = cot_final.rstrip('\n')
        df.loc[i-1] = [question, cot_final]

    df.to_json('/data/haotian/RAP_tune/llm-reasoners/logs/strategyqa_cot/03062024-052230_anthropic/cot.json')


def get_trace_aqua():
    data = data_reader('AQuA','/data/haotian/RAP_tune/llm-reasoners/dataset/AQuA')
    for i in range(1,len(data)+1): 
        mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/AQuAcot/03052024-051243_anthropic/algo_output/{i}.pkl', 'rb'))
        question = data[i-1]['question']
        cot = mcts_result[0]
        print('------------',cot)
        cot = cot.split('Q:')[0]
        # cot = cot.split('\n')[0]
        # cot_steps = cot.split('. ')
        print(cot)
        cot_final = cot
        # for j in range(len(cot_steps)):
        #     cot_final += f'Step {j+1}: ' + cot_steps[j] + ".\n"
        cot_final = cot_final.rstrip('\n')
        df.loc[i-1] = [question, cot_final]

    df.to_json('/data/haotian/RAP_tune/llm-reasoners/logs/AQuAcot/03052024-051243_anthropic/cot.json')


# fire.Fire(get_trace_sq)
fire.Fire(get_trace_gsm8k)
# fire.Fire(get_trace_aqua)
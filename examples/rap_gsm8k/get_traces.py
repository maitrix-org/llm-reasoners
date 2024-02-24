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
df = pd.DataFrame(columns=['question', 'cot'])
data = load_dataset('gsm8k','main','test')
for i in range(1,300): 
    mcts_result = pickle.load(open(f'/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/11012023-143556/algo_output/{i}.pkl', 'rb'))
    question = data['test'][i-1]['question']
    cot = mcts_result
    cot = cot.split('Q:')[0]
    cot_steps = cot.split('. ')
    print(cot)
    cot_final = ""
    for j in range(len(cot_steps)):
        cot_final += f'Step {j+1}: ' + cot_steps[j] + ".\n"
    df.loc[i-1] = [question, cot_final]

df.to_json('/data/haotian/RAP_tune/llm-reasoners/logs/gsm8k_unknown/11012023-143556/cot.json')

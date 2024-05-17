import pickle
from typing import Type, Callable, Optional

import numpy as np
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import DFS

from world_model import CrosswordsWorldModel
from search_config import CrosswordsConfig
from utils import MiniCrosswordsEnv



def tot_crosswords(base_model: LanguageModel,
              search_algo: Type[SearchAlgorithm] = DFS,
              resume: int = 0,
              n_eval: int = 8,
              depth: int = 10,
              batch_size: int = 2,
              max_per_state: int = 3,
              total_states: int = 100,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              temperature: float = 0.7 ,
              **search_algo_params):
    
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/crosswords_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume >= 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    env = MiniCrosswordsEnv()
    ## keep the best 5 candidates, need at most 4 steps to solve
    ## following ToT, eval step will consider number of times to prompt for state evaluation
    search_algo_params |= {'max_per_state': max_per_state, 'total_states': total_states, 'depth': depth}
    world_model = CrosswordsWorldModel(base_model=base_model)
    config = CrosswordsConfig(base_model=base_model,
                         depth=depth, n_eval=n_eval, temperature=temperature )
    search_algo = search_algo(**search_algo_params)
    agent = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    correct = 0
    correct_count = 0
    example_cnt = 0
    answer=''
    answer_list = []
    infoss = []
    
    for index, i in tqdm(enumerate(range(5, 60, 5))):
        example_cnt += 1
        #algo_output = agent(i, prior=True)
        algo_output = agent(i)
        best = 0
        output = ''
        ans = ''
        infos = []
        for output_i, state in enumerate(algo_output):
            env, actions, info = state
            if best < info['info']['r_word']:
                best = info['info']['r_word']
                output = env.ans
                answer = env.ans_gt
            print(f'{output_i}, {env.ans}, {output}')
            infos.append(info)
        answer_list.append((output, answer, best, search_algo.stat_cnt))
        infoss.append(infos)
        if best == 1.0:
            correct = 1
            correct_count += 1
        accuracy = correct_count / example_cnt
        log_str = f'Case #{resume + i}: {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{example_cnt})'
        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
            with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)
    with open(os.path.join(log_dir, 'infoss_dfs_tot.json'), 'w') as f:
        json.dump(infoss, f)


if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    from reasoners.lm import OpenAIModel, Llama2Model, Llama3Model
    import random
    import torch
    import torch.backends.cudnn

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True


    def main(
             model_dir= None, 
             llama_size=None,
             batch_size: int = 2,
             prompts: str = 'examples/ToT/crosswords/prompts/crosswords.json', # not used
             disable_log: bool = False,
             model: str = 'gpt-4',
             temperature: float = 0.7,
             **kwargs):
        
        if model == 'llama2':
            base_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
            raise SystemExit("Non't support yet")
        elif model == 'llama3':
            base_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
            raise SystemExit("Non't support yet")
        else:
            base_model = OpenAIModel(model=model, temperature=temperature, max_tokens=1000)
        #log_dir = 'logs/crosswords_dfs/test-gpt3.5'
        tot_crosswords(base_model=base_model,
                  batch_size=batch_size, # not used
                  disable_log=disable_log,
                  temperature=temperature,
                  **kwargs)


    fire.Fire(main)

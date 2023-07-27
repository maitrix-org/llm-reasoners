import pickle
from typing import Type, Callable, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import DFS

from world_model import crosswordsWorldModel
from search_config import crosswordsConfig
from utils import MiniCrosswordsEnv



def rap_crosswords(base_model: LanguageModel,
              search_algo: Type[SearchAlgorithm] = DFS,
              resume: int = 0,
              n_select_sample: int = 5,
              depth: int = 5,
              batch_size: int = 2,
              max_per_state: int = 3,
              total_states: int = 100,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
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
    world_model = crosswordsWorldModel(base_model=base_model, batch_size=batch_size)
    config = crosswordsConfig(base_model=base_model,
                         batch_size=batch_size,
                         depth=depth)
    search_algo = search_algo(**search_algo_params)
    agent = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    correct_count = 0
    for i in range(0, 100, 5):
        infos = []
        actions = []
        agent(i, best_state=True)
        algo_output = agent.search_config.terminals
        log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{i + 1})'
        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
            with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)


if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    from reasoners.lm import GPTCompletionModel
    import random
    import torch
    import torch.backends.cudnn

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    llama_ckpts = os.environ["LLAMA_CKPTS"]
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank != 0:
        sys.stdout = sys.stderr = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')


    def main(llama_ckpt: str = llama_ckpts,
             llama_size: str = '30B',
             batch_size: int = 2,
             prompts: str = 'examples/crosswords/prompts/crosswords.json',
             disable_log: bool = False,
             **kwargs):
        openai_model = GPTCompletionModel(model='gpt-3.5-turbo')
        rap_crosswords(base_model=openai_model,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  **kwargs)


    fire.Fire(main)

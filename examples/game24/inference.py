import pickle
from typing import Type, Callable, Optional

import numpy as np
# from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

from rap import LanguageModel, RAPAgent, SearchAlgorithm
from rap.algorithm import BeamSearch

from world_model import game24WorldModel
from search_config import game24Config
import utils


def rap_game24(base_model: LanguageModel,
              prompts: dict,
              search_algo: Type[SearchAlgorithm] = BeamSearch,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              n_select_sample: int = 5,
              n_eval_sample: int = 3,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
              reward_confidence_default: float = 0.8,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              **search_algo_params):
    
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/game24_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume >= 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    ## keep the best 5 candidates, need at most 4 steps to solve
    ## following ToT, eval step will consider number of times to prompt for state evaluation
    search_algo_params |= {'beam_size': n_select_sample, 'max_depth': depth_limit}
    world_model = game24WorldModel(base_model=base_model, prompt=prompts,
                                  n_confidence=n_confidence, batch_size=batch_size)
    config = game24Config(base_model=base_model, prompt=prompts,
                         n_actions=n_action, batch_size=batch_size,reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit, n_eval=n_eval_sample)
    search_algo = search_algo(**search_algo_params)
    agent = RAPAgent(world_model=world_model, search_config=config, search_algo=search_algo)

    ## test from 900-999
    dataset = utils.read_data(file='./examples/game24/data/24.csv')[900:906]
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume, desc='game24')):
        print(f'\n======== example: {example} ========')
        agent.world_model = game24WorldModel(base_model=base_model, prompt=prompts,
                                  n_confidence=n_confidence, batch_size=batch_size)
        config.value_cache = {}
        algo_output = agent(example)
        answer = 24
        correct = 0
        output = ''
        # print(f'agent output: {algo_output}')
        ## eval each trace, consider correct if one trace can correctly reach 24
        for end_state in algo_output:
            output = end_state[0][-1][-1]
            print(output.sub_answer.replace('\n', '->'))
            output_check = utils.test_output(output.sub_question, output.sub_answer)
            if output_check['r']:
                correct = output_check['r']
                break

        correct_count += correct
        accuracy = correct_count / (i + 1)
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
    from rap.lm import LLaMAModel, GPTModel
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
             llama_size: str = '13B',
             batch_size: int = 2,
             prompts: str = 'examples/game24/prompts/game24.json',
             disable_log: bool = False,
             **kwargs):
        with open(prompts) as f:
            prompts = json.load(f)
        # llama_model = LLaMAModel(llama_ckpt, llama_size, max_batch_size=batch_size)
        ## try GPT
        gpt_model = GPTModel(model='gpt-3.5-turbo')
        rap_game24(base_model=gpt_model,
                  prompts=prompts,
                  batch_size=batch_size,
                  n_select_sample=5,
                  disable_log=disable_log or local_rank != 0,
                  **kwargs)


    fire.Fire(main)

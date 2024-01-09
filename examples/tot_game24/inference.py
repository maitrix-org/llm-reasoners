import pickle
from typing import Type, Callable, Optional

import numpy as np
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch

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
              depth_limit: int = 4,
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
    search_algo_params |= {'beam_size': n_select_sample, 'max_depth': depth_limit, 'reject_sample': False, 
                           'action_dedup': True, 'return_beam': True, 'early_terminate': False, 'reward_aggregator': 'last'}
    world_model = game24WorldModel(base_model=base_model, prompt=prompts,
                                  n_confidence=n_confidence, batch_size=batch_size)
    config = game24Config(base_model=base_model, prompt=prompts,
                         n_actions=n_action, batch_size=batch_size,reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit, n_eval=n_eval_sample)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    ## test from 900-999
    dataset = utils.read_data(file='./examples/game24/data/24.csv')[900:1000]
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=len(dataset), initial=0, desc='game24')):
        print(f'\n======== example {i}: {example} ========')
        base_model = GPTCompletionModel(model='gpt-3.5-turbo')
        reasoner.world_model = game24WorldModel(base_model=base_model, prompt=prompts,
                                  n_confidence=n_confidence, batch_size=batch_size)
        # reasoner.search_config.value_cache = {}
        algo_output = reasoner(example)
        # print(f'search cache size: {len(reasoner.search_config.value_cache)}')
        answer = 24
        correct = 0
        output = ''
        # print(f'reasoner output: {algo_output}')
        ## eval each trace, consider correct if one trace can correctly reach 24
        for subresult in algo_output:
            output = subresult.terminal_node.state
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
    from reasoners.lm import GPTCompletionModel
    import random
    import torch
    import torch.backends.cudnn

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True


    def main(batch_size: int = 2,
             prompts: str = 'examples/tot_game24/prompts/game24.json',
             disable_log: bool = False,
             model: str = 'gpt-3.5-turbo',
             temperature: float = 0.7,
             **kwargs):
        with open(prompts) as f:
            prompts = json.load(f)
        # llama_model = LLaMAModel(llama_ckpt, llama_size, max_batch_size=batch_size)
        ## try GPT
        openai_model = GPTCompletionModel(model=model, temperature=temperature)
        rap_game24(base_model=openai_model,
                  prompts=prompts,
                  batch_size=batch_size,
                  n_select_sample=5,
                  disable_log=disable_log,
                  **kwargs)


    fire.Fire(main)

import pickle
from typing import Type, Callable, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

from rap import LanguageModel, RAPAgent, SearchAlgorithm
from rap.algorithm import MCTS

from world_model import BlocksWorldModel
from search_config import BWConfig
import utils


def rap_bw(base_model: LanguageModel,
              prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              data_path: str = 'data',
              resume: int = 0,
              depth_limit: int = 5,
              reward_alpha: float = 0.5,
              batch_size = 1,
              goal_reached_reward = 100,
              goal_reward_default = 0.,
              cum_reward: Callable[[list[float]], float] = sum,
              calc_q: Callable[[list[float]], float] = np.mean,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              **search_algo_params):
    
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/bw_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q}
    world_model = BlocksWorldModel(base_model=base_model, prompt=prompt, batch_size=batch_size, depth_limit=depth_limit)

    config = BWConfig(base_model=base_model, prompt=prompt, batch_size=batch_size, reward_alpha=reward_alpha, depth_limit=depth_limit, goal_reached_reward=goal_reached_reward, goal_reward_default=goal_reward_default)
    
    search_algo = search_algo(**search_algo_params)
    agent = RAPAgent(world_model=world_model, search_config=config, search_algo=search_algo)

    dataset = utils.load(dataset_path=data_path)  # [{"goal": str, "init": str}]
    
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume, desc='GSM8k')):
        algo_output = agent(example["question"])
        correct = utils.judge_answer(output, answer)

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
    import random
    import torch
    import torch.backends.cudnn
    from rap.lm import LLaMAModel

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    llama_ckpts = os.environ["LLAMA_CKPTS"]

    def main(llama_ckpt: str = llama_ckpts,
             llama_size: str = '13B',
             batch_size: int = 2,
             prompt_path: str = 'examples/gsm8k/prompts/interactive_examples.json',
             data_path: str = 'examples/gsm8k/prompts/useful_examples.json',
             disable_log: bool = False,
             **kwargs):
        with open(interactive_prompt) as f:
            interactive_prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        llama_model = LLaMAModel(llama_ckpt, llama_size, max_batch_size=batch_size)
        rap_gsm8k(base_model=llama_model,
                  interactive_prompt=interactive_prompt,
                  useful_prompt=useful_prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  **kwargs)


    fire.Fire(main)

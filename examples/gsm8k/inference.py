from typing import Type

import numpy as np
from datasets import load_dataset

from rap import LanguageModel, RAPAgent, SearchAlgorithm
from rap.algorithm import MCTS

from .world_model import GSM8kWorldModel
from .search_config import GSM8kConfig
from . import utils


def rap_gsm8k(base_model: LanguageModel,
              interactive_prompt: dict,
              useful_prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              search_algo_params: dict = None,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 0.8):
    if search_algo_params is None:
        search_algo_params = {
            'cum_reward': np.mean,
            'calc_q': max,
        }
    world_model = GSM8kWorldModel(base_model=base_model, prompt=interactive_prompt,
                                  n_confidence=n_confidence, batch_size=batch_size,
                                  early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = GSM8kConfig(base_model=base_model, prompt=interactive_prompt, useful_prompt=useful_prompt,
                         n_actions=n_action, batch_size=batch_size,
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    agent = RAPAgent(world_model=world_model, search_config=config, search_algo=search_algo)

    dataset = load_dataset("gsm8k", "main")["test"]
    for example in dataset:
        output = agent(example["question"])
        output = utils.retrieve_answer(output.terminal_state[-1].sub_answer)
        answer = utils.retrieve_answer_from_dataset(example["answer"])
        correct = utils.judge_answer(output, answer)
        print(correct, output, answer)


if __name__ == '__main__':
    import os
    import json
    import fire
    from rap.lm import LLaMAModel

    llama_ckpts = os.environ["LLAMA_CKPTS"]


    def main(llama_ckpt: str = llama_ckpts,
             llama_size: str = '13B',
             batch_size: int = 2,
             interactive_prompt: str = 'examples/gsm8k/prompts/interactive_examples.json',
             useful_prompt: str = 'examples/gsm8k/prompts/useful_examples.json',
             **kwargs):
        with open(interactive_prompt) as f:
            interactive_prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        llama_model = LLaMAModel(llama_ckpt, llama_size, max_batch_size=batch_size)
        rap_gsm8k(base_model=llama_model,
                  interactive_prompt=interactive_prompt,
                  useful_prompt=useful_prompt,
                  batch_size=batch_size, **kwargs)


    fire.Fire(main)

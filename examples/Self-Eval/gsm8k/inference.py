import pickle
import os
import sys
from typing import Type, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch
from reasoners.lm import OpenAIModel , Llama2Model, Llama3Model


from world_model import GSM8kWorldModel
from search_config import GSM8kConfig

from utils import construct_full_solution, retrieve_answer_from_dataset, judge_answer, majority_voting

def guided_decoding_gsm8k(base_model: LanguageModel,
                search_algo: Type[SearchAlgorithm] = BeamSearch,
                resume: int = 0,
                n_actions: int = 16,
                depth_limit: int = 16,
                force_terminating_on_depth_limit: bool = True,
                temperature: float = 1,
                reward_alpha: float = 0.5,
                beam_size: int = 5,
                max_depth: int = 16,
                sampling_strategy: str = 'stochastic',
                replace: bool = False,
                beam_search_temperature: float = 0.5,
                beam_search_temperature_decay: float = 1,
                reject_sample: bool = True, 
                reject_min_reward: float = 0.6,
                unbiased: bool = True,
                majority_voting_n: int = 1,
                log_dir: Optional[str] = None,
                disable_log: bool = False,
                disable_tqdm: bool = False,
                **search_algo_params):
    
    # majority voting can not be larger than beam size
    assert majority_voting_n <= beam_size, f'majority_voting_n={majority_voting_n} should be smaller than beam_size={beam_size}'

    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/guided_decoding_gsm8k_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
    
    # set parameters for beam search
    search_algo_params |= {
            'beam_size': beam_size, 
            'max_depth': max_depth,
            'sampling_strategy': sampling_strategy,
            'replace': replace,
            'temperature': beam_search_temperature,
            'temperature_decay': beam_search_temperature_decay,
            'reject_sample': reject_sample,
            'reject_min_reward': reject_min_reward,
            'unbiased': unbiased,
            # just use the last reward
            'reward_aggregator': lambda x: x[-1],
            'return_beam': True,
            }
    
    world_model = GSM8kWorldModel(base_model=base_model)
    config = GSM8kConfig(base_model=base_model,
                            n_actions=n_actions,
                            temperature=temperature,
                            reward_alpha=reward_alpha,
                            force_terminating_on_depth_limit=force_terminating_on_depth_limit,
                            depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    dataset = load_dataset("gsm8k", "main", split=f'test[{resume}:]')
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
                                     desc='GSM8k', disable=disable_tqdm)):
        algo_output = reasoner(example["question"])

        outputs = []
        # certain trace may be rejected, if we set reject_sample=True
        majority_voting_n = min(majority_voting_n, len(algo_output))
        for j in range(majority_voting_n):
            # get the last state
            state = algo_output[j].terminal_node.state
            # use the state to form the full output
            output = construct_full_solution(state, execute=True)
            outputs.append(output)
        
        # get the most common output, if there is a tie, always choose the first one
        output = majority_voting(outputs)

        # get the answer from the dataset
        answer = retrieve_answer_from_dataset(example["answer"])
        # judge the answer
        correct = judge_answer(output, answer)
        # if correct, add 1 to correct_count
        if correct:
            correct_count += 1

        accuracy = correct_count / (i + 1)
        log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{i + 1})'
        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
            with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)

def main(base_lm: str = 'codex',
        llama_path: str = None,
        llama_size: str = None,
        batch_size: int = 1,
        resume: int = 0,
        n_actions: int = 16,
        depth_limit: int = 16,
        force_terminating_on_depth_limit: bool = True,
        temperature: float = 1,
        reward_alpha: float = 0.5,
        beam_size: int = 5,
        max_depth: int = 16,
        sampling_strategy: str = 'stochastic',
        replace: bool = False,
        beam_search_temperature: float = 0.5,
        beam_search_temperature_decay: float = 1,
        reject_sample: bool = True,
        reject_min_reward: float = 0.6,
        unbiased: bool = True,
        majority_voting_n: int = 1,
        log_dir: Optional[str] = None,       
        disable_log: bool = False,
        disable_tqdm: bool = False,
        **kwargs):
    
    np.random.seed(42)

    if base_lm == 'codex':
        base_model = OpenAIModel('code-davinci-002')
    elif base_lm == 'llama2':
        base_model = Llama2Model(llama_path, llama_size, max_batch_size=batch_size)
        raise SystemExit("Non't support yet")
    elif base_lm == 'llama3':
        base_model = Llama3Model(llama_path, llama_size, max_batch_size=batch_size)
        raise SystemExit("Non't support yet")
    else:
        raise NotImplementedError(f'base_lm={base_lm} is not implemented')
    
    guided_decoding_gsm8k(base_model=base_model,
                        resume=resume,
                        n_actions=n_actions,
                        depth_limit=depth_limit,
                        force_terminating_on_depth_limit=force_terminating_on_depth_limit,
                        temperature=temperature,
                        reward_alpha=reward_alpha,
                        beam_size=beam_size,
                        max_depth=max_depth,
                        sampling_strategy=sampling_strategy,
                        replace=replace,
                        beam_search_temperature=beam_search_temperature,
                        beam_search_temperature_decay=beam_search_temperature_decay,
                        reject_sample=reject_sample,
                        reject_min_reward=reject_min_reward,
                        unbiased=unbiased,
                        majority_voting_n=majority_voting_n,
                        log_dir=log_dir,
                        disable_log=disable_log,
                        disable_tqdm=disable_tqdm,
                        **kwargs)
    
if __name__ == '__main__':
    import fire
    fire.Fire(main)
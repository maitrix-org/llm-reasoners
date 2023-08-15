import torch
import random
import warnings
import pickle
import os
import sys
import json
from typing import Type, Optional

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch
from reasoners.lm import LLaMAModel, LlamaCppModel, LlamaModel

from world_model import StrategyQAWorldModel
from search_config import StrategyQAConfig

from utils import judge_answer, majority_voting, parse_answer

def least_to_most_strategyqa(base_model: LanguageModel,
                search_algo: Type[SearchAlgorithm] = BeamSearch,
                resume: int = 0,
                depth_limit: int = 16,
                temperature: float = 0.7,
                beam_size: int = 1,
                self_consistency_n: int = 1,
                max_depth: int = 16,
                data_path: str = "examples/least_to_most_strategyQA/strategyqa_test.json",
                log_dir: Optional[str] = None,
                disable_log: bool = False,
                disable_tqdm: bool = False,
                **search_algo_params):
    
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/least_to_most_strategyqa_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
    
    # set parameters for beam search
    search_algo_params |= {
            'beam_size': beam_size, 
            'max_depth': max_depth,
            }
    
    world_model = StrategyQAWorldModel(base_model=base_model)
    config = StrategyQAConfig(base_model=base_model,
                            temperature=temperature,
                            depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    with open(data_path, 'r') as f:
        dataset = json.load(f)

    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
                                     desc='Strategy QA', disable=disable_tqdm)):
        outputs = []

        for _ in range(self_consistency_n):
            # run the reasoner
            algo_output = reasoner(example["question"])
            # get the last state
            state = algo_output.terminal_node.state[-1]
            # answer
            output = state.sub_answer
            # parse the answer
            output = parse_answer(output)
            # add to outputs
            outputs.append(output)
        
        # get the most common output, if there is a tie, always choose the first one
        output = majority_voting(outputs)

        # get the answer from the dataset
        answer = example["answer"]
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

def main(base_lm: str = "llama",
        llama_size: str = "30B",
        resume: int = 0,
        depth_limit: int = 16,
        temperature: float = 1,
        beam_size: int = 1,
        self_consistency_n: int = 1,
        max_depth: int = 16,
        data_path: str = "examples/least_to_most_strategyQA/strategyqa_test.json",
        log_dir: Optional[str] = None,       
        disable_log: bool = False,
        disable_tqdm: bool = False,
        **kwargs):
    
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    if base_lm == 'llama':
        base_model = LLaMAModel(llama_ckpts, llama_size)
    elif base_lm == 'llama2':
        base_model = LlamaModel(llama_2_ckpts, llama_size)
    else:
        assert False, f'cannot resolve {base_lm=}'
    
    least_to_most_strategyqa(base_model=base_model,
                            resume=resume,
                            depth_limit=depth_limit,
                            temperature=temperature,
                            beam_size=beam_size,
                            self_consistency_n=self_consistency_n,
                            max_depth=max_depth,
                            data_path=data_path,
                            log_dir=log_dir,
                            disable_log=disable_log,
                            disable_tqdm=disable_tqdm,
                            **kwargs)
    
if __name__ == '__main__':
    import fire
    fire.Fire(main)
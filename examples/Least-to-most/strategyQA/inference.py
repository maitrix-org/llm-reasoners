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
from reasoners.lm import  LlamaModel, ExLlamaModel,  Llama2Model, Llama3Model
from collections import Counter

from world_model import StrategyQAWorldModel
from search_config import StrategyQAConfig

from utils import judge_answer, majority_voting, parse_answer

def least_to_most_strategyqa(base_model: LanguageModel,
                search_algo: Type[SearchAlgorithm] = BeamSearch,
                resume: int = 0,
                depth_limit: int = 16,
                temperature: float = 0.8,
                beam_size: int = 1,
                self_consistency_n: int = 10,
                max_depth: int = 16,
                data_path: str = "examples/CoT/strategyQA/data/strategyqa_test.json",
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
    
    world_model = StrategyQAWorldModel(base_model=base_model, temperature=temperature,log_dir=log_dir)
    config = StrategyQAConfig(base_model=base_model,
                            temperature=temperature,
                            self_consistency_n=self_consistency_n,
                            depth_limit=depth_limit,
                            log_dir=log_dir)
    search_algo = search_algo(**search_algo_params)

    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    with open(data_path, 'r') as f:
        dataset = json.load(f)
    
    # the first 500 examples are used
    dataset = dataset[:500]

    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
                                     desc='Strategy QA', disable=disable_tqdm)):
        outputs = []

        seed_id = 0
        np.random.seed(seed_id)
        random.seed(seed_id)
        torch.manual_seed(seed_id)
        torch.cuda.manual_seed(seed_id)
        torch.backends.cudnn.deterministic = True

        for _ in range(self_consistency_n):
            # run the reasoner
            algo_output = reasoner(example["question"])
            # get the last state
            #state = algo_output.terminal_state.state[-1]
            if algo_output.terminal_state is None:
                output = None
            else:
                state = algo_output.terminal_state[-1]  
                # answer
                output = state.sub_answer
                # parse the answer
                output = parse_answer(output)
            # add to outputs
            outputs.append(output)

            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            if local_rank == 0:
                with open(log_dir+"/log.txt", "a") as f:
                    print(f"FINAL ANSWER: {output}", file=f)
        
        # get the most common output, if there is a tie, always choose the first one
        output = majority_voting(outputs)
        if local_rank == 0:
            with open(log_dir+"/log.txt", "a") as f:
                # print the distribution of outputs, counter in one line
                counter = Counter(outputs)
                print(f"\nSC OUTPUTS: {counter}\n", file=f)

                print(f"\nSC ANSWER: {output}\n", file=f)


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
        llama_ckpts: str = None,
        batch_size: int = 5,
        resume: int = 0,
        depth_limit: int = 16,
        temperature: float = 0.8,
        beam_size: int = 1,
        self_consistency_n: int = 10,
        max_depth: int = 16,
        data_path: str = "examples/CoT/strategyQA/data/strategyqa_test.json",
        log_dir: Optional[str] = None,       
        disable_log: bool = False,
        disable_tqdm: bool = False,
        **kwargs):

   
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    if base_lm == 'llama':
        base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
    elif base_lm == 'llama2':
        base_model = Llama2Model(llama_ckpts, llama_size,max_batch_size=batch_size, max_seq_len=4096) 
    elif base_lm == 'llama3':
        base_model = Llama3Model(llama_ckpts, llama_size,max_batch_size=batch_size, max_seq_len=4096) 
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
    
def main_exllama(
        model_dir = '/data/tianyang/llm-reasoners/ckpts/Llama-2-70B-GPTQ',
        lora_dir = None,
        batch_size = 1,
        mem_map = [16,22],
        resume: int = 0,
        depth_limit: int = 16,
        temperature: float = 0.8,
        beam_size: int = 1,
        self_consistency_n: int = 10,
        max_depth: int = 16,
        data_path: str = "examples/CoT/strategyQA/data/strategyqa_test.json",
        log_dir: Optional[str] = None,       
        disable_log: bool = False,
        disable_tqdm: bool = False,
        **kwargs):
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    device = torch.device("cuda:0")
    base_model = ExLlamaModel(model_dir,
                                lora_dir,
                                device,
                                max_batch_size=batch_size,
                                max_new_tokens=256,
                                max_seq_length=2048,
                                mem_map=mem_map)

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
    #fire.Fire(main_exllama)
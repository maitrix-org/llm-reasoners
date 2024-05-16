import pickle
from typing import Type, Optional, Literal

import numpy as np
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch, MCTS, MCTSNode
from reasoners.visualization import TreeLog

from world_model import Game24WorldModel, Game24State, Game24Action
from search_config import Game24Config
import utils


def node_visualizer(x: MCTSNode):
    ret = {}
    if x.action is not None:
        ret['last_step'] = x.action
    if x.state is not None:
        ret = {'current': x.state.current}
        if x.state.output is not None:
            ret['output'] = x.state.output
    return ret


def rap_game24(base_model: LanguageModel,
               prompts: dict,
               search_algo: Type[SearchAlgorithm] = BeamSearch,
               resume: int = 0,
               n_action: int = 4,
               n_beam: int = 5,
               n_eval: int = 3,
               depth_limit: int = 4,
               batch_size: int = 3,
               log_dir: Optional[str] = None,
               disable_log: bool = False,
               calc_reward: Literal['sampling', 'logits'] = 'sampling',
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
    # search_algo_params |= {'beam_size': n_beam, 'max_depth': depth_limit}
    search_algo_params |= {'output_trace_in_each_iter': True, 'depth_limit': depth_limit, 'disable_tqdm': False}
    world_model = Game24WorldModel(base_model=base_model, prompt=prompts, batch_size=batch_size)
    config = Game24Config(base_model=base_model, prompt=prompts, calc_reward=calc_reward,
                          n_actions=n_action, n_eval=n_eval, batch_size=batch_size, depth_limit=depth_limit,)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    # test from 900-999
    dataset = utils.read_data(file='./examples/ToT/game24/data/24.csv')[900:1000]
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=len(dataset), initial=0, desc='game24')):
        # print(f'\n======== example {i}: {example} ========')
        reasoner.world_model = Game24WorldModel(base_model=base_model, prompt=prompts, batch_size=batch_size)
        # algo_output = reasoner(example, action_dedup=True, return_beam=True, early_terminate=False,
        #                        reward_strategy='last_iter')
        algo_output = reasoner(example)
        output = algo_output.terminal_state.output if algo_output.terminal_state is not None else None
        # print(output)
        correct = utils.test_output(example, output)

        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{resume + i + 1}: {correct=}, {output=} ; {accuracy=:.3f} ({correct_count}/{i + 1})'
        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
            with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)
            if isinstance(search_algo, MCTS):
                with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.json'), 'w') as f:
                    # noinspection PyTypeChecker
                    print(TreeLog.from_mcts_results(algo_output, node_data_factory=node_visualizer), file=f)


if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    llama_3_ckpts = os.environ.get("LLAMA3_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama','llama-3'] = 'llama-2',
             llama_ckpts: str = llama_ckpts,
             llama_2_ckpts: str = llama_2_ckpts,
             llama_3_ckpts: str = llama_3_ckpts,
             llama_size: str = '13B',
             llama_cpp_path: str = None,
             llama_cpp_n_batch: int = 512,
             hf_path: str = 'meta-llama/Llama-2-13b-hf',
             hf_peft_path: Optional[str] = None,
             hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             hf_load_awq_path: Optional[str] = None,
             exllama_model_dir: str = 'WizardMath-13B-V1.0-GPTQ',
             exllama_lora_dir: Optional[str] = None,
             exllama_mem_map: Optional[str] = None,
             batch_size: int = 1,
             prompts: str = 'examples/ToT/game24/prompts/game24.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        with open(prompts) as f:
            prompts = json.load(f)
        if base_lm in ['llama', 'llama-2', 'llama-3']:
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        if base_lm == 'llama':
            from reasoners.lm import LlamaModel
            base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            from reasoners.lm import LlamaCppModel
            base_model = LlamaCppModel(llama_cpp_path, n_batch=llama_cpp_n_batch)
        elif base_lm == 'llama-2':
            from reasoners.lm import Llama2Model
            base_model = Llama2Model(llama_2_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama-3':
            from reasoners.lm import Llama3Model
            base_model = Llama3Model(llama_3_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                      max_batch_size=batch_size, max_new_tokens=512, max_seq_length=2048)
        else:
            assert False, f'cannot resolve {base_lm=}'
        rap_game24(base_model=base_model,
                   prompts=prompts,
                   batch_size=batch_size,
                   n_beam=5,
                   disable_log=disable_log or local_rank != 0,
                   search_algo=MCTS,
                   **kwargs)


    fire.Fire(main)

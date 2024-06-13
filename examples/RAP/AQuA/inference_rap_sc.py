import pickle
from typing import Type, Callable, Optional
import fire
import os
import numpy as np
from reasoners.algorithm.mcts import MCTSResult
from regex import F
#from sklearn import base
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode, MCTSAggregation
from reasoners.visualization import TreeLog
from reasoners.benchmark import AQuAEvaluator

from world_model import MATHWorldModel, MATHState, MATHAction
from search_config import MATHConfig
import utils

def node_visualizer(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}

def rap_AQuA_sc(base_model: LanguageModel,
              prompt: dict,
              useful_prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              resume: int = 0,
              n_action: int = 1,
              n_confidence: int = 8,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 1,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 0.8,
              cum_reward: Callable[[list[float]], float] = np.mean,
              calc_q: Callable[[list[float]], float] = max,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = True,
              aggregate: bool = False,
              **search_algo_params):
    
    print(f'aggregate: {aggregate}')
    if aggregate:
        aggregator = MCTSAggregation(utils.retrieve_answer, weight_policy='edge')
    else:
        aggregator = None
    
    search_algo_params |= {'cum_reward': cum_reward, 
                           'calc_q': calc_q, 
                           'disable_tqdm': disable_tqdm, 
                           'output_trace_in_each_iter': output_trace_in_each_iter,
                           'node_visualizer': node_visualizer, 
                           'aggregator': aggregator,
                           'w_exp': 1.0,
                           'n_iters': 1,
                           }
    
    world_model = MATHWorldModel(
        base_model=base_model,
        n_confidence=n_confidence, 
        batch_size=batch_size, 
        temperature=temperature,
        early_stop_base=early_stop_base, 
        early_stop_threshold=early_stop_threshold)
    
    config = MATHConfig(
        base_model=base_model, 
        useful_prompt=useful_prompt,
        n_actions=n_action, 
        batch_size=batch_size, 
        temperature=temperature,
        reward_alpha=reward_alpha, 
        reward_confidence_default=reward_confidence_default,
        force_terminating_on_depth_limit=force_terminating_on_depth_limit, 
        depth_limit=depth_limit)
    
    search_algo = search_algo(**search_algo_params)
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = AQuAEvaluator(output_extractor=utils.retrieve_answer,
                               answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                               init_prompt=prompt,
                               sample_prompt_type="rap",
                               disable_log=disable_log,
                               disable_tqdm=disable_tqdm)
    accuracy = evaluator.evaluate_sc(reasoner, num_shot=4, resume=resume, log_dir=log_dir)
    print(f'accuracy: {accuracy:.4f}')
    return 0

if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    from reasoners.lm import Llama2Model, LlamaCppModel, LlamaModel
    import random
    import torch
    import torch.backends.cudnn

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

    def main_exllama(
        base_lm = 'exllama',
        model_dir = '/path/to/model',
        lora_dir = None,
        batch_size = 1,
        mem_map = [16,22],
        prompt = "examples/AQuA_rap/prompts/prompt_pool.json",
        useful_prompt: str = 'examples/AQuA_rap/prompts/useful_examples.json',
        disable_log = False,
        disable_tqdm = False,
        reward_alpha = 0.5,
        **kwargs):
        
        from reasoners.lm import ExLlamaModel
        device = torch.device("cuda:0")
        base_model = ExLlamaModel(model_dir, lora_dir, device=device, max_batch_size=batch_size, max_new_tokens=512, max_seq_length=4096, mem_map=mem_map)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs
        
        with open(prompt) as f:
            prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)

        rap_AQuA_sc(
            base_model=base_model,
            prompt=prompt,
            useful_prompt=useful_prompt,
            batch_size=batch_size,
            disable_log=disable_log or local_rank != 0,
            disable_tqdm=disable_tqdm or local_rank != 0,
            reward_alpha = reward_alpha,
            **kwargs)
        
    fire.Fire(main_exllama)


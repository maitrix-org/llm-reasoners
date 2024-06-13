import pickle
from typing import Type, Callable, Optional, Literal

import os
import numpy as np
from datasets import load_dataset
from reasoners.algorithm.mcts import MCTSResult
from regex import F
#from sklearn import base
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode
from reasoners.visualization import TreeLog
from reasoners.benchmark import AQuAEvaluator

from world_model import MATHWorldModel, MATHState, MATHAction
from search_config import MATHConfig
import utils
import re

def node_visualizer(x: MCTSNode):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


def rap_AQuA(base_model: LanguageModel,
              prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 5,
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
              aggregate: bool = True,
              data_path: str = None,
              datasetname: str = None,
              **search_algo_params):
    
    world_model = MATHWorldModel(
        base_model=base_model, prompt={},
        n_confidence=n_confidence, 
        batch_size=batch_size, 
        temperature=temperature,
        early_stop_base=early_stop_base, 
        early_stop_threshold=early_stop_threshold)
    
    config = MATHConfig(
        base_model=base_model, 
        prompt={},
        n_actions=n_action, 
        batch_size=batch_size, 
        temperature=temperature,
        reward_alpha=reward_alpha, 
        reward_confidence_default=reward_confidence_default,
        depth_limit=depth_limit)
    
    search_algo_params |= {'cum_reward': cum_reward, 
                           'calc_q': calc_q, 
                           'disable_tqdm': disable_tqdm, 
                           'output_trace_in_each_iter': output_trace_in_each_iter}
    search_algo = search_algo(**search_algo_params)
    
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    evaluator = AQuAEvaluator(output_extractor=utils.rap_extractor,
                               answer_extractor=lambda x: utils.retrieve_answer_from_dataset(x["answer"]),
                               init_prompt=prompt,
                               disable_log=disable_log,
                               disable_tqdm=disable_tqdm,
                               dataset_path=data_path,
                               datasetname=datasetname)
    accuracy = evaluator.evaluate(reasoner, shuffle_prompt=True, num_shot=4, resume=resume, log_dir=log_dir)
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

    def main(
        base_lm: Literal['llama2',' exllama', 'llama3']  = 'exllama',
        model_dir = '/path/to/model',
        llama_size = "7B",
        lora_dir = None,
        batch_size = 1,
        mem_map = [16,22],
        prompt = "examples/Least-to-most/AQuA/prompts/AQuA_example_pool.json",
        disable_log = False,
        disable_tqdm = False,
        data_path="examples/CoT/AQuA/data/", 
        datasetname="test",
        **kwargs):

        if base_lm in ['llama2', 'llama3']:    
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True
        
        if base_lm == 'llama2':
            from reasoners.lm import Llama2Model
            base_model = Llama2Model(model_dir, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama3':
            from reasoners.lm import Llama3Model
            base_model = Llama3Model(model_dir, llama_size, max_batch_size=batch_size)
        else:
            from reasoners.lm import ExLlamaModel
            device = torch.device("cuda:0")
            base_model = ExLlamaModel(model_dir, lora_dir, device=device, max_batch_size=batch_size, max_new_tokens=512, max_seq_length=4096, mem_map=mem_map)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs
       
        print(os.getcwd())
        with open(prompt) as f:
            prompt = json.load(f)

        rap_AQuA(base_model=base_model,
                  prompt=prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  data_path=data_path, 
                  datasetname=datasetname,
                  **kwargs)

    fire.Fire(main)
    
    
    '''
    def main_hf(hf_path = "/data/haotian/RAP_tune/Llama-2-13b-hf",
            batch_size = 1,
            peft_path = None,
            interactive_prompt = "/data/haotian/RAP_tune/llm-reasoners/examples/AQuA/prompts/interactive_examples.json", 
            useful_prompt = "/data/haotian/RAP_tune/llm-reasoners/examples/AQuA/prompts/useful_examples.json",
            disable_log = False,
            disable_tqdm = False,
            quantized = "nf4", # awq, int8, fp4, nf4, None
            load_awq_pth = None,
            **kwargs):
        from reasoners.lm import HFModel
        device = torch.device("cuda:0")
        base_model = HFModel(hf_path, hf_path, device, max_batch_size=batch_size, max_new_tokens=512, peft_pth=peft_path, quantized=quantized, load_awq_pth=load_awq_pth)
        with open(interactive_prompt) as f:
            interactive_prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        rap_AQuA(base_model=base_model,
                    interactive_prompt=interactive_prompt,
                    useful_prompt=useful_prompt,
                    batch_size=batch_size,
                    disable_log=disable_log or local_rank != 0,
                    disable_tqdm=disable_tqdm or local_rank != 0,
                    **kwargs)
    '''
    
    """
    def main(base_lm: str = 'llama2', #llama means llama_v1 and llama2 means llama_v2
             llama_ckpt: str = llama_ckpts,
             llama_2_ckpt: str = llama_2_ckpts,
             llama_size: str = '7B',
             llama_cpp_path: str = None,
             batch_size: int = 1,
             interactive_prompt: str = '/data/haotian/RAP_tune/llm-reasoners/examples/AQuA/prompts/interactive_examples.json',
             useful_prompt: str = '/data/haotian/RAP_tune/llm-reasoners/examples/AQuA/prompts/useful_examples.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        # set base_lm = 'llama' and llama_ckpt = '13B/30B/65B' to use llama with torchscale
        # else set base_lm = 'llama.cpp' and llama_cpp_path = the checkpoint to use llama.cpp

        with open(interactive_prompt) as f:
            interactive_prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        if base_lm == 'llama':
            base_model = LlamaModel(llama_ckpt, llama_size, max_batch_size=batch_size) #LlamaModel
        elif base_lm == 'llama.cpp':
            base_model = LlamaCppModel(llama_cpp_path)
        elif base_lm == 'llama2':
            base_model = Llama2Model(llama_2_ckpt, llama_size, max_batch_size=batch_size) #Llama2Model
        else:
            assert False, f'cannot resolve {base_lm=}'
        prompt_tokens = base_model.tokenizer.encode(interactive_prompt['input'],bos=False,eos=False)
        if local_rank == 0:
            with open('input.txt', 'w') as f:
                print(len(prompt_tokens), file=f)
        rap_AQuA(base_model=base_model,
                  interactive_prompt=interactive_prompt,
                  useful_prompt=useful_prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)

    """
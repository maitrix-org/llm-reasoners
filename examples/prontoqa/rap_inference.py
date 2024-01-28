import itertools
import os
from typing import Optional, Sequence, Any
import json
import fire
from tqdm import tqdm
import pickle

from dataset import ProntoQADataset, ProntoQAProblem, ProntoQAExample
from reasoners import LanguageModel, SearchAlgorithm, Reasoner

from reasoners.lm import llama_cpp_model
from reasoners.visualization import visualize
from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAState, ProntoQAAction
from reasoners.algorithm import MCTS
from reasoners.benchmark import ProntoQAEvaluatorFinal

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def main(model_dir: str=  os.environ['LLAMA2_CKPTS'],
           mem_map: str = [16, 22],
           **search_algo_params):
    import torch, os
    import numpy as np
    from reasoners.lm import ExLlamaModel 
    language_model = ExLlamaModel(model_dir,
                                lora_dir=None, 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=mem_map,
                                log_output=True)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

    with open('examples/prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    world_model = ProntoQAWorldModel(base_model=language_model)
    search_config = ProntoQAConfig(base_model=language_model)
    search_algo = MCTS(output_trace_in_each_iter=True, cum_reward=np.mean, **search_algo_params)
    reasoner =  Reasoner(
            world_model=world_model,
            search_config=search_config,
            search_algo=search_algo
        )

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="rap",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/prontoqa/data/345hop_random_true.json'
        )
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4, log_dir="pronto_logs/")
    print(f"accuracy: {accuracy}")


if __name__ == '__main__':
    fire.Fire(main)
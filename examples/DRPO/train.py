import pickle
import os
import sys
from typing import Type, Optional
import fire
import logging
import random

import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import pytz

from reasoners import LanguageModel as Model, SearchAlgorithm, Reasoner
#Reimplemented the Beam Search Algorithm in the llm-reasoner to accommodate the new dynamic max depth setting
from search_algo import BeamSearch

from reasoners.algorithm import MCTS
from models import VLLMModel, OpenAIChatModel

from world_model import PromptAlignWorldModel
from search_config import PromptAlignSearchConfig

from utils import parse_algo_output

#TODO: update GPU numbers if running out of CUDA memory.
os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3,4"

def interleave_data_arrays(alpaca_data, lima_data, mal_data):
    total_data = len(lima_data) + len(mal_data) + len(alpaca_data)

    frac_lima = max(int(10*(len(lima_data)/total_data)), 1)
    frac_mal = max(int(10*(len(mal_data)/total_data)), 1)
    frac_alpaca = max(int(10*(len(alpaca_data)/total_data)), 1)

    interleaved_data = []

    lima_ptr = 0
    mal_ptr = 0
    alpaca_ptr = 0

    addLimaData = True
    addMalData = True
    addAlpacaData = True

    while( addLimaData or addMalData or addAlpacaData ):

        if addLimaData:
            if ((lima_ptr+1)*frac_lima - 1 < len(lima_data)):
                interleaved_data.extend(lima_data[lima_ptr*frac_lima: (lima_ptr+1)*frac_lima])
                lima_ptr += 1
            else:
                interleaved_data.extend(lima_data[lima_ptr*frac_lima:])
                addLimaData = False

        if addMalData:
            if ((mal_ptr+1)*frac_mal - 1 < len(mal_data)):
                interleaved_data.extend(mal_data[mal_ptr*frac_mal: (mal_ptr+1)*frac_mal])
                mal_ptr += 1
            else:
                interleaved_data.extend(mal_data[mal_ptr*frac_mal:])
                addMalData = False


        if addAlpacaData:
            if ((alpaca_ptr+1)*frac_alpaca - 1 < len(alpaca_data)):
                interleaved_data.extend(alpaca_data[alpaca_ptr*frac_alpaca: (alpaca_ptr+1)*frac_alpaca])
                alpaca_ptr += 1
            else:
                interleaved_data.extend(alpaca_data[alpaca_ptr*frac_alpaca:])
                addAlpacaData = False
    

    return interleaved_data


def run_prompt_align(base_model: Model,
                metrics_model: Model,
                eval_model: Model,
                optimize_model: Model,
                initial_system_prompt: str,
                search_algo: Type[SearchAlgorithm] = BeamSearch,
                n_actions: int = 16,
                temperature: float = 0.7, # for optimize_model
                depth: int = 16,
                max_depth_increase= 10,
                beam_size: int = 5,
                num_training_examples: int = 25,
                log_dir: Optional[str] = None,
                disable_log: bool = False,
                disable_tqdm: bool = False,
                data_dir: str = None,
                metrics_cache_path: str = None,
                ret_icl = True,
                is_GPT = False,
                k = 2,
                **search_algo_params):
        
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/prompt_align_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
    
    # search algo
    search_algo_params |= {
        'beam_size': beam_size,
        'max_depth': depth,
        'reward_aggregator': 'mean' if beam_size > 1 else 'last',
        'max_depth_increase': max_depth_increase
    }

    if 'f' in search_algo_params:
        del search_algo_params['f']
        
    world_model = PromptAlignWorldModel(
        base_model=base_model,
        metrics_model=metrics_model,
        eval_model=eval_model,
        initial_system_prompt=initial_system_prompt,
        depth=depth,
        metrics_cache_path=metrics_cache_path,
        ret_icl= ret_icl,
        is_GPT=is_GPT,
        k = k
    )
    
    search_config = PromptAlignSearchConfig(
        optimize_model=optimize_model,
        n_actions=n_actions,
        temperature=temperature
    )
    
    search_algo = search_algo(**search_algo_params)
    
    reasoner = Reasoner(world_model=world_model, search_config=search_config, search_algo=search_algo)
    
    # LIMA Subsampled Training data
    with open(os.path.join(data_dir, 'lima_subsample_train.json'), 'r') as f:
        lima_data = json.load(f)

    lima_data = [dat['query'] for dat in lima_data]

    # Malicious (Safety) Training Data
    safety_data = []

    with open(os.path.join(data_dir, 'mal_train.txt'), 'r') as file:
        while line := file.readline():
            safety_data.append(line.strip())

    # Alpaca train data
    with open(os.path.join(data_dir, 'alpaca_train.json'), 'r') as f:
        alpaca_data = json.load(f)

    alpaca_data = [dat['instruction'] for dat in alpaca_data]

    # Getting the sample of training data ready
    total_possible_data = len(alpaca_data) + len(safety_data) + len(lima_data)
    fraction_to_be_used = num_training_examples/total_possible_data

    num_alpaca = int(fraction_to_be_used*len(alpaca_data))
    num_lima = int(fraction_to_be_used*len(lima_data))
    num_safety = int(fraction_to_be_used*len(safety_data))

    diff = num_training_examples - (num_safety + num_alpaca + num_lima)

    for i in range(diff):
        if i%3 == 0:
            num_alpaca += 1
        elif i%3 == 1:
            num_lima += 1
        else:
            num_safety += 1

    # Arranging the samples in a fashion that model encounters each kind of data regularly
    examples = interleave_data_arrays(random.sample(alpaca_data, num_alpaca),
                random.sample(lima_data, num_lima),
                random.sample(safety_data, num_safety ))
    
    
    logging.info(f'Loaded {len(examples)} examples')
    
    # shuffle the examples with seed 42
    np.random.seed(42)
    np.random.shuffle(examples)
    
    logging.info(f'Examples shuffled with seed 42')
    
    # run the reasoner
    algo_output = reasoner(example=examples)
    
    if not disable_log:
        with open(os.path.join(log_dir, 'algo_output', 'output.pkl'), 'wb') as f:
            pickle.dump(algo_output, f)

        # get current time (california time) format: yyyy-mm-dd-hh-mm-ss
        california_tz = pytz.timezone('America/Los_Angeles')
        california_time = datetime.now(california_tz)

        # Format the time as requested
        formatted_time = california_time.strftime('%Y-%m-%d-%H-%M-%S')
        
        # output the trace of how the system prompt evolves
        with open(os.path.join(log_dir, 'algo_output', f'trace_{formatted_time}.txt'), 'w') as f:
            for i, sub_result in enumerate(parse_algo_output(algo_output)):
                f.write("-"*20 + f" Step {i} " + "-"*20 + "\n")
                f.write(sub_result + "\n")
                f.write("-"*50 + "\n")
        

def main(
    base_model_name: str = 'mistralai/Mistral-7B-v0.1',
    base_model_family: str = 'mistral',
    eval_model_name: str = 'gpt-4-0125-preview',
    metrics_model_name: str = 'gpt-4-0125-preview',
    optimize_model_name: str = 'gpt-4-0125-preview',
    initial_system_prompt: str = "You are a helpful assistant.",
    n_actions: int = 3,
    temperature: float = 0.2,
    depth: int = 20,
    max_depth_increase: int = 10,
    beam_size: int = 2,
    log_dir: Optional[str] = "logs/mistral-7b-chain",       
    disable_log: bool = False,
    disable_tqdm: bool = False,
    base_model_download_dir = "./tmp",
    data_dir = './data',
    metrics_cache_path: str = "data/metrics_cache.json",
    num_training_examples: int = 180,
    logging_level: str = "INFO",
    ret_icl = True,
    is_GPT = False,
    k = 2,
    **kwargs
):
    # if log_dir is not None, create the directory
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
    # if metrics_cache_path is not None and it does not exist, create it
    if metrics_cache_path is not None and not os.path.exists(metrics_cache_path):
        with open(metrics_cache_path, "w") as f:
            json.dump({}, f)
        
    # set up logging
    if not disable_log:
        logging_text_file = os.path.join(log_dir, 'log.txt')
        
        # clear it anyway
        with open(logging_text_file, 'w'):
            pass
        
        logging.basicConfig(
            level=logging.INFO if logging_level == "INFO" else logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(logging_text_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    # load the models, if multiple models have the same name, we do not reload multiple times

    if base_model_family.lower() == 'mistral':
        base_model = VLLMModel(model_name=base_model_name, download_dir=base_model_download_dir, gpu_memory_utilization=0.9,num_gpus=4)
    elif base_model_family.lower() == 'llama':
        is_awq = (base_model_name.split('-')[-1].lower() == 'awq')
        if is_awq:
            base_model = VLLMModel(
                model_name=base_model_name, 
                quantization="awq",
                dtype="auto",
                num_gpus=1,
                gpu_memory_utilization=0.7,
                download_dir= base_model_download_dir   
                )
        else:
            base_model = VLLMModel(model_name=base_model_name, download_dir=base_model_download_dir, gpu_memory_utilization=0.5)
    elif base_model_family.lower() == 'gpt':
        is_GPT = True
        base_model = OpenAIChatModel(model_name=base_model_name)
    
    
    # Initialize a dictionary to hold model instancesbase_model
    models = {}

    # Always create the eval model
    models['eval'] = OpenAIChatModel(model_name=eval_model_name)

    # Reuse the eval model for optimize and metrics models if their names match, otherwise create new instances
    for model_type, model_name in [('optimize', optimize_model_name), ('metrics', metrics_model_name)]:
        if model_name in models.values():
            # Reuse the existing model instance if the name matches
            models[model_type] = models['eval']
        else:
            # Create a new model instance if the name does not match
            models[model_type] = OpenAIChatModel(model_name=model_name)

    # Access models as needed
    eval_model = models['eval']
    optimize_model = models['optimize']
    metrics_model = models['metrics']
    
    # determine whether initial_system_prompt is a file path
    if os.path.exists(initial_system_prompt):
        with open(initial_system_prompt, 'r') as f:
            initial_system_prompt = f.read()
    
    run_prompt_align(
        base_model=base_model,
        eval_model=eval_model,
        metrics_model=metrics_model,
        optimize_model=optimize_model,
        initial_system_prompt=initial_system_prompt,
        n_actions=n_actions,
        temperature=temperature,
        num_training_examples=num_training_examples,
        depth=depth,
        max_depth_increase=max_depth_increase,
        beam_size=beam_size,
        log_dir=log_dir,
        disable_log=disable_log,
        disable_tqdm=disable_tqdm,
        data_dir=data_dir,
        metrics_cache_path=metrics_cache_path,
        ret_icl = ret_icl,
        is_GPT= is_GPT,
        k = k,
        **kwargs
    )  
    
if __name__ == '__main__':
    fire.Fire(main)
import pickle
from typing import Type, Callable, Optional

import numpy as np
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS

from world_model import BlocksWorldModel
from search_config import BWConfig
import utils

def rap_bw(base_model: LanguageModel,
           prompt: dict,
           search_algo: Type[SearchAlgorithm] = MCTS,
           data_path: str = 'data',
           resume: int = 0,
           depth_limit: int = 6,
           reward_alpha: float = 0.5,
           batch_size = 1,
           goal_reached_reward = 100,
           goal_reward_default = 0.,
           cum_reward: Callable[[list[float]], float] = sum,
           calc_q: Callable[[list[float]], float] = np.mean,
           log_dir: Optional[str] = None,
           disable_log: bool = False,
           domain_file: str = "",
           config_file: str = "",
           lm_plan_file: str = 'lm_plan.tmp',
           **search_algo_params):

    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/bw_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, "depth_limit": depth_limit, "disable_tqdm": False}
    world_model = BlocksWorldModel(base_model=base_model, prompt=prompt, batch_size=batch_size, max_steps=depth_limit)
    config = BWConfig(base_model=base_model, prompt=prompt, batch_size=batch_size,
                      reward_alpha=reward_alpha, goal_reached_reward=goal_reached_reward,
                      goal_reward_default=goal_reward_default)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    dataset = utils.load_blocksworld(config_file, domain_file, data_path, prompt)  # [{"goal": str, "init": str}]
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume, desc='Blocksworld')):
        algo_output = reasoner(example)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # to make sure the plan is saved before evaluation in multi-process setting
        if algo_output.trace is None:
            print("No plan found")
            correct = 0
        else:
            utils.text_to_plan_blocksworld("\n".join(algo_output.trace[1]), example["instance_file"], config_file, domain_file, lm_plan_file)
            correct = utils.validate_plan(domain_file, example["instance_file"], lm_plan_file)[0]

        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{resume + i + 1}: {correct=}; '\
          f'{accuracy=:.3f} ({correct_count}/{i + 1})'
        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
            with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)
            question = {"goal": utils.extract_goals(example, return_raw=True), "init": utils.extract_init_state(example)}
            # append the question to the jsonl file
            with open(os.path.join(log_dir, 'questions.jsonl'), 'a') as f:
                print(json.dumps(question), file=f)


if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random
    import torch
    import torch.backends.cudnn
    from reasoners.lm import LlamaModel, Llama2Model
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    def llama_main(llama_size: str = '13B',
             prompt_path: str = 'examples/rap_blocksworld/prompts/prompt.json',
             data_path: str = 'examples/rap_blocksworld/data/step_4.json',
             disable_log: bool = False,
             config_file: str = "examples/rap_blocksworld/data/bw_config.yaml",
             domain_file: str = "examples/rap_blocksworld/data/generated_domain.pddl",
             lm_plan_file: str = 'lm_plan.tmp',
             depth_limit: int = 6,
             **kwargs):

        from reasoners.lm import LlamaModel
        local_rank = int(os.environ["LOCAL_RANK"])
        llama_ckpts = os.environ["LLAMA_CKPTS"]
        with open(prompt_path) as f:
            prompt = json.load(f)
        llama_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=2)
        rap_bw(llama_model,
               prompt,
               disable_log=disable_log or local_rank != 0,
               data_path=data_path,
               config_file=config_file,
               domain_file=domain_file,
               depth_limit=depth_limit,
               lm_plan_file=lm_plan_file, **kwargs)


    def llamacpp_main(
            llama_path = '/home/shibo/llama.cpp/models/65B/ggml-model-q8_0.bin',
            prompt_path: str = 'examples/rap_blocksworld/prompts/prompt.json',
            data_path: str = 'examples/rap_blocksworld/data/step_4.json',
            disable_log: bool = False,
            config_file: str = "examples/rap_blocksworld/data/bw_config.yaml",
            lm_plan_file: str = 'lm_plan.tmp',
            domain_file: str = "examples/rap_blocksworld/data/generated_domain.pddl",
            depth_limit: int = 6,
            **kwargs):

        from reasoners.lm import LlamaCppModel
        with open(prompt_path) as f:
            prompt = json.load(f)
        llama_model = LlamaCppModel(path=llama_path)
        rap_bw(llama_model,
               prompt,
               disable_log=disable_log,
               data_path=data_path,
               config_file=config_file,
               domain_file=domain_file,
               depth_limit=depth_limit,
               lm_plan_file=lm_plan_file, **kwargs)

    def llama_hf_main(
            llama_path = '/data/haotian/RAP_tune/Llama-2-7b-hf',
            peft_path = None,
            prompt_path: str = 'examples/rap_blocksworld/prompts/prompt.json',
            data_path: str = 'examples/rap_blocksworld/data/step_4.json',
            disable_log: bool = False,
            config_file: str = "examples/rap_blocksworld/data/bw_config.yaml",
            domain_file: str = "examples/rap_blocksworld/data/generated_domain.pddl",
            lm_plan_file: str = 'lm_plan.tmp',
            depth_limit: int = 6,
            quantized = "nf4", # awq, int8, fp4, nf4, None
            load_awq_pth = None,
            **kwargs
            ):
        from reasoners.lm import HFModel #maybe other transformer models also support, we have not check that
        with open(prompt_path) as f:
            prompt = json.load(f)
        device = torch.device("cuda:0")
        llama_model = HFModel(llama_path, llama_path, device=device, max_batch_size=1, max_new_tokens=512, quantized=quantized, peft_pth=peft_path, load_awq_pth=load_awq_pth)
        rap_bw(llama_model,
               prompt,
               disable_log=disable_log,
               data_path=data_path,
               config_file=config_file,
               domain_file=domain_file,
               depth_limit=depth_limit,
               lm_plan_file=lm_plan_file, **kwargs)
    #for exllama use please refer to https://github.com/turboderp/exllama and put it under /llm-reasoners/
    def exllama_main(
            model_dir = '/data/haotian/RAP_tune/Llama-2-70B-GPTQ',
            lora_dir = None,
            prompt_path: str = 'examples/rap_blocksworld/prompts/prompt.json',
            data_path: str = 'examples/rap_blocksworld/data/step_4.json',
            disable_log: bool = False,
            config_file: str = "examples/rap_blocksworld/data/bw_config.yaml",
            domain_file: str = "examples/rap_blocksworld/data/generated_domain.pddl",
            lm_plan_file: str = 'lm_plan.tmp',
            depth_limit: int = 6,
            batch_size: int = 1,
            mem_map = None,
            **kwargs
            ):
        print(model_dir)
        from reasoners.lm import ExLlamaModel  # Maybe other transformer models also support
        with open(prompt_path) as f:
            prompt = json.load(f)
        device = torch.device("cuda:0")
        llama_model = ExLlamaModel(model_dir, 
                                   lora_dir, 
                                   device=device, 
                                   max_batch_size=batch_size, 
                                   max_new_tokens=200, 
                                   max_seq_length=2048, 
                                   mem_map=mem_map)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs
        rap_bw(llama_model,
               prompt,
               disable_log=disable_log,
               data_path=data_path,
               config_file=config_file,
               domain_file=domain_file,
               depth_limit=depth_limit,
               lm_plan_file=lm_plan_file,
               batch_size=batch_size, **kwargs)
    
    def llama2_main(llama_size: str = '70B',
             prompt_path: str = 'examples/rap_blocksworld/prompts/prompt.json',
             data_path: str = 'examples/rap_blocksworld/data/step_4.json',
             disable_log: bool = False,
             config_file: str = "examples/rap_blocksworld/data/bw_config.yaml",
             domain_file: str = "examples/rap_blocksworld/data/generated_domain.pddl",
             lm_plan_file: str = 'lm_plan.tmp',
             depth_limit: int = 6,
             **kwargs):

        from reasoners.lm import Llama2Model
        local_rank = int(os.environ["LOCAL_RANK"])
        llama2_ckpts = os.environ["LLAMA_2_CKPTS"]
        with open(prompt_path) as f:
            prompt = json.load(f)
        llama_model = Llama2Model(llama2_ckpts, llama_size, max_batch_size=1)
        rap_bw(llama_model,
               prompt,
               disable_log=disable_log or local_rank != 0,
               data_path=data_path,
               config_file=config_file,
               domain_file=domain_file,
               depth_limit=depth_limit,
               lm_plan_file=lm_plan_file, **kwargs)


    fire.Fire(exllama_main) # user will need to switch the model in the code

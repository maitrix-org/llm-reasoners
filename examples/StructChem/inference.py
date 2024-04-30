import torch
import random
import warnings
import pickle
import os
import io
import sys
import json
from typing import Type, Optional, Literal
import numpy as np
# from datasets import load_dataset
from tqdm import tqdm
from datetime import datetime
import fire
import transformers

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import BeamSearch
from collections import Counter

from world_model import StructChemWorldModelF, StructChemWorldModelR
from search_config import StructChemConfigF, StructChemConfigR
from prompt import initial_instruction
from utils import extract_formulae_reasoning, judge_answer

# from utils import judge_answer, majority_voting, parse_answer

from reasoners.lm.openai_model import OpenAIModel
from reasoners.benchmark import GSM8KEvaluator
from reasoners.lm.hf_model import HFModel

local_rank = int(os.environ.get("LOCAL_RANK", 0))


def structChem(base_model: LanguageModel,
                search_algo: Type[SearchAlgorithm] = BeamSearch,
                resume: int = 0,
                depth_limit: int = 16,
                temperature: float = 0.8,
                beam_size: int = 1,
                max_depth: int = 16,
                data_path: str = "examples/structChem/chemmc.json",
                log_dir: Optional[str] = None,
                disable_log: bool = False,
                disable_tqdm: bool = False,
                **search_algo_params):
    
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/structChem_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
    
    # set parameters for beam search
    search_algo_params |= {
            'beam_size': beam_size, 
            'max_depth': max_depth,
            }
    search_algo = search_algo(**search_algo_params)
    
    # Initialization for reasoners - Formulae
    world_model = StructChemWorldModelF(base_model=base_model, temperature=temperature)
    config = StructChemConfigF(base_model=base_model,
                            temperature=temperature,
                            depth_limit=depth_limit)
    reasoner_f = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    # Initialization for reasoners - Reasoning
    world_model = StructChemWorldModelR(base_model=base_model, temperature=temperature)
    config = StructChemConfigR(base_model=base_model,
                            temperature=temperature,
                            depth_limit=depth_limit)
    reasoner_r = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    with open(data_path, 'r') as f:
        dataset = json.load(f)

    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
                                     desc='StructChem', disable=disable_tqdm)):

        seed_id = 0
        np.random.seed(seed_id)
        random.seed(seed_id)
        torch.manual_seed(seed_id)
        torch.cuda.manual_seed(seed_id)
        torch.backends.cudnn.deterministic = True

        ### 1. Query the model using the overall instruction to get the initial 
        ###    formulae and reasoning process

        problem = example['problem_text']

        with io.StringIO() as f:
            f.write(initial_instruction.strip()+"\n\n")
            f.write(f"Now try to solve the following problem:\n{problem}")
            model_input = f.getvalue()
        
        output = base_model.generate(
            [model_input],
            max_new_tokens=512,
            hide_input=False,
            do_sample=True,
            top_k=32000,
            top_p=0.95,
            temperature=temperature,
            eos_token_id='\n',
            num_return_sequences=1
        ).text[0].strip()
        
        formulae, reasoning = extract_formulae_reasoning(output)

        ### 2. Leverage reasoning and search_algo for the best formulae and reasoning,
        ###    respectively, with confidence scores.
        
        ## 2.1 Initiate the refinement process for formulae in a tree-searching style
        algo_output = reasoner_f(example['problem_text'] + "<concatenate>" + formulae)
        # get the last state
        state = algo_output.terminal_state[-1]
        # get the final refined formulae
        output_formulae = state.formulae

        ## 2.2 Initiate the refinement process for reasoning in a tree-searching style
        algo_output = reasoner_r(example['problem_text'] + "<concatenate>" + output_formulae + "<concatenate>" + reasoning)
        # get the last state
        state = algo_output.terminal_state[-1]
        # get the final refined reasoning
        output_reasoning = state.reasoning

        ### 3. With the refined formulae and reasoning, get the final answer. 
        with io.StringIO() as f:
            f.write(initial_instruction.strip()+"\n\n"+problem+"\n\n" +output_formulae+"\n\n"+output_reasoning)
            model_input = f.getvalue()
        
        output_ans = base_model.generate(
            [model_input],
            max_new_tokens=512,
            hide_input=False,
            do_sample=True,
            top_k=32000,
            top_p=0.95,
            temperature=temperature,
            eos_token_id='\n',
            num_return_sequences=1
        ).text[0].strip()

        if local_rank == 0:
            with open("logs/structChem/log.txt", "a") as f:
                # print the distribution of outputs, counter in one line
                print(f"\nStructChem ANSWER: {output_ans}\n", file=f)

        # get the answer from the dataset
        answer = example["answer_number"]
        unit = example['unit']
        # judge the answer
        correct = judge_answer(output, answer, unit)
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


def main(base_lm:Literal['hf', 'google', 'openai', 'anthropic','exllama'],
         model_dir = None, 
         batch_size = 1, 
         prompt = "examples/StructChem/prompt.py", 
         data_path = "examples/StructChem/chemmc.json",
         resume = 0, 
         log_dir = None, 
         temperature = 0, 
         n_sc = 1, 
         disable_log: bool = False,
         disable_tqdm: bool = False,
         quantized = 'int8',
         **kwargs):

    if base_lm == "openai":
        base_model = OpenAIModel("gpt-4", additional_prompt="ANSWER")
    elif base_lm == "hf":
        base_model = HFModel(model_dir, model_dir, quantized=quantized)

    structChem(base_model=base_model,
                        resume=resume,
                        temperature=temperature,
                        data_path=data_path,
                        log_dir=log_dir,
                        disable_log=disable_log,
                        disable_tqdm=disable_tqdm,
                        **kwargs)


if __name__ == '__main__':
    fire.Fire(main(base_lm='openai'))
"""
CUDA_VISIBLE_DEVICES=2 python examples/cot_gsm8k/inference.py \
--model_dir $Gemma_ckpts \ 
"""



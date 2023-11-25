from typing import Type, Callable, Optional, Literal

import numpy as np

from reasoners.benchmark import GSM8KEvaluator

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import  GreedySearch, GreedySearchNode, GreedySearchResult

from world_model import GSM8kWorldModel, GSM8kState, GSM8kAction
from search_config import GSM8kConfig
import utils


#grace imports
import os
import random
import time
import pickle
import math
from argparse import ArgumentParser
from collections import namedtuple

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from constants import *
from grace_utils.t5_discriminator import T5Discriminator, T5EnergyDiscriminator
# from verifier_utils.t5_verifier import T5Verifier
from grace_utils.reason import generate_guided_reasoning
import json
from data_utils.utils import prepare_icl_input, create_demos, is_correct, extract_answer, is_correct_program, extract_answer_program, extract_answer_llc
import wandb
from collections import Counter
from data_utils.utils import evaluate
from torch.nn.utils.rnn import pad_sequence
from grace_utils.args import TASKS


# def node_visualizer(x: GreedySearchNode[GSM8kState, GSM8kAction]):
#     if not x.state:
#         return {}
#     return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


def main(args):
    if args.model_tokenizer_path is None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer_path)
    

    ## set padding_side to left for llama tokenizer
    if 'llama' in args.model_name_or_path:
        tokenizer.padding_side = 'left'
    
    print("Loading model from {}".format(args.model_name_or_path))
    if 't5' in args.model_name_or_path:
        model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, 
                                                       torch_dtype = torch.bfloat16 if args.bf16 else torch.float32).to(args.device1)
    elif 'llama' in args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, 
                                                       torch_dtype = torch.bfloat16 if args.bf16 else torch.float32,
                                                       load_in_8bit=args.eightbit,device_map="auto" if args.eightbit else None)
                                                       
        tokenizer.pad_token = tokenizer.eos_token
    
    if not args.eightbit:
        model = model.to(args.device1)
    
    model.eval()

    if args.disc_path is not None and not args.generator_only:
        ## load discriminator tokenizer 
        print("loading discriminator tokenizer from {}".format(args.disc_path))
        disc_tokenizer = T5Tokenizer.from_pretrained(args.disc_path)
        
        print("Loading discriminator from {}".format(args.disc_path))
        if os.path.isdir(args.disc_path):
            for _, _, files in os.walk(args.disc_path):
                for fname in files:
                    if 'best' in fname or 'pytorch_model.bin' in fname:
                        args.disc_path = os.path.join(args.disc_path, fname)
                        break
        
        ckpt = torch.load(args.disc_path)
        ## loading discriminator weights
        if args.generation_type == 'step':
            discriminator = T5Discriminator(model_name_or_path=ckpt['args'].model, 
            device=args.device2, args=ckpt['args'])
            discriminator.t5.resize_token_embeddings(len(disc_tokenizer))
            discriminator.load_state_dict(ckpt['state_dict'])
        elif args.generation_type in ['step-score', 'step-qrs']:
            disc_backbone = 'google/flan-t5-large'
            discriminator = T5EnergyDiscriminator(model_name_or_path=disc_backbone, 
            device=args.device2, args=args)
            discriminator.model.resize_token_embeddings(len(disc_tokenizer))
            discriminator.load_state_dict(ckpt)

        elif args.generation_type == 'token':
            assert len(tokenizer) == len(disc_tokenizer) - 2, "model and discriminator tokenizers are not the same. They have to be the same for token-level generation"

        #if len(tokenizer) != len(disc_tokenizer) - 2: # two for [SEP] and [CLS]
        #    print("WARNING: model and discriminator tokenizers are not the same. Will use a hack to make it work")
        
        discriminator.eval()

    demos = None
    eval_examples = []
    with open(args.in_file, 'r') as rf:
        for line in rf:
            d = json.loads(line)
            eval_examples.append(d)
    
    n_examples = args.n_examples if args.n_examples is not None else len(eval_examples)
    end_idx = min(args.start_idx + n_examples, len(eval_examples))
    eval_examples = eval_examples[args.start_idx:end_idx]
    
    if args.icl:
        ## assert demos are not in the inputs
        demos = []
        ## load demos 
        print("Using In-context learning with {} demos".format(args.n_demos))
        ## in file location 
        data_path = os.path.join(*args.in_file.split('/')[:-1])
        demos_path = os.path.join(data_path, args.demos_file_name)

        with open(demos_path, "r") as f:
            for line in f:
                demo = json.loads(line)
                demos.append(demo)
                assert demo['question'] not in [a['question'] for a in eval_examples], "The demo {} is in the eval examples!".format(demo['question'])
        ## process demos
        demos = create_demos(demos, step_delimiter=args.step_delimiter, add_delimiter_after_demo=True)[:args.n_demos]

    #inputs = [inputs[1]]
    #gt_outputs = [gt_outputs[1]]
    print("Task: {}".format(args.task))

    text_table = wandb.Table(columns=["question", "gold answer", "solution"])
    
    if args.generator_only:
        print("Evaluating generator only...")
        eval_d = evaluate(model, tokenizer, eval_examples, demos=demos, instruction=args.instruction, 
        task=args.task, args=args)
        print("solve rate w/o discriminator: {:.2f}".format(eval_d['eval_acc'] * 100))

        wandb.log({"solve_rate": eval_d['eval_acc']})
        for ex, gen in zip(eval_examples, eval_d['generated_solutions']):
            text_table.add_data(ex['question'], ex['answer'], gen)
        
        wandb.log({"outputs": text_table})
        return

    if args.task in ["mathqa"]:
        assert args.step_delimiter.strip() == ';', "Step delimiter for {} should be ;"
    elif args.task in ["gsm8k", "svamp", "multiarith", "coin_flip"] and not args.icl:
        ## not few-shot 
        assert args.step_delimiter.strip() == '|', "Step delimiter for {} should be |".format(args.task)
    
    extract_answer_fn = None 
    if args.task in ['arithmetic', 'gsm8k', 'svamp', 'multiarith', 'coin_flip']:
        extract_answer_fn = extract_answer
    elif args.task in ['asdiv', 'mathqa']:
        extract_answer_fn = extract_answer_program
    elif args.task in ['last_letter_concatenation']:
        extract_answer_fn = extract_answer_llc
    else:
        raise NotImplementedError("Task {} not implemented!".format(args.task))
    

    print("Running guided decoding...")

    solve_rate = 0

    if args.use_verifier:
        all_solutions = []

    count=0
    for inp in tqdm(eval_examples):
        count+=1
        # if count>1:
        #     break
        qn, gt_ans = inp['question'], (inp['answer'] if 'answer' in inp else inp['gt_sol'])

        qn_with_input = prepare_icl_input(qn, demos=demos, instruction=args.instruction)
        disc_input = qn
        if not disc_input.startswith(Q_DELIM):
            disc_input = Q_DELIM + " " + disc_input
        
        n_samples = args.n_verifier_samples if args.use_verifier else args.n_self_consistency
        gen_sols = []

        print(f"qn_with_input: {qn_with_input}")
        print(f"disc_input: {disc_input}")
        print(f"num_samples: {n_samples}")
        for _ in range(n_samples):
            results = generate_guided_reasoning(
                            model=model, 
                            model_tokenizer=tokenizer,
                            discriminator=discriminator,
                            disc_tokenizer = disc_tokenizer,
                            model_input_text=qn_with_input,
                            disc_input_text=disc_input, 
                            n_candidate_steps=args.n_candidate_steps,
                            beta=args.beta,
                            generation_type=args.generation_type,
                            args=args,
                            )
    
            sol = results[0]
            gen_sols.append(sol)
        print(f"solution: {sol}")
        if count >5:
            raise Exception
        
       
        all_solutions.append({"question": qn_with_input, 
                                    "solutions": gen_sols,
                                    "gt_ans": gt_ans})
                                
    
    for row in text_table.data:
        print()
        formatted_row = " | ".join(str(item) for item in row)
        print(formatted_row)
    print("Solve rate = {:.2f}".format(solve_rate*100 / len(eval_examples)))


def grace_gsm8k(base_model: LanguageModel,
            #   prompt: GSM8kPromptDict,
            #   useful_prompt: GSM8kUsefulPrompt,
              search_algo: Type[SearchAlgorithm] = GreedySearch,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 2,
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
              **search_algo_params):

    aggregator = None

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm,
                           'output_trace_in_each_iter': output_trace_in_each_iter,
                        #    'node_visualizer': node_visualizer, 
                           'aggregator': aggregator
                           }
    world_model = GSM8kWorldModel(base_model=base_model,
                                  n_confidence=n_confidence, batch_size=batch_size, temperature=temperature,
                                  early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = GSM8kConfig(base_model=base_model, 
                         n_actions=n_action, batch_size=batch_size, temperature=temperature,
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = GSM8KEvaluator(output_extractor=utils.retrieve_answer,
                               answer_extractor=utils.retrieve_answer_from_dataset,
                            #    init_prompt=prompt,
                               sample_prompt_type="cot",
                               disable_log=disable_log,
                               disable_tqdm=disable_tqdm)

    accuracy = evaluator.evaluate(reasoner, num_shot=4, resume=resume, log_dir=log_dir)
    print(accuracy)


if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')


    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama', 'google/flan-t5-large'] = 'llama-2',
             llama_ckpts: str = llama_ckpts,
             llama_2_ckpts: str = llama_2_ckpts,
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
             useful_prompt: str = 'examples/rap_gsm8k/prompts/useful_examples.json',
             prompt: str = 'examples/rap_gsm8k/prompts/prompt_pool.json',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
      
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        with open(prompt) as f:
            prompt = json.load(f)
        if base_lm in ['llama', 'llama2']:
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
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                      max_batch_size=batch_size, max_new_tokens=200, max_seq_length=3072)
        elif base_lm == "google/flan-t5-large":
            from reasoners.lm import FlanT5Model
            base_model = FlanT5Model()

        else:
            assert False, f'cannot resolve {base_lm=}'

        grace_gsm8k(base_model=base_model,
                  useful_prompt=useful_prompt,
                  prompt=prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)


    fire.Fire(main)

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

from collections import Counter
from data_utils.utils import evaluate
from torch.nn.utils.rnn import pad_sequence
from grace_utils.args import TASKS



def grace_gsm8k(base_model: LanguageModel,
              search_algo: Type[SearchAlgorithm] = GreedySearch,
              resume: int = 0,
              n_action: int = 4,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              temperature: float = 0.8,
              reward_alpha: float = 0.5,
              log_dir: Optional[str] = None,
              disable_log: bool = False,
              disable_tqdm: bool = False,
              discriminator_path:str = None,
              **search_algo_params):


    ckpt = torch.load(os.path.join(discriminator_path,"pytorch_model.bin"))
    discriminator_tokenizer = T5Tokenizer.from_pretrained(discriminator_path)#"/data/adithya/ckpts/discrim/gsm8k/"

    disc_backbone = 'google/flan-t5-large'
    discriminator_model = T5EnergyDiscriminator(model_name_or_path=disc_backbone, 
    device="cuda")
    discriminator_model.model.resize_token_embeddings(len(discriminator_tokenizer))
    discriminator_model.load_state_dict(ckpt)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    discriminator_model.to(device)
    search_algo_params |= {  'max_depth': 5
                           }
    world_model = GSM8kWorldModel(base_model=base_model)
    config = GSM8kConfig(base_model=base_model, discriminator_tokenizer=  discriminator_tokenizer, discriminator_model =  discriminator_model,
                         reward_alpha=reward_alpha, 
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    evaluator = GSM8KEvaluator(output_extractor=utils.retrieve_answer,
                               answer_extractor=utils.retrieve_answer_from_dataset,
                               sample_prompt_type="grace",
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
    llama_3_ckpts = os.environ.get("LLAMA3_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')


    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'llama-3', 'hf', 'exllama', 'google/flan-t5-large'] = 'llama-2',
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
             disable_log: bool = False,
             disable_tqdm: bool = False,
             discriminator_path:str = None,
             **kwargs):
      
        if base_lm in ['llama', 'llama-2',"llama-3"]:
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
                                      max_batch_size=batch_size, max_new_tokens=200, max_seq_length=3072)
        elif  "flan-t5" in base_lm:
            import torch
            import torch.backends.cudnn
            np.random.seed(23)
            random.seed(23)
            torch.manual_seed(23)
            torch.cuda.manual_seed(23)
            torch.backends.cudnn.deterministic = True
            from flan_t5 import FlanT5Model
            base_model = FlanT5Model()

        else:
            assert False, f'cannot resolve {base_lm=}'

        grace_gsm8k(base_model=base_model,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  discriminator_path=discriminator_path,
                  **kwargs)


    fire.Fire(main)
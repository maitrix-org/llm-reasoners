from typing import Literal, Optional

from reasoners import LanguageModel
from prompts.game24 import standard_prompt
from datetime import datetime
import os
import sys
import utils
from tqdm import tqdm
import numpy as np


def cot_game24(base_model: LanguageModel, disable_log: bool = False, resume=0, **kwargs):
    if not disable_log:
        log_dir = f'logs/game24_cot/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
    dataset = utils.read_data(file='./examples/tot_game24/data/24.csv')[900:1000][resume:]
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=len(dataset), initial=0, desc='game24', disable=disable_log)):
        lm_input = standard_prompt.format(input=example)
        output = base_model.generate([lm_input], eos_token_id='\n', temperature=0., additional_prompt='CONTINUE').text[0].split('\n')[0]
        correct = utils.test_output(example, output)
        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{i + 1}: {correct=}, {output=} ; {accuracy=:.3f} ({correct_count}/{i + 1})'
        if not disable_log:
            tqdm.write(log_str)
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)

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

    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama'] = 'llama-2',
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
             prompts: str = 'examples/tot_game24/prompts/game24.json',
             openai_mode: str = 'gpt-4-1106-preview',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        with open(prompts) as f:
            prompts = json.load(f)
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
                                      max_batch_size=batch_size, max_new_tokens=512, max_seq_length=2048)
        elif base_lm == 'openai':
            from reasoners.lm import OpenAIModel
            base_model = OpenAIModel(openai_mode)
        elif base_lm == 'gemini':
            from reasoners.lm import BardCompletionModel
            base_model = BardCompletionModel('gemini-pro')
        elif base_lm == 'claude':
            from reasoners.lm import ClaudeModel
            base_model = ClaudeModel('claude-3-opus-20240229')
        else:
            assert False, f'cannot resolve {base_lm=}'
        cot_game24(base_model=base_model, disable_log=disable_log or local_rank > 0, kwargs=kwargs)


    fire.Fire(main)

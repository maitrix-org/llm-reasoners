import os
import sys
import json
import warnings
import fire
from reasoners.lm import LLaMAModel, LlamaCppModel, LlamaModel, ExLlamaModel
import random
from typing import Literal
import torch
import torch.backends.cudnn
from tqdm import tqdm
from utils import extract_final_answer, eval_output

llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
exllama_ckpt = os.environ.get("EXLLAMA_CKPT", None)
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank != 0:
    sys.stdout = open(os.devnull, 'w')
    warnings.filterwarnings('ignore')


def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama'] = 'exllama',
            llama_ckpt: str = llama_ckpts,
            llama_2_ckpt: str = llama_2_ckpts,
            exllama_ckpt: str = exllama_ckpt,
            llama_size: str = '70B',
            mem_map: list[int] = [16, 22],
            llama_cpp_path: str = None,
            self_consistency: int = 1,
            batch_size: int = 2,
            max_seq_len: int = 3072,
            prompt_path: str = 'examples/cot_strategyQA/prompt.json',
            data_file_path: str = 'examples/rap_strategyQA/data/strategyqa_test.json',
            disable_log: bool = False,
            disable_tqdm: bool = False,
            resume: int = 0,
            log_dir: str = "logs/cot_strategyqa-dev-70B-sc10",
            temperature: float = 0,
            **kwargs):
    # set base_lm = 'llama' and llama_ckpt = '13B/30B/65B' to use llama with torchscale
    # else set base_lm = 'llama.cpp' and llama_cpp_path = the checkpoint to use llama.cpp

    if base_lm == 'llama':
        base_model = LLaMAModel(llama_ckpt, llama_size, max_batch_size=batch_size, max_seq_len=max_seq_len)
    elif base_lm == 'llama.cpp':
        base_model = LlamaCppModel(llama_cpp_path)
    elif base_lm == 'llama2':
        base_model = LlamaModel(llama_2_ckpt, llama_size, max_batch_size=batch_size)
    elif base_lm == 'exllama':
        device = torch.device("cuda:0")
        base_model = ExLlamaModel(
            model_dir = f"{exllama_ckpt}/Llama-2-{llama_size}-GPTQ",
            lora_dir = None,
            device = "cuda:0",
            max_batch_size = batch_size,
            max_new_tokens = 256,
            max_seq_length = max_seq_len,
            mem_map = mem_map
        )
    
    # load the dataset
    with open(data_file_path, 'r') as f:
        dataset = json.load(f)
    
    dataset = dataset[resume:]

    ### write all answers to json for submission
    answer_dict = {}
    if os.path.isfile(os.path.join(log_dir, 'all_answers-1.json')):
        with open(os.path.join(log_dir, 'all_answers-1.json')) as f:
            answer_dict = json.load(f)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=resume >= 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
    
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
    
    # load the prompt
    with open(prompt_path, 'r') as f:
        prompt_pool = json.load(f)
        prompt_pool = prompt_pool['cot_pool']

    correct_count = 0
    disable_tqdm = disable_tqdm or \
        (torch.distributed.is_initialized() and torch.distributed.get_rank() != 0)
    for i, example in enumerate(tqdm(dataset,
                                        total=resume + len(dataset),
                                        initial=resume,
                                        desc='strategyqa',
                                        disable=disable_tqdm)):
    
        # random sample 4 prompts
        prompt_list = random.sample(prompt_pool, 4)

        # generate the prompts
        prompt = "\n\n".join(prompt_list)

        # get the question
        question = example['question']

        prompt = f"{prompt}\n\nQ: {question}\nA:"

        print("Q: ", question, flush=True)

        # Introduce a list to store answers for self-consistency
        answers = []

        for _ in range(self_consistency):
            # generate the answer
            output = base_model.generate(
                [prompt],
                max_new_tokens=256,
                do_sample=True,
                top_p=0.95,
                temperature=temperature,
                num_return_sequences=1,
                eos_token_id="\n"
            ).text[0]

            print(f"Attempt {_ + 1}: ", output, flush=True)

            try:
                output = extract_final_answer(output)
            except:
                output = ""
            
            answers.append(output)

        # Determine the most consistent answer (here, we simply choose the most frequent one)
        final_answer = max(answers, key=answers.count)

        output = final_answer

        if "answer" in example:
            answer = example["answer"]
            correct_count += eval_output(answer, output)
            accuracy = correct_count / (i + 1)
            log_str = f'Case #{resume + i + 1}: {correct_count=}, {output=}, {answer=};{accuracy=:.3f} ({correct_count}/{i + 1})'
        else:
            log_str = f'Case #{resume + i + 1}: QID {example["qid"]}, {output=};'

        tqdm.write(log_str)

        if (not disable_log) and \
            (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)

        answer_dict[example['qid']] = {"answer": output, "decomposition": [], "paragraphs": []}
        with open(os.path.join(log_dir, 'all_answers.json'), 'w') as f:
            json.dump(answer_dict, f, indent=2)


if __name__ == '__main__':
    fire.Fire(main)
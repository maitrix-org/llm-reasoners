import os
import sys
import json
import warnings
import fire
from reasoners.lm import LlamaCppModel, LlamaModel, ExLlamaModel, HFModel, Llama2Model
import random
from typing import Literal
import torch
import torch.backends.cudnn
from tqdm import tqdm
from utils import extract_final_answer, eval_output

# llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
# llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
# exllama_ckpt = os.environ.get("EXLLAMA_CKPT", None)
# local_rank = int(os.environ.get("LOCAL_RANK", 0))
# if local_rank != 0:
#     sys.stdout = open(os.devnull, 'w')
#     warnings.filterwarnings('ignore')


def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama'] = 'hf',
            llama_ckpt: str = None,
            llama_2_ckpt: str = None,
            exllama_model_dir: str = None,
            llama_size: str = None,
            mem_map: list[int] = None,
            llama_cpp_path: str = None,
            self_consistency: int = 1,
            batch_size: int = 1,
            max_seq_len: int = 3072,
            prompt_path: str = 'examples/cot_strategyQA/prompt.json',
            data_file_path: str = 'examples/rap_strategyQA/data/strategyqa_test.json',
            disable_log: bool = False,
            disable_tqdm: bool = False,
            resume: int = 0,
            log_dir: str = None,
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
            model_dir = f"{exllama_model_dir}/Llama-2-{llama_size}-GPTQ",
            lora_dir = None,
            device = "cuda:0",
            max_batch_size = batch_size,
            max_new_tokens = 256,
            max_seq_length = max_seq_len,
            mem_map = mem_map
        )
    elif base_lm == 'hf':
        base_model = HFModel(exllama_model_dir, exllama_model_dir,quantized='int8')
    from datetime import datetime
    log_dir =  f'logs/strategyqa_'\
                        f'cot/'\
                        f'{datetime.now().strftime("%m%d%Y-%H%M%S")}'
    model_type = exllama_model_dir.split('/')[-1]
    log_dir = log_dir + f'_{model_type}'
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
    do_sample = True
    if temperature == 0 and isinstance(base_model, HFModel):
        print("Greedy decoding is not supported by HFModel. Using temperature = 1.0 instead.")
        temperature == 1.0
        do_sample = False
    import transformers
    import pickle
    if isinstance(base_model.model, transformers.GemmaForCausalLM):
        eos_token_id = [108,109]
    else:
        assert isinstance(base_model.model, transformers.LlamaForCausalLM)
        eos_token_id = [13]
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
                do_sample=do_sample,
                temperature=temperature,
                num_return_sequences=1,
                eos_token_id=eos_token_id
            ).text[0]

            print(f"Attempt {_ + 1}: ", output, flush=True)

            try:
                output_ans = extract_final_answer(output)
            except:
                output_ans = ""
            
            answers.append(output_ans)

        # Determine the most consistent answer (here, we simply choose the most frequent one)
        if len(answers) == 0:
            final_answer = ""
        else:
            final_answer = max(answers, key=answers.count)



        output_ans = final_answer

        if "answer" in example:
            answer = example["answer"]
            correct_count += eval_output(answer, output_ans)
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
        
        with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb')  as f:
            pickle.dump(output, f)


if __name__ == '__main__':
    fire.Fire(main)
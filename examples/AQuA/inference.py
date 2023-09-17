import pickle
from typing import Type, Callable, Optional

import numpy as np
from datasets import load_dataset
from reasoners.algorithm.mcts import MCTSResult
from sklearn import base
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode
from reasoners.visualization import TreeLog

from world_model import MATHWorldModel, MATHState, MATHAction
from search_config import MATHConfig
import utils
import re
from datasets import Dataset

def data_reader(dataset,dataset_path, split=None, sample_size=100):
    questions = []
    answers = []
    options = []
    filename = os.path.join(dataset_path, f'{dataset}.json')
    lines = Dataset.from_json(filename)
    if split is not None:
        start, end = split
        lines = lines[start:end]
    for i in range(len(lines)):
        data = lines[i]
        if isinstance(data, dict):
            options_list = data['options']
            question_with_options = data['question'] + " Options: " + (" ".join(data['options'])).replace('A)','A) ').replace('B)','B) ').replace('C)','C) ').replace('D)','D) ').replace('E)','E) ') + "."
            questions.append(question_with_options)
            answers.append(data['correct'])
            options.append(options_list)
        else:
            raise ValueError("Unexpected data format")
    return Dataset.from_dict({"question": questions, "answer": answers, "options":options})

def node_visualizer(x: MCTSNode[MATHState, MATHAction]):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


def rap_AQuA(base_model: LanguageModel,
              interactive_prompt: dict,
              useful_prompt: dict,
              search_algo: Type[SearchAlgorithm] = MCTS,
              resume: int = 0,
              n_action: int = 4,
              n_confidence: int = 8,
              depth_limit: int = 5,
              force_terminating_on_depth_limit: bool = True,
              batch_size: int = 1,
              temperature: float = 0.8,
              early_stop_base: int = 2,
              early_stop_threshold: float = 0.5,
              reward_alpha: float = 0.5,
              reward_confidence_default: float = 0.8,
              cum_reward: Callable[[list[float]], float] = np.mean,
              calc_q: Callable[[list[float]], float] = max,
              log_dir: Optional[str] = None,
              datasetname: str = 'AQuA_clean',
              dataset_path: str = r"/data/haotian/RAP_tune/llm-reasoners/dataset/AQuA",
              disable_log: bool = False,
              disable_tqdm: bool = False,
              output_trace_in_each_iter: bool = True,
              **search_algo_params):
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/aqua_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm, 'output_trace_in_each_iter': output_trace_in_each_iter}
    world_model = MATHWorldModel(base_model=base_model, prompt=interactive_prompt,
                                  n_confidence=n_confidence, batch_size=batch_size, temperature=temperature,
                                  early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = MATHConfig(base_model=base_model, prompt=interactive_prompt, useful_prompt=useful_prompt,
                         n_actions=n_action, batch_size=batch_size, temperature=temperature,
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)
    dataset = data_reader(datasetname, dataset_path)
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
                                     desc='AQuA_clean', disable=disable_tqdm)):
        try:
            algo_output = reasoner(example["question"])
        except AssertionError as e:
            from reasoners.algorithm import MCTSResult
            algo_output = MCTSResult(terminal_state=None, cum_reward=None, trace=None, trace_of_nodes=None, tree_state=None)
        if algo_output.terminal_state is None:
            output = None
        else:
            output = utils.retrieve_answer(algo_output.terminal_state[-1].sub_answer)
        # print(output)
        answer = utils.retrieve_answer_from_dataset(example["answer"])
        correct = utils.judge_answer(output, answer)
        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{resume + i + 1}: {correct=}, {output=}, {answer=} ; {accuracy=:.3f} ({correct_count}/{i + 1})'

        tqdm.write(log_str)
        if not disable_log:
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
            with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.pkl'), 'wb') as f:
                pickle.dump(algo_output, f)
            if isinstance(search_algo, MCTS):
                with open(os.path.join(log_dir, 'algo_output', f'{resume + i + 1}.json'), 'w') as f:
                    # noinspection PyTypeChecker
                    print(TreeLog.from_mcts_results(algo_output, node_data_factory=node_visualizer), file=f)


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

    def main_exllama(
                model_dir = '/data/haotian/RAP_tune/Llama-2-70B-GPTQ',
                lora_dir = None,
                batch_size = 1,
                mem_map = [18,22],
                interactive_prompt = "/data/haotian/RAP_tune/llm-reasoners/examples/AQuA/prompts/interactive_examples.json",
                useful_prompt = "/data/haotian/RAP_tune/llm-reasoners/examples/AQuA/prompts/useful_examples.json",
                disable_log = False,
                disable_tqdm = False,
                **kwargs):
        from reasoners.lm import ExLlamaModel
        device = torch.device("cuda:0")
        base_model = ExLlamaModel(model_dir, lora_dir, device=device, max_batch_size=batch_size, max_new_tokens=512, max_seq_length=2048, mem_map=mem_map)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs
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
            base_model = LLaMAModel(llama_ckpt, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            base_model = LlamaCppModel(llama_cpp_path)
        elif base_lm == 'llama2':
            base_model = LlamaModel(llama_2_ckpt, llama_size, max_batch_size=batch_size)
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


    fire.Fire(main_exllama)
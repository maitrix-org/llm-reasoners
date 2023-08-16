import pickle
from typing import Type, Callable, Optional

import numpy as np
from reasoners.visualization import TreeLog
from tqdm import tqdm
from datetime import datetime

from reasoners import LanguageModel, Reasoner, SearchAlgorithm
from reasoners.algorithm import MCTS, MCTSNode

from world_model import strategyQAWorldModel, strategyQAState, strategyQAAction
from search_config import strategyQAConfig
import utils
from dataset import get_prompt_examples, get_examples, extract_golden_answer 


def node_visualizer(x: MCTSNode[strategyQAState, strategyQAAction]):
    if not x.state:
        return {}
    return {"question": x.state[-1].sub_question, "answer": x.state[-1].sub_answer}


def rap_strategyQA(base_model: LanguageModel,
              interactive_prompt: dict,
              useful_prompt: dict,
              decompose_prompt: str,
              search_algo: Type[SearchAlgorithm] = MCTS,
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
              output_trace_in_each_iter: bool = False,
              **search_algo_params):
    if not disable_log:
        if log_dir is None:
            log_dir = f'logs/strategyQA_{search_algo.__name__}/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir, exist_ok=resume > 0)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)

    search_algo_params |= {'cum_reward': cum_reward, 'calc_q': calc_q, 'disable_tqdm': disable_tqdm, \
                           'output_trace_in_each_iter': output_trace_in_each_iter}
    
    world_model = strategyQAWorldModel(base_model=base_model, prompt=interactive_prompt,
                                n_confidence=n_confidence, batch_size=batch_size, temperature=temperature,
                                early_stop_base=early_stop_base, early_stop_threshold=early_stop_threshold)
    config = strategyQAConfig(base_model=base_model, prompt=interactive_prompt, useful_prompt=useful_prompt, decompose_prompt=decompose_prompt,
                         n_actions=n_action, batch_size=batch_size, temperature=temperature,
                         reward_alpha=reward_alpha, reward_confidence_default=reward_confidence_default,
                         force_terminating_on_depth_limit=force_terminating_on_depth_limit, depth_limit=depth_limit)
    search_algo = search_algo(**search_algo_params)
    reasoner = Reasoner(world_model=world_model, search_config=config, search_algo=search_algo)

    dataset = examples = get_examples(folder='examples/rap_strategyQA/data/', split='test')
    correct_count = 0
    for i, example in enumerate(tqdm(dataset, total=resume + len(dataset), initial=resume,
                                     desc='strategyQA', disable=disable_tqdm)):
        print(example["question"])
        algo_output = reasoner(example["question"])
        if algo_output.terminal_state is None:
            output = None
        else:
            output = utils.extract_answer(algo_output.terminal_state[-1].sub_answer)
        answer = extract_golden_answer(example)
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
    from reasoners.lm import LLaMAModel, LlamaCppModel, LlamaModel
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


    def main(base_lm: str = 'llama', #llama means llama_v1 and llama2 means llama_v2
             llama_ckpt: str = llama_ckpts,
             llama_2_ckpt: str = llama_2_ckpts,
             llama_size: str = '13B',
             llama_cpp_path: str = None,
             batch_size: int = 2,
             interactive_prompt: str = 'examples/rap_strategyQA/prompts/interactive_examples-1.json',
             useful_prompt: str = 'examples/rap_strategyQA/prompts/useful_examples-1.json',
             decompose_prompt: str = 'examples/rap_strategyQA/problem_decompose_examples-1.0.1.txt',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        # set base_lm = 'llama' and llama_ckpt = '13B/30B/65B' to use llama with torchscale
        # else set base_lm = 'llama.cpp' and llama_cpp_path = the checkpoint to use llama.cpp

        with open(interactive_prompt) as f:
            interactive_prompt = json.load(f)
        with open(useful_prompt) as f:
            useful_prompt = json.load(f)
        decompose_prompts = get_prompt_examples(path=decompose_prompt)
        if base_lm == 'llama':
            base_model = LLaMAModel(llama_ckpt, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            base_model = LlamaCppModel(llama_cpp_path)
        elif base_lm == 'llama2':
            base_model = LlamaModel(llama_2_ckpt, llama_size, max_batch_size=batch_size)
        else:
            assert False, f'cannot resolve {base_lm=}'
        rap_strategyQA(base_model=base_model,
                  interactive_prompt=interactive_prompt,
                  useful_prompt=useful_prompt,
                  decompose_prompt=decompose_prompt,
                  batch_size=batch_size,
                  disable_log=disable_log or local_rank != 0,
                  disable_tqdm=disable_tqdm or local_rank != 0,
                  **kwargs)


    fire.Fire(main)
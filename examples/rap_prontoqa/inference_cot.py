from reasoners.lm import ExLlamaModel
import json
import fire
import itertools
import os
from typing import Sequence, Any
import json
from tqdm import tqdm
import pickle

from dataset import ProntoQADataset, ProntoQAProblem, ProntoQAExample
from reasoners import LanguageModel, SearchAlgorithm, Reasoner

import prompts.finish
import prompts.valid
import prompts.next_step

from reasoners.lm import llama_cpp_model
from reasoners.visualization import visualize
from search_config import ProntoQAConfig
from world_model import ProntoQAWorldModel, ProntoQAState, ProntoQAAction
from reasoners.algorithm import MCTS
from reasoners.benchmark import ProntoQAEvaluatorFinal


class CoTReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0.8, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
    def __call__(self, example, prompt=None):
        # *base_facts, init_state = example.test_example.question.split(". ")
        # input_prompt += prompts.next_step.EXAMPLES
        input_prompt = prompt
        print(f"input_prompt: '{input_prompt}'\n")
        input_prompt += "Q: " + example.test_example.question + " " + example.test_example.query + "\nA:"
        print(f"input_prompt: '{input_prompt}'\n")
        output = self.base_model.generate([input_prompt], eos_token_id="\n", hide_input=True, temperature=self.temperature, do_sample=True).text[0]
        # output = "Next 4.1:" + output
        steps = [s.split("So")[1].strip()+'.' for s in output.split('.') if "So" in s]
        # deduplicate
        return "\n".join(steps)

def main(temperature=0.0, log_name="name"):

    import torch, os
    import numpy as np
    from reasoners.lm import ExLlamaModel 
    language_model = ExLlamaModel(os.environ['LLAMA2_CKPTS'],
                                None, 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=None,
                                log_output=True) #please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

    # dataset = ProntoQADataset.from_file(
    #     'examples/rap_prontoqa/data/345hop_random_true.json'
    # )

    with open('examples/rap_prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner =  CoTReasoner(base_model=language_model)

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="cot",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/rap_prontoqa/data/345hop_random_true.json'
        ),
        output_extractor=lambda x: x,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4)
    print(f"accuracy: {accuracy}")

if __name__ == '__main__':
    fire.Fire(main)

# CUDA_VISIBLE_DEVICES=1 python examples/rap_prontoqa/inference_cot.py --temperature 0.0
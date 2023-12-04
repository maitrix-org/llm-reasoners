from reasoners.lm import ExLlamaModel
import json
from reasoners.benchmark import GSM8KEvaluator
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

def format_examples(sampled_data):
    formatted_examples = ""
    for i, entry in enumerate(sampled_data, 1):
        facts = f"Facts {i}: {entry['Facts']}\n"
        query = f"Query {i}: {entry['Query']}\n"
        claims_and_next = ""
        for j, (claim, next_step) in enumerate(zip(entry['claims'], entry['next_steps']), 1):
            claims_and_next += f"Claim {i}.{j}: {claim}\nNext {i}.{j}: {next_step}\n"
        formatted_examples += facts + query + claims_and_next + "\n"

    return formatted_examples

class CoTReasoner():
    def __init__(self, base_model, n_sc=1, temperature=0.8, bs=1):
        assert n_sc == 1 or temperature > 0, \
            "Temperature = 0 indicates greedy decoding. There is no point running multiple chains (n_sc > 1)"
        self.base_model = base_model
        self.temperature = temperature
        self.n_sc = n_sc
        self.bs = bs
    def __call__(self, example, prompt=None):
        *base_facts, init_state = example.test_example.question.split(". ")
        input_prompt = ""
        # input_prompt += prompts.next_step.EXAMPLES
        input_prompt += format_examples(prompt)
        input_prompt += prompts.next_step.FACTS_FORMAT.format(len(prompt),". ".join(base_facts))
        input_prompt += prompts.next_step.QUERY_FORMAT.format(len(prompt), example.test_example.query)
        input_prompt += prompts.next_step.CLAIM_FORMAT.format(len(prompt), init_state)
        input_prompt += prompts.next_step.NEXT_STEP_PREFIX.format(len(prompt))

        print(f"input_prompt: '{input_prompt}'\n")
        output = self.base_model.generate([input_prompt], eos_token_id=[4231], hide_input=True, temperature=self.temperature, do_sample=True).text[0]
        # output = "Next 4.1:" + output
        steps = [s.split(":")[-1].split("\n")[0].strip() for s in output.split("5.")[1:-1:2]]
        # deduplicate

        return "\n".join(steps)

def main():

    import torch, os
    import numpy as np
    from reasoners.lm import ExLlamaModel 
    language_model = ExLlamaModel(os.environ['LLAMA2_CKPTS'],
                                None, 
                                max_batch_size=1, 
                                max_new_tokens=200, 
                                max_seq_length=2048, 
                                mem_map=None,
                                log_output=True)#please set mem_map if you need model parallelism, e.g. mem_map = [16,22] with 2 GPUs

    dataset = ProntoQADataset.from_file(
        'examples/rap_prontoqa/data/345hop_random_true.json'
    )

    with open('examples/rap_prontoqa/data/example_next_steps.json') as f:
        init_prompt = json.load(f)
    
    reasoner =  CoTReasoner(base_model=language_model)

    evaluator = ProntoQAEvaluatorFinal(
        init_prompt=init_prompt['next_steps'],
        sample_prompt_type="rap",
        disable_log=False,
        disable_tqdm=False, dataset = ProntoQADataset.from_file(
            'examples/rap_prontoqa/data/345hop_random_true.json'
        ),
        output_extractor=lambda x: x,
        answer_extractor=lambda x: "\n".join(x.test_example.chain_of_thought[2::2])
    )

    accuracy = evaluator.evaluate(reasoner, num_shot=4, log_dir="pronto_logs/")
    print(f"accuracy: {accuracy}")

if __name__ == '__main__':
    fire.Fire(main)

    """
CUDA_VISIBLE_DEVICES=2 python examples/cot_gsm8k/inference.py \
--exllama_model_dir $LLAMA2_CKPTS \
--exllama_lora_dir None \
--exllama_mem_map None \
--temperature 0.0
    """

    """
CUDA_VISIBLE_DEVICES=1 python examples/cot_gsm8k/inference.py --exllama_model_dir $LLAMA2_CKPTS --exllama_lora_dir None --exllama_mem_map None --temperature 0.8 --n_sc 10 --log_dir logs/4-shot-cot-llama2-70b-sc-10-temp-0.8-speed --batch_size 4 | tee cot_sc.log
    """


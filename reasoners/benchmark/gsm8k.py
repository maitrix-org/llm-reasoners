import datasets
import json
from tqdm import tqdm
import torch
import os, pickle
from datetime import datetime
import sys
import random
import copy
from reasoners import Evaluator

class GSM8KEvaluator(Evaluator):
    def __init__(self,
                 output_extractor,
                 answer_extractor,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="l2m") -> None:

        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x["question"]
        self.full_dataset = datasets.load_dataset('gsm8k', 'main', split='test')
        self._dataset_name = 'gsm8k'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=4):

        sample_prompt_type = self.sample_prompt_type
        if sample_prompt_type == "l2m":
            prompt = {}
            if shuffle_prompt:
                decomp_examples = random.sample(self.init_prompt["decomposition_pool"], num_shot)
                solv_examples = random.sample(self.init_prompt["solving_pool"], num_shot)
            else:
                decomp_examples = self.init_prompt["decomposition_pool"][:num_shot]
                solv_examples = self.init_prompt["solving_pool"][:num_shot]
            prompt["decomposition"] = "".join(decomp_examples) + self.init_prompt["composition_prefix"]
            prompt["overall"] = "".join(decomp_examples) + self.init_prompt["overall_prefix"]
            prompt["solving"] = "".join(solv_examples) + self.init_prompt["solving_prefix"]

        elif sample_prompt_type == "cot":
            prompt = {}
            if shuffle_prompt:
                examples = random.sample(self.init_prompt["cot_pool"], num_shot)
            else:
                examples = self.init_prompt["cot_pool"][:num_shot]
            prompt["cot"] = "".join(examples) + self.init_prompt["prefix"]

        elif sample_prompt_type == "rap":

            ret = copy.deepcopy(self.init_prompt)
            ret['interactive_examples'], ret['useful_examples'] = zip(*random.sample(list(zip(ret['interactive_examples'],
                                                                                            ret['useful_examples'])),
                                                                                    k=num_shot))
            return ret
            
        elif sample_prompt_type == "grace":
            return None
            
        else:
            raise NotImplementedError
        return prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        try:
            output = int(output)
            answer = int(answer)
            return output == answer
        except ValueError:
            pass
        try:
            output = float(output)
            answer = float(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer

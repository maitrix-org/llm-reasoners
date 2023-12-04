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
import itertools

class ProntoQAEvaluatorFinal(Evaluator):
    def __init__(self, 
                 output_extractor= lambda x: x.terminal_state.body if x.terminal_state is not None else "",
                 answer_extractor= lambda x: x.test_example.answer,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="l2m", dataset=None) -> None:

        dataset_list = list(dataset)
        dataset_list= dataset_list[:60]
        self.queries = [obj.test_example.query.split(':', 1)[1].strip() for obj in dataset_list]
        self.dataset = iter(dataset_list)
        self.answers = [obj.test_example.answer for obj in dataset_list]
        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x
        self.full_dataset = list(dataset_list)
        self._dataset_name = 'prontoqa'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=6,
                      sample_prompt_type="rap"):

        if sample_prompt_type == "rap":

            ret = random.sample(list(self.init_prompt), k=num_shot)
            return ret
            # return []

        else:
            raise NotImplementedError


    def eval_output(self, answer, output):
        if output is None:
            return False
        try:
            output = str(output)
            answer = str(answer)
            return output == answer
        except ValueError:
            pass
        try:
            output = str(output)
            answer = str(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer
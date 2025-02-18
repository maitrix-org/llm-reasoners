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

def load_ProsQA(json_file):
    with open(json_file, 'r') as f:
        data_list = json.load(f)  
    data = []
    for entry in data_list:
        cur_data = {
            "question": entry["question"],
            "answer": entry["answer"],
            "steps": entry["steps"],
            "idx_to_symbol": entry["idx_to_symbol"],
            "edges": entry["edges"],
            "root": entry["root"],
            "target": entry["target"],
            "neg_target": entry["neg_target"]
        }
        data.append(cur_data)  
    return data

class ProsQAEvaluator(Evaluator):
    def __init__(self,
                 output_extractor,
                 answer_extractor,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type=None) -> None:

        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor #lambda x: x["answer"]
        self.input_processor = lambda x: x["question"] + "\n\nPlease output your conclusion like 'Answer: XXX is a YYY.'"
        self.full_dataset = load_ProsQA('examples/LongCoT_Search/ProsQA/data/prosqa_test.json')
        self._dataset_name = 'prosQA'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm

    def sample_prompt(self,
                  sample_prompt_type=None, 
                  shuffle_prompt=True,
                  num_shot=4):
        return self.init_prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        try:
            output = str(output)
            answer = str(answer)
            return output == answer
        except ValueError:
            pass
        return output == answer
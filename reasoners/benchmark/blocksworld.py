import datasets
import json
from tqdm import tqdm
import torch
import os, pickle
from datetime import datetime
import sys
import random

class BWEvaluator():
    def __init__(self, 
                 output_extractor,
                 answer_extractor,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False,
                 sample_prompt_type="rap") -> None:

        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.input_processor = lambda x: x["question"]
        self.full_dataset = datasets.load_dataset('gsm8k', 'main', split='test')
        self._dataset_name = 'gsm8k'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm
        self.sample_prompt_type = sample_prompt_type
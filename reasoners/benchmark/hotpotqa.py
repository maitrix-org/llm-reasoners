import json
from tqdm import tqdm
from datetime import datetime
import random
from reasoners import Evaluator

class Hotpotqaevaluator(Evaluator):
    def __init__(self,
                 output_extractor,
                 answer_extractor,
                 data_path,
                 toolset,
                 init_prompt=None,
                 disable_log=False,
                 disable_tqdm=False) -> None:

        self.init_prompt = init_prompt
        self.output_extractor = output_extractor
        self.answer_extractor = answer_extractor
        self.toolset = toolset
        self.input_processor = lambda x: x["question"]
        with open(data_path, 'r', encoding='utf-8') as json_file:
            self.full_dataset = json.load(json_file)
        self._dataset_name = 'hotpotqa'
        self.disable_log = disable_log
        self.disable_tqdm = disable_tqdm

    def sample_prompt(self,
                      shuffle_prompt=True,
                      num_shot=6):
        prompt = {}
        if shuffle_prompt:
            examples = random.sample(self.init_prompt["react_pool"], num_shot)
        else:
            examples = self.init_prompt["react_pool"][:num_shot]
        prompt['ReAct'] = self.init_prompt["prefix"] + self.toolset[0].description + "".join(examples)
        prompt['prefix'] = self.init_prompt["prefix"]
        prompt['examples'] = examples
        return prompt

    def eval_output(self, answer, output):
        if output is None:
            return False
        return output.lower() == answer.lower()

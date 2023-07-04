import os, sys
import random
import json
import torch
import numpy as np
import re
from collections import defaultdict
from datasets import load_dataset
import fire

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rap import WorldModel, SearchConfig, RAPAgent
from rap.algorithm import BeamSearch
from rap.lm import LLaMAModel

class GSMWorldModel(WorldModel):
    """GSM World Model
    
    State: [[sub_question_1, sub_answer_1, confidence_1], [sub_question_2, sub_answer_2, confidence_2], ...]
    """
    def __init__(self, base_model, prompt, max_n_confidence=8, batch_size=2) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.max_n_confidence = max_n_confidence
    
    def init_state(self) -> list:
        return []

    def step(self, state: list, action: str) -> list:
        state = state.copy()
        input = self.prompt["input"] + self.prompt["question_prefix"] + self.example + "\n" + "\n".join([f'{self.prompt["subquestion_prefix"].format(idx + 1)} {q}\n{self.prompt["answer_prefix"]} {a}' for idx, (q, a, c) in enumerate(state)]) + "\n" + self.prompt["subquestion_prefix"].format(len(state) + 1) + " " + action + "\n" + self.prompt["answer_prefix"].format(len(state) + 1)
        answer_list = []
        answer_dict = defaultdict(list)
        for idx in range(0, self.max_n_confidence, self.batch_size):
            n_samples = min(self.max_n_confidence - idx, self.batch_size)
            outputs = self.base_model([input] * n_samples, end_token="\n", hide_input=True)["text"]
            for output in outputs:
                result = output.strip()
                match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
                if match is None:
                    continue
                sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')
                answer_dict[sub_answer].append(output)
                answer_list.append(sub_answer)
            if len(answer_dict) == 0:
                continue
            sorted_answer_dict = sorted(answer_dict.items(), key=lambda p: len(p[1]), reverse=True)
            max_len = len(sorted_answer_dict[0][1])
            if max_len < 2:
                continue
            if len(sorted_answer_dict) < 2:
                break
            second_max_len = len(sorted_answer_dict[1][1])
            if max_len >= len(answer_dict) / 2 and max_len > second_max_len:
                break

        if len(answer_dict) == 0:
            confidence, answer = 0, ""
        else:
            answer = sorted_answer_dict[0][1][0]  # [0]: maximum; [1]: list of outputs; [0]: first output in the list
            confidence = max_len / len(answer_list)
        
        state.append((action, answer, confidence))
        return state

    def is_terminal(self, state: list) -> bool:
        if len(state) > 0 and "Now we can answer" in state[-1][0]:
            return True
        else:
            return False

class GSMConfig(SearchConfig):

    def __init__(self, base_model: LLaMAModel, prompt, n_actions=4, batch_size=2) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_actions = n_actions

    def get_actions(self, state: list) -> list[str]:
        input = self.prompt["input"] + self.prompt["question_prefix"] + self.example + "\n" + "\n".join([f'{self.prompt["subquestion_prefix"].format(idx + 1)} {q}\n{self.prompt["answer_prefix"]} {a}' for idx, (q, a, c) in enumerate(state)]) + "\n" + self.prompt["subquestion_prefix"].format(len(state) + 1)
        outputs = []
        for idx in range(0, self.n_actions, self.batch_size):
            n_samples = min(self.n_actions - idx, self.batch_size)
            outputs += self.base_model([input] * n_samples, end_token="\n", hide_input=True)["text"]

        return_actions = [output.strip() for output in outputs]
        return return_actions

    def prior_policy(self, state: list, action: str) -> float:
        # not used in beam search
        return 1.0

    def reward(self, state: list, action: str, **kwargs) -> float:
        return kwargs["next_state"][-1][2]
    
    def fast_reward(self, state: list, action: str, **kwargs) -> float:
        return kwargs["next_state"][-1][2]


def main(llama_path, llama_size, prompt_path):

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    prompt = json.load(open(prompt_path))

    base_model = LLaMAModel(llama_path, llama_size, max_batch_size=2)
    world_model = GSMWorldModel(base_model, prompt, batch_size=2)
    config = GSMConfig(base_model, prompt)
    algorithm = BeamSearch(beam_size=4, max_depth=6)
    agent = RAPAgent(world_model, config, algorithm)
    
    gsm8k_dataset = load_dataset("gsm8k", "main")
    for question in gsm8k_dataset["test"]:
        print(agent(question["question"], output_trace=True))

if __name__ == "__main__":
    fire.Fire(main)
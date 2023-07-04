import os, sys
import random
import json
import torch
import numpy as np
import re
from collections import defaultdict
import fire

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from rap import WorldModel, SearchConfig, RAPAgent
from rap.algorithm import BeamSearch
from rap.lm import LLaMAModel

from examples.game24.prompts.game24_data import *

class game24WorldModel(WorldModel):
    """game24 World Model
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    State: equation (left: left numbers) conf
    """
    def __init__(self, base_model, prompt, max_n_confidence=8, batch_size=2) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.max_n_confidence = max_n_confidence
    
    def init_state(self) -> list:
        ## input, output
        return [(self.example, '')]

    def step(self, state: list, action: str) -> list:
        state = state.copy()
        ## new action is the new state
        state = (state, action)
        return state

    def is_terminal(self, state: list) -> bool: 
        ## if there is no left number or LLM is sure it can reach 24
        x, y = state[0], state[1]
        last_line = y.strip().split('\n')[-1]
        current_numbers = get_current_numbers(y if y else x)
        if 'left: ' not in last_line or current_numbers == 24:
            return True
        else:
            return False
        
    

class game24Config(SearchConfig):

    def __init__(self, base_model: LLaMAModel, prompt, n_actions=4, batch_size=2) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_eval = 3

    def get_actions(self, state: list) -> list[str]:
        #### propose possible actions (bfs)
        #### state, output, prompts
        input = propose_prompt_wrap(state[0], state[1], self.prompt)
        outputs = []
        for idx in range(0, self.n_actions, self.batch_size):
            n_samples = min(self.n_actions - idx, self.batch_size)
            outputs += self.base_model([input] * n_samples, end_token="\n", hide_input=True)["text"]

        return_actions = [state[0] + output + '\n' for output in outputs]
        return return_actions

    def prior_policy(self, state: list, action: str) -> float:
        # not used in beam search
        return 1.0

    def reward(self, state: list, action: str, **kwargs) -> float:  
        ## get values (state eval) for each action
        ## impossible, maybe, sure
        value_prompt = value_prompt_wrap(state, action, self.prompt)
        value_outputs = self.base_model([value_prompt] * self.n_eval, end_token="\n", hide_input=True)["text"]
        value = value_outputs_unwrap(state, action, value_outputs)
        return value


def main(llama_path, llama_size, prompt_path):

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    prompt = json.load(open(prompt_path))

    base_model = LLaMAModel(llama_path, llama_size, max_batch_size=2)
    world_model = game24WorldModel(base_model, prompt, batch_size=2)
    config = game24Config(base_model, prompt)
    ## keep the best 5 candidates, need at most 4 steps to solve
    algorithm = BeamSearch(beam_size=5, max_depth=5)
    agent = RAPAgent(world_model, config, algorithm)
    
    game24_dataset = read_data(file='./game24/24.csv')
    for question in game24_dataset:
        print(agent(question, output_trace=True))

if __name__ == "__main__":
    fire.Fire(main)
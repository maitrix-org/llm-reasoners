import io

import numpy as np
import re

from reasoners import SearchConfig, LanguageModel
from world_model import crosswordsState, crosswordsAction
from utils import *
from prompts.crosswords import * 


class crosswordsConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 n_eval=8,
                 batch_size=2,
                 depth=10,) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.n_eval = n_eval
        self.batch_size = batch_size
        self.depth = depth
        self.confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  # TODO: ad hoc
        self.cache = {}

    def prompt_wrap(self, obs):
        return propose_prompt.format(input=obs)

    def parse_line(self, input_str):
        # regular expression pattern to match the input string format
        pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

        # use regex to extract the parts of the input string
        match = re.match(pattern, input_str)

        if match:
            # extract the matched groups
            parts = [match.group(1), match.group(2), match.group(3)]
            return parts
        else:
            return None

    def parse_response(self, response):
        # split the response into lines
        lines = response.split('\n')

        # parse each line
        parsed_lines = [self.parse_line(line) for line in lines]

        # filter out the lines that didn't match the format
        parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), self.confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]

        return parsed_lines if len(parsed_lines) >= 1 else None

    def get_actions(self, state: crosswordsState) -> list[crosswordsAction]:
        env, actions, trace = state
        obs = env.render()
        if obs in self.cache: 
            print('cache hit')
            return self.cache[obs]
        print('call gpt')
        # print(f"current obs: {obs}")
        # print(f'prompt: {self.prompt_wrap(obs)}')
        responses = self.base_model.generate(self.prompt_wrap(obs), num_return_sequences=self.n_eval, stop=None).text
        print(responses)

        candidates_to_scores = {}
        for response in responses:
            parsed_response = self.parse_response(response)
            if parsed_response:
                for candidate, score in parsed_response:
                    candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
        
        return_actions = []
        for candidate in candidates_to_scores:
            return_actions.append((candidate, candidates_to_scores[candidate]))
        self.cache[obs] = return_actions

        # print(f'propose actions: {return_actions}')
        return return_actions

    def fast_reward(self, state: crosswordsState, action: crosswordsAction) -> tuple[float, dict]:
        ## don't need fast_reward for beam search
        return 0

    def reward(self, state: crosswordsState, action: crosswordsAction, next_state: crosswordsState) -> float:
        env, actions, infos = state
        
        return 0
    

    def search_condition(self, state: crosswordsState) -> bool:
        env, actions, info = state
        #print(f'{env.steps} {self.depth} {env.steps < self.depth} {not any(_ == 2 for _ in env.status)}')
        # print(env.get_ans(env.board))
        # print(env.status)
        if env.steps < self.depth and not any(_ == 2 for _ in env.status):
            return True
        return False
    
    def state_condition(self, state: crosswordsState) -> bool:
        env, actions, info = state
        count = info['count']
        print(f'curent count: {count}')
        return (count['impossible'] < 1)
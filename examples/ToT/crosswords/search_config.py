import io

import numpy as np
import re

from reasoners import SearchConfig, LanguageModel
from world_model import CrosswordsState, CrosswordsAction
from utils import *
from prompts.crosswords import * 
from reasoners.lm import OpenAIModel, Llama2Model, Llama3Model

class CrosswordsConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 n_eval=8,
                 depth=10,
                 temperature=0) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.n_eval = n_eval
        self.depth = depth
        self.temperature = temperature
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

    def get_actions(self, state: CrosswordsState) -> list[CrosswordsAction]:
        env, actions, trace = state
        obs = env.render(status=True)
        if obs in self.cache: 
            print('cache hit')
            return self.cache[obs]
        print('call gpt')
        # print(f"current obs: {obs}")
        # print(f'prompt: {self.prompt_wrap(obs)}')

        if isinstance(self.base_model, OpenAIModel):
            eos_token_id = []
        elif isinstance(self.base_model, Llama2Model):
            eos_token_id = ["\n"]
        elif isinstance(self.base_model, Llama3Model):
            eos_token_id = ["\n\n", ".\n", ".\n\n","\n"]
        responses = self.base_model.generate([self.prompt_wrap(obs)], #+"Make sure using the format 'h1. apple (medium)' in answer."
                                            num_return_sequences=self.n_eval,
                                            stop=None,
                                            hide_input=True,
                                            do_sample=True,
                                            temperature=self.temperature,
                                            eos_token_id=eos_token_id).text
        #print(self.prompt_wrap(obs))
        
        #responses = self.base_model.generate(self.prompt_wrap(obs), num_return_sequences=self.n_eval, stop=None).text
        #print(responses)

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

    def fast_reward(self, state: CrosswordsState, action: CrosswordsAction) -> tuple[float, dict]:
        ## don't need fast_reward for beam search
        return 0

    def reward(self, state: CrosswordsState, action: CrosswordsAction, next_state: CrosswordsState) -> float:
        env, actions, infos = state
        
        return 0
    

    def search_condition(self, state: CrosswordsState) -> bool:
        env, actions, info = state
        if env.steps < self.depth and not any(_ == 2 for _ in env.status):
            return True
        return False
    
    def state_condition(self, state: CrosswordsState) -> bool:
        env, actions, info = state
        if len(info) == 0:
            return True
        count = info['count']
        # print(f'current count: {count}')
        return (count['impossible'] < 1)
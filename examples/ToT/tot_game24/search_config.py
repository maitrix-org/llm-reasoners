import copy
import re
from typing import Literal

import numpy as np
import scipy
import torch

from reasoners import SearchConfig, LanguageModel
from world_model import Game24State, Game24Action

from prompts.game24 import output_prompt, propose_prompt, value_prompt, value_last_step_prompt, value_map


class Game24Config(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=2,
                 depth_limit=4,
                 temperature=0.7,
                 n_eval=5,
                 calc_reward: Literal['sampling', 'logits'] = 'sampling') -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_eval = n_eval
        self.value_cache = {}
        self.depth_limit = depth_limit
        self.temperature = temperature
        self.calc_reward = calc_reward

    @staticmethod
    def output_prompt_wrap(state: Game24State) -> str:
        return output_prompt.format(input=state.input, history='\n'.join(state.history))

    @staticmethod
    def propose_prompt_wrap(state: Game24State) -> str:
        return propose_prompt.format(input=state.current)

    @staticmethod
    def value_prompt_wrap(state: Game24State) -> str:
        return value_prompt.format(input=state.current)

    @staticmethod
    def value_last_step_prompt_wrap(state: Game24State) -> str:
        return value_last_step_prompt.format(input=state.input, answer=state.output)

    @staticmethod
    def retrieve_value(output: list[str]) -> float:
        output_names = [x.split('\n')[-1] for x in output]
        value = sum(v * output_names.count(k) for k, v in value_map.items())
        return value

    def get_actions(self, state: Game24State) -> list[Game24Action]:
        if state.current == '':
            return []
        # print(f'Generating actions for {state}')
        if state.current == '24':
            prompt = self.output_prompt_wrap(state)
            output = \
            self.base_model.generate([prompt], num_return_sequences=1, do_sample=False, eos_token_id='\n').text[0]
            output = 'Answer: ' + output.strip()
            return [output]
        elif ' ' not in state.current:
            return []
        else:
            prompt = self.propose_prompt_wrap(state)
            output = \
            self.base_model.generate([prompt], num_return_sequences=1, do_sample=False, eos_token_id='Input').text[0]
            output = output.strip()
            if '\n\n' in output:
                output = output.split('\n\n')[0]
            output = output.split('\n')
            actions = [x for x in output if 'left' in x]
            # set does not guarantee order, but dict does guarantee
            # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
            actions = list(dict.fromkeys(actions))
            return actions

    def _reward(self, state: Game24State, action: Game24Action) -> float:
        if state.current == '':
            return 0.
        next_state = copy.deepcopy(state)
        if 'Answer' in action:
            match = re.match(r'Answer: (.*)', action)
            next_state.output = match[1] if match is not None else ''
        else:
            match = re.match(r'.*\(left: (.*)\)', action)
            next_state.current = match[1] if match is not None else ''
            next_state.history.append(action)

        if len(next_state.history) >= self.depth_limit:
            return 0.
        if next_state.output is None:
            prompt = self.value_prompt_wrap(next_state)
        else:
            prompt = self.value_last_step_prompt_wrap(next_state)
        if prompt in self.value_cache:
            return self.value_cache[prompt]

        if self.calc_reward == 'sampling':
            value_outputs = []
            for idx in range(0, self.n_eval, self.batch_size):
                n_samples = min(self.n_eval - idx, self.batch_size)
                output = self.base_model.generate([prompt], do_sample=True, temperature=self.temperature,
                                                  num_return_sequences=n_samples).text
                value_outputs += [o.strip().split('\n\n')[0] for o in output]
            # print(value_outputs)
            value = self.retrieve_value(value_outputs)
        elif self.calc_reward == 'logits':
            value_keys = list(value_map.keys())
            logits = self.base_model.get_next_token_logits([prompt], value_keys)[0]
            logits = scipy.special.softmax(logits)
            value = np.sum(logits * np.array(list(value_map.values())))
        else:
            raise NotImplementedError

        self.value_cache[prompt] = value
        # print(f'Reward of {state}, {action=} is {value:.5f}')
        return value

    def fast_reward(self, state: Game24State, action: Game24Action) -> tuple[float, dict]:
        reward = self._reward(state, action)
        return reward, {'reward': reward}

    # We calculate the full reward in fast_reward in Game24SearchConfig, direct return it
    def reward(self, state: Game24State, action: Game24Action, **kwargs) -> tuple[float, dict]:
        return self.fast_reward(state, action)

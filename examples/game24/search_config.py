import io

import numpy as np
import re

from rap import SearchConfig, LanguageModel
from world_model import game24State, game24Action
import utils


class game24Config(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=2,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 n_eval=5,
                 force_terminating_on_depth_limit=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_eval = n_eval
        self.depth_limit = depth_limit
        self.reward_confidence_default = reward_confidence_default

    def get_actions(self, state: game24State) -> list[game24Action]:
        x, y = state[0], state[1]
        with io.StringIO() as f:
            f.write(utils.propose_prompt_wrap(x, y, self.prompt) + "\n")
            f.write(utils.value_prompt_wrap(x, y, self.prompt) + "\n")
            model_input = utils.propose_prompt_wrap(state[0], state[1], self.prompt)
        # print(f'propose input: {x}, {y}, {model_input}')
        outputs = []
        # for idx in range(0, self.n_actions, self.batch_size):
        #     n_samples = min(self.n_actions - idx, self.batch_size)
        #     outputs += self.base_model.generate([model_input] * n_samples, max_gen_len=512, end_token=")", hide_input=True).text
        outputs = self.base_model.generate([model_input] * 1, max_gen_len=256, end_token=")", hide_input=True).text[0]

        ## some post-process
        outputs = outputs.split('Input: ')[0]
        return_actions = outputs.split('\n')[:-1]
        return_actions = [y + _ + '\n' for _ in return_actions]
        print(f'propose actions: {return_actions}')
        return return_actions

    def fast_reward(self, state: game24State, action: game24Action) -> tuple[float, dict]:
        ## don't need fast_reward for beam search
        return 0

    def reward(self, state: game24State, action: game24Action, next_state: game24State) -> float:
        ## get values (state eval) for each action
        ## impossible, maybe, sure
        x, y = next_state[0], next_state[1]
        value_prompt = utils.value_prompt_wrap(x, y, self.prompt)
        # print(f'reward prompt: {value_prompt}')
        value_outputs = []
        for idx in range(0, self.n_eval, self.batch_size):
            n_samples = min(self.n_eval - idx, self.batch_size)
            value_outputs += self.base_model.generate([value_prompt] * n_samples, max_gen_len=256, hide_input=True).text
        ## postprocess
        ## find the first value result: impossible/sure/likely + \n
        ## by locating \n + num num num
        pattern = r"\n\d+ \d+ \d+( \d+|\n)"
        for i, v_o in enumerate(value_outputs):
            try:
                value_outputs[i] = v_o[:re.search(pattern, v_o).start()]
            except:
                # print(f'no matching: {v_o}')
                if 'sure' in v_o:
                    value_outputs[i] = 'sure'
                elif 'likely' in v_o:
                    value_outputs[i] = 'likely'
                else:
                    value_outputs[i] = 'impossible'
        value = utils.value_outputs_unwrap(x, y, value_outputs)
        print(f'new_state checking: {x}, {value_outputs}, value: {value}')
        return value
import copy
import re

from reasoners import SearchConfig, LanguageModel
from world_model import Game24State, Game24Action

from prompts.game24 import output_prompt, propose_prompt, value_prompt, value_last_step_prompt, value_map


class Game24config(SearchConfig[Game24State, Game24Action]):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=2,
                 depth_limit=4,
                 temperature=0.7,
                 n_eval=5) -> None:
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
        if state.current == '24':
            prompt = self.output_prompt_wrap(state)
        else:
            prompt = self.propose_prompt_wrap(state)
        output = self.base_model.generate([prompt], num_return_sequences=1).text[0]

        actions = output.split('\n')

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        actions = list(dict.fromkeys(actions))
        return actions

    def _reward(self, state: Game24State, action: Game24Action) -> float:
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

        value_outputs = []
        for idx in range(0, self.n_eval, self.batch_size):
            n_samples = min(self.n_eval - idx, self.batch_size)
            value_outputs += self.base_model.generate([prompt], do_sample=True, temperature=self.temperature,
                                                      num_return_sequences=n_samples).text

        value = self.retrieve_value(value_outputs)
        self.value_cache[value_prompt] = value
        return value

    def fast_reward(self, state: Game24State, action: Game24Action) -> tuple[float, dict]:
        reward = self._reward(state, action)
        return reward, {'reward': reward}

    # We calculate the full reward in fast_reward in Game24SearchConfig, direct return it
    def reward(self, state: Game24State, action: Game24Action,
               reward: float = None, **kwargs) -> tuple[float, dict]:
        return reward, {}

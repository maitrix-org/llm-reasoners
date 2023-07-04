import io
from typing import NamedTuple
from collections import defaultdict
from rap import WorldModel, LanguageModel
import utils


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str


game24State = list[SubResult]
game24Action = str


class game24WorldModel(WorldModel[game24State, game24Action]):
    """
    game24 World Model
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    State: x, y
    """

    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_confidence=8,
                 batch_size=2,) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence

    def init_state(self) -> list:
        ## input, output
        return (self.example, '')

    def step(self, state: game24State, action: game24Action) -> tuple[game24State, dict]:
        x, y = state[0], state[1]

        with io.StringIO() as f:
            f.write(utils.propose_prompt_wrap(x, y, self.prompt) + "\n")
            f.write(utils.value_prompt_wrap(x, y, self.prompt) + "\n")
            # model_input = f.getvalue()

        ## new action is the new state
        state = SubResult(x, action)
        # answer = utils.get_current_numbers(action)
        # state.append(SubResult(action, answer))
        return state

    def is_terminal(self, state: game24State) -> bool:
        ## if there is no left number or LLM is sure it can reach 24
        x, y = state[0], state[1]
        # last_line = y.strip().split('\n')[-1]
        current_numbers = utils.get_current_numbers(y if y else x)
        print(f'\ncurrent numbers in state: {current_numbers}')
        # if 'left: ' not in last_line or current_numbers == '24':
        if current_numbers == '24':
            return True
        else:
            return False

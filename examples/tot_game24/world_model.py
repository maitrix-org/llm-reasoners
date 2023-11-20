import io
from typing import NamedTuple
from collections import defaultdict
from reasoners import WorldModel, LanguageModel
import utils


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str
    step_id: int


game24State = list[SubResult]
game24Action = str


class game24WorldModel(WorldModel):
    """
    game24 World Model
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    step_id: bfs level
    State: x, y, step_id
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
        # x = '4 5 6 10'
        # y = '4 * 5 = 20 (left: 6 10 20)\n10 - 6 = 4 (left: 4 20)\n4 + 20 = 24 (left: 24)\n'
        # x = '1 2 4 7'
        # y = '7 - 1 = 6 (left: 2 4 6)\n6 - 2 = 4 (left: 4 6)\n4 * 6 = 24 (left: 24)\n'
        # return (x, y, 0)
        return (self.example, '', 0)

    def step(self, state: game24State, action: game24Action) -> tuple[game24State, dict]:
        x, y, step_id = state[0], state[1], state[2]

        # with io.StringIO() as f:
        #     f.write(utils.propose_prompt_wrap(x, y, self.prompt) + "\n")
        #     f.write(utils.value_prompt_wrap(x, y, self.prompt) + "\n")
            # model_input = f.getvalue()

        ## new action is the new state
        state = SubResult(x, action, step_id + 1)
        # answer = utils.get_current_numbers(action)
        # state.append(SubResult(action, answer))
        return state, {'new_state': state}

    def is_terminal(self, state: game24State) -> bool:
        ## if there is no left number or LLM is sure it can reach 24
        x, y, step_id = state[0], state[1], state[2]
        # if step_id == 4:
        #     exit()
        last_line = y.strip().split('\n')[-1]
        flatten_y = y.strip().replace('\n', '->')
        print(f"-- new action --{flatten_y}")
        current_numbers = utils.get_current_numbers(y if y else x)
        # print(f'\ncurrent numbers in state: {current_numbers} at step {state[2]}')
        if 'answer: ' in last_line.lower():
            print(f"terminal states with answer: {last_line}")
        return False

from reasoners import WorldModel, LanguageModel
from typing import NamedTuple

from utils import get_indent
import io
import regex
import math
import re
from utils import ANS_RE, extract_answer


class SubResult(NamedTuple):
    action: str
    action_prob: float
    action_length: int
    evaluation: str
    action_confidence: float

GSM8kState = list[SubResult]
# action should include both the action and the action_prob
GSM8kAction = (str, float, int) # (action, action_prob, action_length)
GSM8kExample = str


class GSM8kWorldModel(WorldModel[GSM8kState, GSM8kAction, GSM8kExample]):
    """
    GSM8k World Model
    State: [(action, action_prob, evaluation, action_confidence), ...]
    Action: step
    """

    def __init__(self,
                base_model: LanguageModel,
                eval_num: int = 1, # only 1 evaluation is needed
                temperature: float = 0.0 # set to greedy for confidence
                ) -> None:
        
        super().__init__()

        self.base_model = base_model
        self.eval_num = eval_num
        self.temperature = temperature

    def init_state(self) -> list:
        return []

    # given a set of actions, find out which action to take, here we get the discriminator to filter out some of the generated texts.
    def step(self, state: GSM8kState, action: GSM8kAction) -> tuple[GSM8kState, dict]:
        state = state.copy()
        # print(f" in world model : state: {state}")
        # print(f"in world model :  action : {action}")
        state.append(SubResult(
            action=action[0], action_prob = 0.0,action_length =0, evaluation="", action_confidence=0.0 ))

        return state, {}


    def is_terminal(self, state: GSM8kState) -> bool:
        # several ways to terminate:
        # 1. state > 0
        # 2. the last state's action is "    return result" (do rstrip() to remove \n)
        # 3. or the last state is empty 

        if len(state) > 0:
            generated_ans = ''.join([x.action for x in state])
            return "[invalid]" != extract_answer(generated_ans)
        
        return False
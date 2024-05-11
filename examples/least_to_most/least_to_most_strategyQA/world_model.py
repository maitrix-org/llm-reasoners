import io
from typing import NamedTuple, TypedDict
from collections import defaultdict
from prompt import solve_prompt
from reasoners import WorldModel, LanguageModel
import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))


class SubResult(NamedTuple):
    sub_question: str
    sub_answer: str


StrategyQAState = list[SubResult]
StrategyQAAction = str


class StrategyQAWorldModel(WorldModel):
    """
    StrategyQA world model
    State: [[sub_question_1, sub_answer_1], [sub_question_2, sub_answer_2], ...]
    Action: sub_question
    """

    def __init__(self,
                 base_model: LanguageModel,
                 temperature=0.7,
                 log_dir=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.log_dir = log_dir

    def init_state(self) -> list:
        return []

    def step(self, state: StrategyQAState, action: StrategyQAAction) -> tuple[StrategyQAState, dict]:
        state = state.copy()

        with io.StringIO() as f:
            f.write(solve_prompt.strip() + "\n\n")
            f.write(self.example + "\n\n")
            # iterate over state
            for sub_question, sub_answer in state:
                f.write(f"Q: {sub_question}\nA: {sub_answer}\n\n")
            
            f.write(f"Q: {action}\n")
            f.write("A:")
        
            model_input = f.getvalue()

        sub_answer = self.base_model.generate(
            [model_input],
            hide_input=True,
            top_k=32000,
            top_p=0.95,
            do_sample=True,
            temperature=self.temperature,
            eos_token_id='\n'
        ).text[0].strip()

        # local rank
        if local_rank == 0:
            with open(self.log_dir+"/log.txt", "a") as f:
                print(f"ACTION: {action}", file=f)
                print(f"ANSWER: {sub_answer}", file=f)
                print(f"\n", file=f)

        state.append(SubResult(action, sub_answer))

        return state, {}

    def is_terminal(self, state: StrategyQAState) -> bool:
        if len(state) > 0 and state[-1].sub_question == self.example:
            return True
        else:
            return False

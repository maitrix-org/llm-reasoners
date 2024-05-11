import sys
from dataclasses import dataclass
from typing import Optional

import prompts.output
import prompts.transition
from reasoners import WorldModel, LanguageModel
from reasoners.base import Example
from examples.prontoqa.dataset import ProntoQAExample

@dataclass
class ProntoQAState:
    last_state: "Optional[ProntoQAState]"
    last_action: "Optional[ProntoQAAction]"
    body: str

    def __str__(self) -> str:
        return self.body


ProntoQAAction = str


class ProntoQAWorldModel(WorldModel[ProntoQAState, ProntoQAAction, ProntoQAExample]):
    def __init__(self, base_model: LanguageModel) -> None:
        super().__init__()
        self.base_model = base_model
        self.example: ProntoQAExample = self.example

    def init_state(self) -> ProntoQAState:

        *base_facts, init_state = self.example.test_example.question.split(". ")

        return ProntoQAState(body=init_state, last_state=None, last_action=None)

    def step(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[ProntoQAState, dict]:

        input_prompt = ""

        match action:
            case "Finish.":  # transition to terminal state
                input_prompt += prompts.output.EXAMPLES
                input_prompt += prompts.output.QUERY_FORMAT.format(self.example.test_example.query)
                input_prompt += prompts.output.CLAIM_FORMAT.format(state)
                input_prompt += prompts.output.OUTPUT_PREFIX
                print("Reached terminal state.")

            case _:  # transition to non-terminal state
                input_prompt += prompts.transition.EXAMPLES
                input_prompt += prompts.transition.FACTS_FORMAT.format(state, action)
                input_prompt += prompts.transition.NEXT_CLAIM_PREFIX
                print("Reached non-terminal state.")

        output = self.base_model.generate([input_prompt], eos_token_id="\n", hide_input=True, temperature=0).text[
            0].strip()

        print(input_prompt, file=sys.stderr, flush=True)
        print(f"S[{state}] A[{action}] -> S'[{output}]", flush=True)
        return ProntoQAState(body=output, last_state=state, last_action=action), {}

    def is_terminal(self, state: ProntoQAState) -> bool:

        return state.last_action == "Finish."

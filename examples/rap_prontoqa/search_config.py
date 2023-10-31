import sys
from typing import Optional
import torch 

import prompts.finish
import prompts.valid
from examples.rap_prontoqa.dataset import ProntoQAExample
from reasoners import SearchConfig
from world_model import ProntoQAState, ProntoQAAction, ProntoQAWorldModel


class ProntoQAConfig(SearchConfig[ProntoQAState, ProntoQAAction]):

    def __init__(self, world_model: ProntoQAWorldModel):
        super().__init__()
        self.world_model = world_model
        self.example: ProntoQAExample = self.example

    def get_actions(self, state: ProntoQAState) -> list[ProntoQAAction]:
        *base_facts, init_state = self.example.test_example.question.split(". ")
        facts = base_facts + ["Finish."]

        return facts

    def fast_reward(
            self,
            state: ProntoQAState,
            action: ProntoQAAction,
    ) -> tuple[float, dict]:
        return self.reward(state, action)

    def reward(self,
               state: ProntoQAState,
               action: ProntoQAAction,
               **kwargs
               ) -> tuple[float, dict]:

        if not state.last_state:
            return 0, {}

        input_prompt = ""

        match action:
            case "Finish.":
                input_prompt += prompts.finish.EXAMPLES
                input_prompt += prompts.finish.TARGET_FORMAT.format(self.example.test_example.query)
                input_prompt += prompts.finish.CLAIM_FORMAT.format(state)
                input_prompt += prompts.finish.OUTPUT_PREFIX
            case _:
                input_prompt += prompts.valid.EXAMPLES
                input_prompt += prompts.valid.FACTS_FORMAT.format(state.last_state or "", action)
                input_prompt += prompts.valid.NEXT_STEP_FORMAT.format(state)
                input_prompt += prompts.valid.VALID_PREFIX

        output_logits = self.world_model.base_model.get_next_token_logits(
            input_prompt,
            candidates=["Yes", "No"]
            # ,
            # postprocess="softmax"
        )


        print(f"output_logits shape: {len(output_logits)} {len(output_logits[0])} {output_logits}")
        # reward: float = output_logits[0][0].item()
        reward:float = torch.softmax(torch.tensor(output_logits[0]), dim=0)[0].item()

        print(input_prompt, file=sys.stderr, flush=True)
        match action:
            case "Finish.":
                print(f"S[{state}] Q[{self.example.test_example.query}] -> R[{reward}]", flush=True)
            case _:
                print(f"S[{state.last_state}] A[{action}] S'[{state}] -> R[{reward}]", flush=True)

        return reward, {}

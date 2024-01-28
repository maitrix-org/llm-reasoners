import sys
from typing import Optional
import torch 

import prompts.finish
import prompts.valid
import prompts.next_step
from examples.prontoqa.dataset import ProntoQAExample
from reasoners import SearchConfig, LanguageModel
from world_model import ProntoQAState, ProntoQAAction, ProntoQAWorldModel


def format_examples(sampled_data):
    formatted_examples = ""
    for i, entry in enumerate(sampled_data, 1):
        facts = f"Facts {i}: {entry['Facts']}\n"
        query = f"Query {i}: {entry['Query']}\n"
        claims_and_next = ""

        for j, (claim, next_step) in enumerate(zip(entry['claims'], entry['next_steps']), 1):
            claims_and_next += f"Claim {i}.{j}: {claim}\nNext {i}.{j}: {next_step}\n"

        formatted_examples += facts + query + claims_and_next + "\n"

    return formatted_examples


class ProntoQAConfig(SearchConfig[ProntoQAState, ProntoQAAction,ProntoQAExample]):

    def __init__(self, base_model: LanguageModel, temperature=0.8, n_candidates=4):
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.n_candidates = n_candidates
        self.example: ProntoQAExample = self.example

    def get_actions(self, state: ProntoQAState) -> list[ProntoQAAction]:
        # *base_facts, init_state = self.example.test_example.question.split(". ")
        # facts = base_facts + ["Finish."]
        *base_facts, init_state = self.example.test_example.question.split(". ")

        input_prompt = ""
        # input_prompt += prompts.next_step.EXAMPLES
        input_prompt += format_examples(self.prompt)
        input_prompt += prompts.next_step.FACTS_FORMAT.format(len(self.prompt),". ".join(base_facts))
        input_prompt += prompts.next_step.QUERY_FORMAT.format(len(self.prompt), self.example.test_example.query)
        input_prompt += prompts.next_step.CLAIM_FORMAT.format(len(self.prompt),state)
        input_prompt += prompts.next_step.NEXT_STEP_PREFIX.format(len(self.prompt))

        # print(f"input_prompt: {input_prompt}")
        outputs = self.base_model.generate([input_prompt] * self.n_candidates, eos_token_id="\n", hide_input=True, temperature=self.temperature, do_sample=True).text
        outputs = [output.strip() for output in outputs]
        # deduplicate
        outputs = list(dict.fromkeys(outputs))

        return outputs

    # OLD fast reward code
    def fast_reward(
            self,
            state: ProntoQAState,
            action: ProntoQAAction,
    ) -> tuple[float, dict]:
        """
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
        )

        

        # reward: float = output_logits[0][0].item()
        reward:float = torch.softmax(torch.tensor(output_logits[0]), dim=0)[0].item()

        print(input_prompt, file=sys.stderr, flush=True)
        match action:
            case "Finish.":
                print(f"S[{state}] Q[{self.example.test_example.query}] -> R[{reward}]", flush=True)
            case _:
                print(f"S[{state.last_state}] A[{action}] S'[{state}] -> R[{reward}]", flush=True)
        """
        # intuition reward

        *base_facts, init_state = self.example.test_example.question.split(". ")

        input_prompt = ""
        # input_prompt += prompts.next_step.EXAMPLES
        input_prompt += format_examples(self.prompt)
        input_prompt += prompts.next_step.FACTS_FORMAT.format(len(self.prompt),". ".join(base_facts))
        input_prompt += prompts.next_step.QUERY_FORMAT.format(len(self.prompt),self.example.test_example.query)
        input_prompt += prompts.next_step.CLAIM_FORMAT.format(len(self.prompt),state)
        input_prompt += prompts.next_step.NEXT_STEP_PREFIX.format(len(self.prompt))
        outputs = input_prompt + " " + action
        intuition = self.base_model.get_loglikelihood(input_prompt, [outputs])[0]


        return intuition, {"self-eval": 0, "intuition": intuition}
    
    def fast_reward_new(self, state: ProntoQAState, action: ProntoQAAction) -> tuple[float, dict]:
        # TODO: currently trying to implement this
        previous_action = state.buffered_action + "\n" if state.buffered_action != "" else ""

        input_prompt=""
        # input_prompt += prompts.next_step.EXAMPLES
        input_prompt += format_examples(self.prompt)
        input_prompt += prompts.next_step.FACTS_FORMAT.format(state.last_state or "", action)
        input_prompt += prompts.next_step.NEXT_STEP_FORMAT.format(state)
        input_prompt += prompts.next_step.VALID_PREFIX
        
        icl_template = self.prompt["icl_list"][state.step_idx // 2]

    
        intuition = self.base_model.get_loglikelihood(input_prompt, [input_prompt + action])[0]



        return self.calculate_reward(intuition, self_eval), {'intuition': intuition}


    # using this from blocksworld to calculate reward, have to change.
    def calculate_reward(self, intuition, self_eval, goal_reached=None):
        # to provide a unified interface for reward and fast_reward
        if goal_reached is None:
            goal_reward = self.goal_reward_default
        elif goal_reached[0]:
            goal_reward = self.goal_reached_reward
        else:
            goal_reward = goal_reached[1]
        return (intuition + self_eval) * self.reward_alpha + goal_reward * (1 - self.reward_alpha)

    def reward(self,
               state: ProntoQAState,
               action: ProntoQAAction,
               **kwargs
               ) -> tuple[float, dict]:
        self_eval = kwargs["self-eval"]
        intuition = kwargs["intuition"]
        return intuition, {"self-eval": self_eval, "intuition": intuition}
        # if not state.last_state:
        #     return 0.1, {}

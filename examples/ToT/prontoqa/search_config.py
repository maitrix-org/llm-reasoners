import sys
from typing import Optional
import torch 

import prompts.finish
import prompts.valid_rap
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
        
        *base_facts, init_state = self.example.test_example.question.split(". ")

        input_prompt = ""
        input_prompt += format_examples(self.prompt)
        input_prompt += prompts.next_step.FACTS_FORMAT.format(len(self.prompt) + 1,". ".join(base_facts))
        input_prompt += prompts.next_step.QUERY_FORMAT.format(len(self.prompt) + 1, self.example.test_example.query)
        input_prompt += prompts.next_step.CLAIM_FORMAT.format(len(self.prompt) + 1, state)
        input_prompt += prompts.next_step.NEXT_STEP_PREFIX.format(len(self.prompt) + 1)

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
        *base_facts, init_state = self.example.test_example.question.split(". ")
        input_prompt = ""
        match action:
            case "Finish.":
                input_prompt += prompts.finish.EXAMPLES
                input_prompt += prompts.finish.TARGET_FORMAT.format(self.example.test_example.query)
                input_prompt += prompts.finish.CLAIM_FORMAT.format(state)
                input_prompt += prompts.finish.OUTPUT_PREFIX
            case _:
                input_prompt = prompts.valid_rap.TEMPLATE.replace("[[STATE]]", state.body)\
                    .replace("[[ACTION]]", action)\
                    .replace("[[QUERY]]", self.example.test_example.query)\
                    .replace("[[FACTS]]", ". ".join(base_facts) + ".")

        output_logits = self.base_model.get_next_token_logits(
            input_prompt,
            candidates=["Yes", "No"]
        )
        print("output_logits: ", output_logits)
        # self_eval:float = torch.softmax(torch.tensor(output_logits[0]), dim=0)[0].item()
        self_eval:float = output_logits[0][0].item()

        # intuition reward

        *base_facts, init_state = self.example.test_example.question.split(". ")

        input_prompt = ""
        # input_prompt += prompts.next_step.EXAMPLES
        input_prompt += format_examples(self.prompt)
        input_prompt += prompts.next_step.FACTS_FORMAT.format(len(self.prompt) + 1,". ".join(base_facts))
        input_prompt += prompts.next_step.QUERY_FORMAT.format(len(self.prompt) + 1, self.example.test_example.query)
        input_prompt += prompts.next_step.CLAIM_FORMAT.format(len(self.prompt) + 1, state)
        input_prompt += prompts.next_step.NEXT_STEP_PREFIX.format(len(self.prompt) + 1)
        outputs = input_prompt + " " + action
        intuition = self.base_model.get_loglikelihood(input_prompt, [outputs])[0]

        match action:
            case "Finish.":
                print(f"S[{state}] Q[{self.example.test_example.query}] -> Self-eval[{self_eval}] Intuition[{intuition}]", flush=True)
            case _:
                print(f"S[{state.last_state}] A[{action}] S'[{state}] -> Self-eval[{self_eval}] Intuition[{intuition}]", flush=True)

        return intuition + self_eval, {"self-eval": self_eval, "intuition": intuition}

    def reward(self,
               state: ProntoQAState,
               action: ProntoQAAction,
               **kwargs
               ) -> tuple[float, dict]:
        self_eval = kwargs["self-eval"]
        intuition = kwargs["intuition"]
        return intuition + self_eval, {"self-eval": self_eval, "intuition": intuition}

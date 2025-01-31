import numpy as np
import time
import torch

from world_model import MathState, MathAction, MathModel
from reasoners import SearchConfig, LanguageModel
from loguru import logger

from transformers import AutoTokenizer

class MathConfig(SearchConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        prm: LanguageModel,
        prompt: dict,
        batch_size=8,
        num_actions=3,
        temperature=0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.reward_model = prm
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.temperature = temperature

    def get_actions(self, state: MathState) -> list[MathAction]:
        start_time = time.time()
        problem_state = (
            "## Step " + "\n\n## Step ".join([f"{step}" for step in state.steps])
            if len(state.steps) != 0
            else ""
        )

        prompts = (
            self.prompt["icl"]
            .replace("<init_state>", self.example["init"])
            .replace("<problem_state>", problem_state)
        )

        response = self.base_model.generate(
            [prompts],
            num_return_sequences=self.num_actions,
            temperature=self.temperature,
            do_sample=True,
            stop=f"## Step {state.step_idx + 2}",
        )  # TODO: Initial actions which are already taken might change here. Check if this is being correctly handled.
        
        actions = [
            action.replace(f"## Step {state.step_idx + 2}", "").strip() for action in response[0]
        ]

        logger.debug(
            f"Generated actions at step {state.step_idx} are:\n{actions[0]}"
        )
        logger.debug(f"TIME: Generating actions took {time.time() - start_time} seconds")
        return actions

    # Reward function for peiyi9979/math-shepherd-mistral-7b-prm
    def reward(
        self,
        state: MathState,
        action: MathAction,
        intuition: float = None
    ) -> float:

        good_token = "+"
        bad_token = "-"
        step_tag = "ки"

        current_problem_state = "\n".join(
            [f"Step {step.strip()} {step_tag}" for step in state.steps]
        )

        action_to_take, _, _ = MathModel.step_helper(state, action)

        current_problem_state += (
            f"\nStep {state.step_idx + 1} {action_to_take} {step_tag}"
        )

        input_for_prm = f"{self.example['init']} {current_problem_state}"
        intuition = np.exp(self.reward_model.get_loglikelihood(input_for_prm + " ", [input_for_prm + " " + good_token])[0])
        # the probability of the good token and the bad token always sum to 1
        # so we can just take the probability of the good token

        logger.debug(
            f"Reward for step {state.step_idx} is: {intuition} where the potential step is: {action_to_take}"
        )

        return intuition, {"intuition": intuition}
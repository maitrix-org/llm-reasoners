import numpy as np
import torch
import time

from world_model import MathState, MathAction, MathModel
from reasoners import SearchConfig, LanguageModel
from loguru import logger

class MathConfig(SearchConfig):
    def __init__(
        self,
        base_model: LanguageModel,
        prm: LanguageModel,
        prompt: dict,
        batch_size=8,
        reward_alpha=0.5,
        goal_reward_default=0.0,
        goal_reached_reward=0.5,
        num_actions=3,
        temperature=0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.reward_model = prm
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.reward_alpha = reward_alpha
        self.goal_reward_default = goal_reward_default
        self.goal_reached_reward = goal_reached_reward
        self.num_actions = num_actions
        self.temperature = temperature

    def get_actions(self, state: MathState) -> list[MathAction]:

        start_time = time.time()
        problem_state = (
            "## Step " + "\n\n## Step ".join([f"{step}" for step in state.steps])
            if len(state.steps) != 0
            else ""
        )

        current_step = len(state.steps) + 1

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
        logger.info(f"TIME: Generating actions took {time.time() - start_time} seconds")
        return actions

    def fast_reward(self, state: MathState, action: MathAction) -> tuple[float, dict]:
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

        self_eval = None  # remove if we use self-eval later

        logger.debug(
            f"Reward for step {state.step_idx} is: {intuition} where the potential step is: {action_to_take}"
        )

        return self.calculate_reward(intuition, self_eval, state.end), {
            "intuition": intuition,
            "self_eval": self_eval,
        }

    def calculate_reward(self, intuition, self_eval=None, goal_reached=False):
        if not goal_reached:
            goal_reward = self.goal_reward_default
        else:
            goal_reward = self.goal_reached_reward

        if self_eval is not None:
            return (
                intuition + self_eval
            ) + goal_reward
        else:
            return intuition + goal_reward


    def reward(
        self,
        state: MathState,
        action: MathAction,
        intuition: float = None,
        self_eval: float = None,
        goal_reached: tuple[bool, float] = False,
    ) -> float:
        assert (
            intuition is not None
        ), "intuition is required to calculate reward in this search config, consider passing it in fast_reward"
        assert (
            goal_reached is not None
        ), "goal_reached is required to calculate reward in this search config, consider passing it in world model's step"
        return (
            self.calculate_reward(intuition, self_eval=None, goal_reached=False),
            {"intuition": intuition, "goal_reached": goal_reached},
        )

    def update_example(self, example, prompt=None) -> None:
        super().update_example(example, prompt=prompt)

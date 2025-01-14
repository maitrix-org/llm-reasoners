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

        actions = self.base_model.generate(
            [prompts],
            num_return_sequences=self.num_actions,
            temperature=self.temperature,
            do_sample=True,
            stop=f"## Step {state.step_idx + 2}",
        )  # TODO: Initial actions which are already taken might change here. Check if this is being correctly handled.
        logger.debug(
            f"Generated actions at step {state.step_idx} are:\n{actions[0][0]}"
        )
        logger.info(f"TIME: Generating actions took {time.time() - start_time} seconds")
        return actions[0]

    def fast_reward(self, state: MathState, action: MathAction) -> tuple[float, dict]:
        good_token = "+"
        bad_token = "-"
        step_tag = "ки"

        current_problem_state = state.problem_state
        current_problem_state = "\n".join(
            [f"Step {step.strip()} {step_tag}" for step in state.steps]
        )

        action_to_take, _, _ = MathModel.step_helper(state, action)

        current_problem_state += (
            f"\nStep {state.step_idx + 1} {action_to_take} {step_tag}"
        )

        candidate_tokens = self.reward_model.tokenizer.encode(
            f"{good_token} {bad_token}"
        )[1:]
        step_tag_id = self.reward_model.tokenizer.encode(f"{step_tag}")[-1]

        input_for_prm = f"{self.example['init']} {current_problem_state}"
        input_id = torch.tensor([self.reward_model.tokenizer.encode(input_for_prm)])

        with torch.no_grad():
            logits = self.reward_model.model(input_id).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores = scores[input_id == step_tag_id]

        try:
            intuition = float(step_scores[-1])
        except IndexError:
            intuition = 0.0

        self_eval = None  # remove if we use self-eval later

        if len(state.steps) != 0:
            logger.debug(
                f"Reward for step {state.step_idx} is: {intuition} where the potential step is: {action_to_take}"
            )

        return self.calculate_reward(intuition, self_eval, state.end), {
            "intuition": intuition,
            "self_eval": self_eval,
        }

    def batched_fast_reward(self, state, actions: list[MathAction]) -> tuple[list[float], list[dict]]:

        import time

        start_time = time.time()

        good_token = "+"
        bad_token = "-"
        step_tag = "ки"
        
        batch_size = len(actions)
        
        current_problem_state = state.problem_state
        current_problem_state = "\n".join(
            [f"Step {step.strip()} {step_tag}" for step in state.steps]
        )

        current_problem_states = [current_problem_state] * batch_size
        
        actions_to_take = []
        for i, action in enumerate(actions):
            action_to_take, _, _ = MathModel.step_helper(state, action)
            current_problem_states[i] += f"\nStep {state.step_idx + 1} {action_to_take} {step_tag}"
            actions_to_take.append(action_to_take)
        
        candidate_tokens = self.reward_model.tokenizer.encode(f"{good_token} {bad_token}")[1:]
        step_tag_id = self.reward_model.tokenizer.encode(f"{step_tag}")[-1]
        
        input_texts = [f"{self.example['init']} {problem_state}" for problem_state in current_problem_states]
        input_ids = self.reward_model.tokenizer.batch_encode_plus(input_texts)["input_ids"]
        
        max_length = max(len(ids) for ids in input_ids)
        padded_input_ids = [ids + [self.reward_model.tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
        input_tensor = torch.tensor(padded_input_ids)
        
        with torch.no_grad():
            logits = self.reward_model.model(input_tensor).logits[:, :, candidate_tokens]
            scores = logits.softmax(dim=-1)[:, :, 0]
            
            step_scores = []
            attention_mask = (input_tensor != self.reward_model.tokenizer.pad_token_id)
            
            for i in range(batch_size):
                step_positions = (input_tensor[i] == step_tag_id) & attention_mask[i]
                sequence_scores = scores[i][step_positions]
                step_scores.append(sequence_scores)
        
        intuitions = []
        for scores in step_scores:
            try:
                intuition = float(scores[-1])
            except IndexError:
                intuition = 0.0
            intuitions.append(intuition)
        
        for i in range(batch_size):
            if len(state.steps) != 0:
                logger.debug(
                    f"Reward for step {state.step_idx} is: {intuitions[i]} where the potential step is: {actions_to_take[i]}"
                )
        
        rewards_and_meta = []
        for i in range(batch_size):
            reward = self.calculate_reward(intuitions[i], None, state.end)
            meta = {
                "intuition": intuitions[i],
                "self_eval": None,
            }
            rewards_and_meta.append((reward, meta))
        
        rewards, metadata = zip(*rewards_and_meta)

        logger.info(f"TIME: Batched fast reward took {time.time() - start_time} seconds")
        
        return list(rewards), list(metadata)
        

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

    def batch_reward(self, states: list[MathState], actions: list[MathAction]):
        return self.batched_fast_reward(states, actions)

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

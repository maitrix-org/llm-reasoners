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
        prm_tokenizer_path: str,
        prompt: dict,
        batch_size=8,
        num_actions=3,
        temperature=0.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.reward_model = prm
        self.prm_tokenizer = AutoTokenizer.from_pretrained(prm_tokenizer_path)
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.temperature = temperature
        
        # Precompute token IDs for '+' and '-'
        self.plus_token_id = self.prm_tokenizer.encode('+', add_special_tokens=False)[-1]
        self.minus_token_id = self.prm_tokenizer.encode('-', add_special_tokens=False)[-1]

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
    # def reward(
    #     self,
    #     state: MathState,
    #     action: MathAction,
    #     intuition: float = None
    # ) -> float:

    #     good_token = "+"
    #     bad_token = "-"
    #     step_tag = "ки"

    #     current_problem_state = "\n".join(
    #         [f"Step {step.strip()} {step_tag}" for step in state.steps]
    #     )

    #     action_to_take, _, _ = MathModel.step_helper(state, action)

    #     current_problem_state += (
    #         f"\nStep {state.step_idx + 1} {action_to_take} {step_tag}"
    #     )

    #     input_for_prm = f"{self.example['init']} {current_problem_state}"
    #     intuition = np.exp(self.reward_model.get_loglikelihood(input_for_prm + " ", [input_for_prm + " " + good_token])[0])
    #     # the probability of the good token and the bad token always sum to 1
    #     # so we can just take the probability of the good token

    #     logger.debug(
    #         f"Reward for step {state.step_idx} is: {intuition} where the potential step is: {action_to_take}"
    #     )

    #     return intuition, {"intuition": intuition}

    # Reward function for RLHFlow/Llama3.1-8B-PRM-Deepseek-Data while using HF Model
    # def reward(
    #     self,
    #     state: MathState,
    #     action: MathAction,
    #     intuition: float = None
    # ) -> float:
    #     problem_prompt = self.example['init']
    #     existing_steps = state.steps
    #     action_to_take, _, _ = MathModel.step_helper(state, action)
        
    #     # Build conversation history
    #     conversation = []
    #     if existing_steps:
    #         first_step = existing_steps[0]
    #         conversation.append({"role": "user", "content": f"{problem_prompt} {first_step}"})
    #         conversation.append({"role": "assistant", "content": "+"})
            
    #         for step in existing_steps[1:]:
    #             conversation.append({"role": "user", "content": step})
    #             conversation.append({"role": "assistant", "content": "+"})
        
    #     new_content = action_to_take if existing_steps else f"{problem_prompt} {action_to_take}"
    #     conversation.append({"role": "user", "content": new_content})
        
    #     # Tokenize conversation
    #     input_ids = self.prm_tokenizer.apply_chat_template(
    #         conversation,
    #         return_tensors="pt"
    #     ).to(self.reward_model.device)
        
    #     # Get logits for +/- tokens at -3 position
    #     with torch.no_grad():
    #         outputs = self.reward_model.model(input_ids)
    #         logits = outputs.logits[0, -3, [self.plus_token_id, self.minus_token_id]]
        
    #     # Calculate probability of '+'
    #     intuition = torch.softmax(logits, dim=-1)[0].item()
        
    #     logger.debug(f"Reward for step {state.step_idx}: {intuition} | Step: {action_to_take}")
    #     return intuition, {"intuition": intuition}

    def reward(
        self,
        state: MathState,
        action: MathAction,
        intuition: float = None
    ) -> float:
        problem_prompt = self.example['init']
        existing_steps = state.steps
        action_to_take, _, _ = MathModel.step_helper(state, action)
        
        # Build conversation history using the PRM's tokenizer
        conversation = []
        if existing_steps:
            first_step = existing_steps[0]
            conversation.append({"role": "user", "content": f"{problem_prompt} {first_step}"})
            conversation.append({"role": "assistant", "content": "+"})
            
            for step in existing_steps[1:]:
                conversation.append({"role": "user", "content": step})
                conversation.append({"role": "assistant", "content": "+"})
        
        new_content = action_to_take if existing_steps else f"{problem_prompt} {action_to_take}"
        conversation.append({"role": "user", "content": new_content})
        
        # Generate formatted input string using PRM's chat template
        input_ids = self.prm_tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True
        )
        input_for_prm = self.prm_tokenizer.decode(input_ids, skip_special_tokens=False)

        # Get log probabilities for +/- using SGLang's API
        prefix = input_for_prm.strip() + " "
        good_content = prefix + "+"
        bad_content = prefix + "-"
        
        # Get normalized log probabilities for both options
        log_probs = self.reward_model.get_loglikelihood(
            prefix=prefix,
            contents=[good_content, bad_content]
        )
        
        # Calculate probability of '+' using softmax
        log_prob_plus, log_prob_minus = log_probs
        intuition = np.exp(log_prob_plus) / (np.exp(log_prob_plus) + np.exp(log_prob_minus))
        
        logger.debug(f"Reward for step {state.step_idx}: {intuition} | Step: {action_to_take}")
        return intuition, {"intuition": intuition}
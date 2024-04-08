import io
import numpy as np

from world_model import MATHState, MATHAction
from reasoners import SearchConfig, LanguageModel


class MATHConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 useful_prompt: dict,
                 n_actions=4,
                 batch_size=2,
                 temperature=0.8,
                 reward_alpha=0.5,
                 reward_confidence_default=0.8,
                 depth_limit=5,
                 force_terminating_on_depth_limit=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.useful_prompt = useful_prompt
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default

    def get_actions(self, state: MATHState) -> list[MATHAction]:
        with io.StringIO() as f:
            f.write(self.prompt["input"])
            f.write(self.prompt["question_prefix"] + self.example + "\n")
            for idx, (q, a, _) in enumerate(state):
                f.write(self.prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
                f.write(self.prompt["answer_prefix"].format(idx + 1) + " " + a + "\n")
            f.write(self.prompt["subquestion_prefix"].format(len(state) + 1))
            if self.force_terminating_on_depth_limit and len(state) + 1 >= self.depth_limit:
                f.write(" " + self.prompt["overall_question_prefix"])
            model_input = f.getvalue()

        outputs = []
        for idx in range(0, self.n_actions, self.batch_size):
            n_samples = min(self.n_actions - idx, self.batch_size)
            outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=self.temperature,
                                                eos_token_id='\n').text

        return_actions = [output.strip() for output in outputs]
        return return_actions

    def fast_reward(self, state: MATHState, action: MATHAction) -> tuple[float, dict]:
        with io.StringIO() as f:
            f.write(self.useful_prompt["input"])
            f.write(self.useful_prompt["question_prefix"] + self.example + "\n")
            for idx, (q, _, _) in enumerate(state):
                f.write(self.useful_prompt["subquestion_prefix"].format(idx + 1) + " " + q + "\n")
            f.write(self.useful_prompt["new_subquestion_prefix"].format(len(state) + 1) + " " + action + "\n")
            f.write(self.useful_prompt["useful_prefix"])
            model_input = f.getvalue()

        logits = self.base_model.get_next_token_logits(model_input, ["Yes", "No"])[0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        useful_prob = probs[0]
        fast_reward, _ = self.calculate_reward(useful_prob)
        return fast_reward, {'r_useful': useful_prob}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful, 'r_conf': r_conf}

    def reward(self, state: MATHState, action: MATHAction,
               r_useful: float = None,
               confidence: float = None) -> float:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)

import io
import numpy as np
from prompt import code_prompt
from world_model import GSM8kState, GSM8kAction
from reasoners import SearchConfig, LanguageModel
from typing import Tuple
from reasoners.lm import OpenAIModel, Llama2Model, Llama3Model

class GSM8kConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 n_actions=16,
                 temperature=1,
                 reward_alpha=0.5,
                 depth_limit=10,
                 force_terminating_on_depth_limit=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.temperature = temperature
        self.n_actions = n_actions
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha

    def get_actions(self, state: GSM8kState) -> list[GSM8kAction]:
        with io.StringIO() as f:
            f.write(code_prompt)
            f.write("\n\n\n\n\n")
            f.write(f'Q: {self.example}\n\n# solution in Python:\n\n\ndef solution():\n    """{self.example}"""\n')
            for a, _, _, _ , _ in state:
                f.write(f"{a}\n")
            
            # get the prompt
            model_input = f.getvalue()
        
        # let's prompt
        if isinstance(self.base_model, OpenAIModel):
            eos_token_id = []
        elif isinstance(self.base_model, Llama2Model):
            eos_token_id = ["\n"]
        elif isinstance(self.base_model, Llama3Model):
            eos_token_id = ["\n\n", ".\n", ".\n\n","\n"]
            
        outputs = self.base_model.generate(model_input,
                                    temperature=self.temperature,
                                    max_tokens=256,
                                    top_p=1,
                                    num_return_sequences=self.n_actions,
                                    stop="\n",
                                    logprobs=5,
                                    hide_input=True,
                                    do_sample=True,
                                    eos_token_id=eos_token_id)
        
        # outputs = self.base_model.generate(prompt=model_input,
        #                                     max_tokens=256,
        #                                     temperature=self.temperature,
        #                                     top_p=1,
        #                                     num_return_sequences=self.n_actions,
        #                                     stop="\n",
        #                                     logprobs=5)
        
        return_actions = []
        for i in range(self.n_actions):
            action = outputs.text[i]
            # de-duplicate if the action is already in return_actions
            if action in [a[0] for a in return_actions]:
                continue

            log_prob = outputs.log_prob[i]
            # get the token_logprobs
            token_logprobs = log_prob["token_logprobs"]
            # let's calculate the probability of this action, without normalizing
            action_prob = np.exp(sum(token_logprobs))
            # action length
            action_length = len(token_logprobs)

            return_actions.append((action, action_prob, action_length))

        return return_actions

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        return 0, {}

    def reward(self, state: GSM8kState, 
               action: GSM8kAction,
               action_confidence: float = None,
               **kwargs) -> float:
        
        assert action_confidence is not None, "action_confidence should not be None"

        # reward should be accumulated
        acc_action_prob, acc_action_length, acc_action_confidence = 1, 0, 1
        for _, prob, length, _, confidence in state:
            acc_action_prob *= prob
            acc_action_length += length
            acc_action_confidence *= confidence
        
        # add the current action
        acc_action_prob *= action[1]
        acc_action_length += action[2]
        acc_action_confidence *= action_confidence

        # normalize the action_prob
        acc_action_prob = acc_action_prob ** (1 / acc_action_length)

        return acc_action_prob ** self.reward_alpha * acc_action_confidence ** (1 - self.reward_alpha), \
            {
                "acc_action_prob": acc_action_prob, 
                "cur_action_prob": action[1]**(1/action[2]),
            }
                                            
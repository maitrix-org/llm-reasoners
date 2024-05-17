import io
import re
from typing import TypedDict, Optional

import numpy as np

from world_model import GSM8kState, GSM8kAction, GSM8kPrompt
from reasoners import SearchConfig, LanguageModel



class GSM8kConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=1,
                 temperature=0.8,
                 reward_alpha=0,
                 reward_confidence_default=0.8,
                 depth_limit=5) -> None:
        super().__init__()
        self.base_model = base_model
        self.example = ''
        self.prompt: GSM8kPrompt = prompt
        self.batch_size = batch_size
        self.temperature = temperature
        self.n_actions = n_actions
        self.depth_limit = depth_limit
        self.reward_alpha = reward_alpha
        self.reward_confidence_default = reward_confidence_default
        self.overall_question: Optional[str] = None

    def update_example(self, example: str, prompt: dict) -> None:
        super().update_example(example, prompt)
        self.overall_prompt = self.prompt["overall"].replace("{QUESTION}", self.example)
        output = self.base_model.generate([self.overall_prompt],
                                          hide_input=True,
                                          do_sample=False,
                                          temperature=0,
                                          eos_token_id=["\",", "\"."]).text[0]  # [613, 1642] eos_token_id = ["\",", "\"."] 
            
        self.overall_question = output.split('"')[1]

    def get_actions(self, state: GSM8kState, ) -> list[GSM8kAction]:
        
        # "!" is a mark for terminal node
        # it won't be sent to LLMs
        # just for our convenience to judge whether it's terminal

        if len(state) > 0 and state[-1].sub_question.endswith('".'):
            return [' "' + self.overall_question + '"!']

        # if almost reach depth limit
        if len(state) >= self.depth_limit - 1:
            return [' "' + self.overall_question + '"!']

        with io.StringIO() as f:
            f.write(self.prompt["decomposition"]
                    .replace("{QUESTION}", self.example)
                    .replace("{LAST_QUESTION}",
                            self.overall_question))
            f.write("".join([q for q, a, _ in state]))
            model_input = f.getvalue()


        n_actions = self.n_actions
        temperature = self.temperature
        raw_outputs = []
        for idx in range(0, n_actions, self.batch_size):
            n_samples = min(n_actions - idx, self.batch_size)
            # breakpoint()
            raw_outputs += self.base_model.generate([model_input] * n_samples,
                                                hide_input=True,
                                                do_sample=True,
                                                temperature=temperature,
                                                eos_token_id=["\",", "\".", "\n" ]).text  # [613, 1642, 13] ["\",", "\".", "\n" ] 
        outputs = []
        for o in raw_outputs:
            if o.endswith("\n"):
                print("Warning: output ends with newline. Fixed it temporarily by adding a dot.")
                print("Output:", o)
                o = o[:-1] + '.'
            if not (o.endswith('".') or o.endswith('",')):
                print("Warning: output does not end with quote mark, this may cause unexpected behavior")
                print("Output:", o)
                #outputs.append(o)
                continue
            elif not o.startswith(' "'):
                print("Warning: output does not start with quote mark, this may cause unexpected behavior")
                print("Output:", o)
                #outputs.append(o)
                continue
            else:
                outputs.append(o)
                # keep the quote mark and "," / "." so that we can know whether it's terminal

        # set does not guarantee order, but dict does guarantee
        # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
        outputs = list(dict.fromkeys(outputs))
        return outputs

    def fast_reward(self, state: GSM8kState, action: GSM8kAction) -> tuple[float, dict]:
        return 1, {'r_useful': 1}

    def calculate_reward(self, r_useful, r_conf=None):
        if r_conf is None:
            r_conf = self.reward_confidence_default
        return r_useful ** self.reward_alpha * r_conf ** (1 - self.reward_alpha), {'r_useful': r_useful,
                                                                                   'r_conf': r_conf}

    def reward(self, state: GSM8kState, action: GSM8kAction,
               r_useful: float = None,
               confidence: float = None) -> tuple[float, dict]:
        assert r_useful is not None, "useful_reward is required to calculate reward in this search config, consider passing it in fast_reward"
        assert confidence is not None, "confidence is required to calculate reward in this search config, consider passing it in world model's step"
        return self.calculate_reward(r_useful, confidence)

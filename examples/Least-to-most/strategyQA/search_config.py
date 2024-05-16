import io
from typing import Tuple
from prompt import decompose_prompt
from world_model import StrategyQAAction, StrategyQAState
from reasoners import SearchConfig, LanguageModel
from utils import extract_subquestions
import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))

class StrategyQAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 temperature=0.7,
                 depth_limit=10,
                 self_consistency_n=1,
                 force_terminating_on_depth_limit=True,
                 log_dir=None) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.depth_limit = depth_limit
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.self_consistency_n = self_consistency_n
        self.actions = []
        self.log_dir = log_dir

    def get_actions(self, state: StrategyQAState) -> list[StrategyQAAction]:
        # the first time, we need to query the model to get the subquestions
        if len(state) == 0 and len(self.actions) == 0:
            # query the model using the prompt
            with io.StringIO() as f:
                f.write(decompose_prompt.strip()+"\n\n")
                f.write(f"Q: {self.example}\nA: To answer the question \"{self.example}\", we need to know:")

                model_input = f.getvalue()
            
            subqs_lm_list = []
            for _ in range(self.self_consistency_n // 5):
                subqs_lm_sub_list = self.base_model.generate(
                    [model_input]*5,
                    max_new_tokens=256,
                    hide_input=False,
                    do_sample=True,
                    top_k=32000,
                    top_p=0.95,
                    temperature=self.temperature,
                    eos_token_id='\n', #"\n"
                    num_return_sequences=1
                ).text

                for subqs_lm in subqs_lm_sub_list:
                    subqs_lm_list.append(subqs_lm)

            subqs_lists = []
            for subqs_lm in subqs_lm_list:
                ## extract sub-questions from lm response
                subqs_list = extract_subquestions(subqs_lm)
                subqs_list = [subq[1:-1].strip('.').strip('"') for subq in subqs_list]
                ## remove duplicate subquestions
                subqs_list = list(dict.fromkeys(subqs_list))
                ## if less than 3 subquestions, just use the final question
                if len(subqs_list) < 2:
                    subqs_list = subqs_list[-1:]
                ## if more than 5 subquestions, keep 5, add final question
                if len(subqs_list) > 5:
                    subqs_list = subqs_list[:5]

                # before adding the final question, we need to check if it is already in the list
                if self.example in subqs_list:
                    # get the all the indexes of the final question
                    indexes = [i for i, x in enumerate(subqs_list) if x == self.example]
                    # remove all the indexes
                    subqs_list = [i for j, i in enumerate(subqs_list) if j not in indexes]

                # add the final question
                subqs_list.append(self.example)

                subqs_lists.append(subqs_list)

                if local_rank == 0:
                    with open(self.log_dir+"/log.txt", "a") as f:
                        print(f"Subquestions: {subqs_list}", file=f)
            
            # set the actions
            self.actions = subqs_lists
        

        action = self.actions[0].pop(0)

        # pop the first action and del it from actions
        if len(self.actions[0]) == 0:
            self.actions = self.actions[1:]

        return [action]


    def fast_reward(self, state: StrategyQAState, action: StrategyQAAction) -> Tuple[float, dict]:
        return 0, {}
    
    def reward(self, state: StrategyQAState, action: StrategyQAAction) -> Tuple[float, dict]:
        return 0, {}



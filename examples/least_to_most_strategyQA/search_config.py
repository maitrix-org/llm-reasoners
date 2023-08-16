import io
from typing import Tuple
from prompt import decompose_prompt
from world_model import StrategyQAAction, StrategyQAState
from reasoners import SearchConfig, LanguageModel
from utils import extract_subquestions

class StrategyQAConfig(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 temperature=0.7,
                 depth_limit=10,
                 force_terminating_on_depth_limit=True) -> None:
        super().__init__()
        self.base_model = base_model
        self.temperature = temperature
        self.depth_limit = depth_limit
        self.force_terminating_on_depth_limit = force_terminating_on_depth_limit
        self.actions = []

    def get_actions(self, state: StrategyQAState) -> list[StrategyQAAction]:
        # the first
        if len(self.actions) == 0 and len(state) == 0:
            # query the model using the prompt
            with io.StringIO() as f:
                f.write(decompose_prompt.strip()+"\n\n")
                f.write(f"Q: {self.example}\nA:")

                model_input = f.getvalue()
            
        
            # make sure we have subquestions
            while len(self.actions) == 0:
            
                output = self.base_model.generate(
                    [model_input],
                    max_new_tokens=256,
                    hide_input=True,
                    do_sample=True,
                    temperature=self.temperature,
                    eos_token_id='\n'
                ).text[0].strip()

                # parse the output
                sub_questions = extract_subquestions(output)

                # set the actions
                self.actions = sub_questions
        
            return [self.actions[0]]

        else:
            if len(state) < len(self.actions):
                return [self.actions[len(state)]]
            else:
                # the last, we should return the example and set actions to empty
                self.actions = []
                return [self.example]


    def fast_reward(self, state: StrategyQAState, action: StrategyQAAction) -> Tuple[float, dict]:
        return 0, {}
    
    def reward(self, state: StrategyQAState, action: StrategyQAAction) -> Tuple[float, dict]:
        return 0, {}



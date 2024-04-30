import io
from typing import Tuple
from prompt import initial_instruction
from world_model import StructChemAction, StructChemState
from reasoners import SearchConfig, LanguageModel
# from utils import extract_subquestions
from utils import extract_formulae_reasoning
import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))

class StructChemConfigF(SearchConfig):
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

    def get_actions(self, state: StructChemState) -> list[StructChemAction]:
        # the first time, we need to query the model to get the initial overall generation
        if len(state) == 0 and len(self.actions) == 0:
            self.actions = [self.example.split("<concatenate>")[1],]
        else:
            self.actions = [state[-1][0],]
        action = self.actions.pop(0)

        # pop the first action and del it from actions
        if len(self.actions) == 0:
            self.actions = self.actions[1:]

        return [action]


    def fast_reward(self, state: StructChemState, action: StructChemAction) -> Tuple[float, dict]:
        return 0, {}
    
    def reward(self, state: StructChemState, action: StructChemAction) -> Tuple[float, dict]:
        return 0, {}


class StructChemConfigR(SearchConfig):
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

    def get_actions(self, state: StructChemState) -> list[StructChemAction]:
        # the first time, we need to query the model to get the initial overall generation
        if len(state) == 0 and len(self.actions) == 0:
            self.actions = [self.example.split("<concatenate>")[2],]
        else:
            self.actions = [state[-1][0],]
        action = self.actions.pop(0)

        # pop the first action and del it from actions
        if len(self.actions) == 0:
            self.actions = self.actions[1:]

        return [action]


    def fast_reward(self, state: StructChemState, action: StructChemAction) -> Tuple[float, dict]:
        return 0, {}
    
    def reward(self, state: StructChemState, action: StructChemAction) -> Tuple[float, dict]:
        return 0, {}



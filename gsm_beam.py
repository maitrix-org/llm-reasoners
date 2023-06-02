import os
# set the working directory to the root of the project
from rap import RAP, WorldModel, RewardModel
from rap.algorithms import BeamSearch
from rap.models import DummyLM

import random

class GSMWorld(WorldModel):
    def step(self, state, action):
        return state + action
    def is_terminal(self, state):
        return len(state) >= 10
    def get_actions(self, state):
        return ["a", "b", "c"]
    def update_question(self, question):
        pass

class GSMReward(RewardModel):
    def __init__(self) -> None:
        super().__init__()
        self.lm = DummyLM()

    def prior_reward(self, state, action):
        pass

    def reward(self, state, action, next_state):
        return random.random()

class GSM(RAP):
    def __init__(self) -> None:
        super().__init__(BeamSearch(GSMWorld(), GSMReward()))
    def preprocess(self, example):
        return ""
    def postprocess(self, example, output_state):
        return output_state

if __name__ == '__main__':
    gsm = GSM()
    print(gsm(""))
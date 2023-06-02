import os, sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from rap import WorldModel, RewardModel, AgentModel, RAP
from rap.algorithms import BeamSearch

class GSMWorldModel(WorldModel):
    """
    """
    def init_state(self) -> list[tuple[str, str]]:
        return []

    def step(self, state: list[tuple[str, str]], action: str) -> str:
        return state + [(action, "|ans|")]

    def is_terminal(self, state: list[tuple[str, str]]) -> bool:
        return False

class GSMRewardModel(RewardModel):
    def prior_reward(self, state: list[tuple[str, str]], action: str) -> float:
        return random.random()
    
class GSMAgentModel(AgentModel):
    def get_actions(self, state: str) -> list[str]:
        return [f"|question_{i}|" for i in range(10)]


if __name__ == "__main__":
    world = GSMWorldModel()
    reward = GSMRewardModel()
    agent = GSMAgentModel()
    search = BeamSearch(world, reward, agent)

    rap = RAP(agent, world, reward, search)
    rap.update_example("|Q|")
    print(rap("|Q|", output_trace=True))

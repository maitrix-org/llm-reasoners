from typing import Generic, TypeVar, Any, overload
from typing_extensions import Literal
from abc import ABC, abstractmethod

S = TypeVar("S")
A = TypeVar("A")

class WorldModel(ABC, Generic[S, A]):
    @abstractmethod
    def step(self, state: S, action: A) -> S:
        return NotImplemented
    
    @abstractmethod
    def is_terminal(self, state: S) -> bool:
        return NotImplemented
    
    @abstractmethod
    def get_actions(self, state: S) -> list[A]:
        return NotImplemented

class RewardModel(ABC, Generic[S, A]):
    # shibo: Maybe if an user wants to use both prior_reward and reward, he/she should instantiate two RewardModel objects?
    @abstractmethod
    def prior_reward(self, state: S, action: A) -> float:
        return NotImplemented

    def reward(self, state: S, action: A, next_state: S) -> float:
        return self.prior_reward(state, action)


class SearchAlgorithm(ABC, Generic[S]):
    # shibo: should allow user to specify the return_dict, e.g. reward, action, state, etc.
    @overload
    def __call__(self, init_state: S, output_trace: Literal[False] = ...) -> S: ...

    @overload
    def __call__(self, init_state: S, output_trace: Literal[True]) -> list[tuple[A | None, S]]: ...

    @abstractmethod
    def __call__(self, init_state: S, output_trace: bool = False) -> S | list[tuple[A | None, S]]:
        return NotImplemented


class LanguageModel(ABC):
    def query(self, query: str) -> float:
        return NotImplemented


class Agent(ABC, Generic[S, A]):
    def __init__(self, algorithm: SearchAlgorithm[S]):
        self.algorithm = algorithm

    def rap():


    @abstractmethod
    def preprocess(self, example) -> S:
        return NotImplemented

    @abstractmethod
    def postprocess(self, example, output_state: S) -> Any:
        return NotImplemented

    def __call__(self, example) -> Any:
        init_state = self.preprocess(example)
        output_state = self.algorithm(init_state)
        output = self.postprocess(example, output_state)
        return output

if __name__ == '__main__':
    class MyWorldModel(WorldModel[str, str]): ...
    class MyRewardModel(RewardModel[str, str]): ...
    from .algorithms.mcts import MCTS
    world_model = MyWorldModel()
    reward_model = MyRewardModel()
    mcts = MCTS(world_model, reward_model)
    class MyRAP(RAP[str, str]): ...
    rap = MyRAP(mcts)
    rap("123")
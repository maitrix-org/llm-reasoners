from typing import Generic, TypeVar, Any, overload
from typing_extensions import Literal
from abc import ABC, abstractmethod

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")

class WorldModel(ABC, Generic[State, Action]):
    @abstractmethod
    def init_state(self) -> State:
        return NotImplemented
    
    @abstractmethod
    def step(self, state: State, action: Action) -> State:
        return NotImplemented
    
    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        return NotImplemented
    
    def update_example(self, example: Example) -> None:
        self.example = example

class RewardModel(ABC, Generic[State, Action]):

    @abstractmethod
    def prior_reward(self, state: State, action: Action) -> float:
        return NotImplemented

    def reward(self, state: State, action: Action, next_state: State = None, next_states: list[State] = None) -> float:
        return self.prior_reward(state, action)

    def update_example(self, example: Example) -> None:
        self.example = example

class SearchAlgorithm(ABC, Generic[State]):
    # shibo: maybe allow user to specify the return_dict, e.g. reward, action, state, etc.
    @overload
    def __call__(self, init_state: State, output_trace: Literal[False] = ...) -> State: ...

    @overload
    def __call__(self, init_state: State, output_trace: Literal[True]) -> list[tuple[Action | None, State]]: ...

    @abstractmethod
    def __call__(self, init_state: State, output_trace: bool = False) -> State | list[tuple[Action | None, State]]:
        return NotImplemented
    
    def update_example(self, example: Example) -> None:
        self.example = example


class LanguageModel(ABC):
    def query(self, query: str) -> str:
        return NotImplemented


class AgentModel(ABC, Generic[State, Action]):
    @abstractmethod
    def get_actions(self, state: State) -> list[Action]:
        pass
    
    def update_example(self, example: Example) -> None:
        return NotImplemented


class RAP():
    def __init__(self, agent, world, reward, search) -> None:
        self.agent = agent
        self.world = world
        self.reward = reward
        self.search = search
    
    def __call__(self, example, output_trace=False) -> Any:
        self.world.update_example(example)
        return self.search(output_trace)

    def update_example(self, example) -> None:
        self.world.update_example(example)
        self.search.update_example(example)
        self.agent.update_example(example)
        self.reward.update_example(example)
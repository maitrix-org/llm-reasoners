from typing import Generic, TypeVar
from abc import ABC, abstractmethod

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")

class LanguageModel(ABC):
    @abstractmethod
    def __call__(self, inputs: list[str], **kwargs) -> dict: ...

class WorldModel(ABC, Generic[State, Action]):
    @abstractmethod
    def init_state(self) -> State: ...
    @abstractmethod
    def step(self, state: State, action: Action) -> State: ...
    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...
    def update_example(self, example: Example) -> None:
        self.example = example

class SearchConfig(ABC, Generic[State, Action]):
    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...
    @abstractmethod
    def prior_policy(self, state: State, action: Action) -> float: ...
    @abstractmethod
    def reward(self, state, action, **kwargs) -> float: ...
    def update_example(self, example: Example) -> None:
        self.example = example

class SearchAlgorithm(ABC):
    @abstractmethod
    def __call__(self, example, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> dict: ...

class RAPAgent(ABC):
    def __init__(self, world_model, search_config, search_algo) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example, **kwargs) -> dict:
        self.world_model.update_example(example)
        self.search_config.update_example(example)
        return self.search_algo(self.world_model, self.search_config, **kwargs)
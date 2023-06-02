from abc import ABC, abstractmethod


class WorldModel(ABC):
    @abstractmethod
    def init_state(self): ...

    @abstractmethod
    def step(self, state, action): ...

    @abstractmethod
    def is_terminal(self, state) -> bool: ...


class SearchPolicy(ABC):
    @abstractmethod
    def get_actions(self, state): ...

    @abstractmethod
    def prior_reward(self, state, action) -> float: ...

    @abstractmethod
    def reward(self, state, action, next_state=None) -> float: ...


class SearchAlgorithm(ABC):
    @abstractmethod
    def __call__(self, example, world_model, search_policy, output_trace: bool = False, **kwargs): ...


class RAPAgent(ABC):
    def __init__(self, world_model, search_policy, search_algo) -> None:
        self.world_model = world_model
        self.search_policy = search_policy
        self.search_algo = search_algo

    def __call__(self, example, output_trace=False):
        return self.search_algo(example, self.world_model, self.search_policy, output_trace=output_trace)

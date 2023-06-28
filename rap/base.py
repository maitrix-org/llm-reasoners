from typing import Generic, TypeVar, Union, NamedTuple, Protocol
from abc import ABC, abstractmethod

import numpy as np

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]


class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: np.ndarray


class LanguageModel(ABC):
    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_gen_len: int = ...,
                 temperature: float = ...,
                 top_p: float = ...,
                 end_token: str = ...,
                 hide_input: bool = ...,
                 **kwargs) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_gen_len: Maximum length of generated text.
        :param temperature: Temperature for sampling. 0 for greedy decoding.
        :param top_p: Top-p for sampling.
        :param end_token: Token id for end of sentence.
        :param hide_input: If set true, decode only the generated part.
        :return: A dict of output, dict["text"]: str ; dict["log_prob"]: np.ndarray
        """
        ...

    @abstractmethod
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :return:
        """
        ...

    @abstractmethod
    def get_ll(self,
               prefix: str,
               contents: list[str],
               **kwargs) -> np.ndarray:
        """Get the log likelihood of the contents given the prefix.

        :param prefix: The prefix to be excluded from the log likelihood.
        :param contents: The contents to evaluate (must include the prefix).
        """
        ...


class WorldModel(ABC, Generic[State, Action]):
    def __init__(self) -> None:
        self.example = None

    @abstractmethod
    def init_state(self) -> State: ...

    @abstractmethod
    def step(self, state: State, action: Action) -> State: ...

    @abstractmethod
    def is_terminal(self, state: State) -> bool: ...

    def update_example(self, example: Example) -> None:
        self.example = example


class SearchConfig(ABC, Generic[State, Action]):
    def __init__(self) -> None:
        self.example = None

    @abstractmethod
    def get_actions(self, state: State) -> list[Action]: ...

    @abstractmethod
    def fast_reward(self, state: State, action: Action) -> float: ...

    @abstractmethod
    def reward(self, state, action, **kwargs) -> float: ...

    def update_example(self, example: Example) -> None:
        self.example = example


class HasTerminalStateAndTrace(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> HasTerminalStateAndTrace: ...


class RAPAgent(ABC, Generic[State, Action, Example]):
    def __init__(self,
                 world_model: WorldModel[State, Action],
                 search_config: SearchConfig[State, Action],
                 search_algo: SearchAlgorithm) -> None:
        self.world_model = world_model
        self.search_config = search_config
        self.search_algo = search_algo

    def __call__(self, example: Example, **kwargs) -> HasTerminalStateAndTrace[State]:
        self.world_model.update_example(example)
        self.search_config.update_example(example)
        return self.search_algo(self.world_model, self.search_config, **kwargs)

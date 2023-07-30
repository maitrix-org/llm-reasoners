from typing import Generic, TypeVar, Union, NamedTuple, Protocol, Optional
from abc import ABC, abstractmethod

import numpy as np
from transformers import StoppingCriteriaList

State = TypeVar("State")
Action = TypeVar("Action")
Example = TypeVar("Example")
Trace = tuple[list[State], list[Action]]


class GenerateOutput(NamedTuple):
    text: list[str]
    log_prob: list[np.ndarray] = None


class LanguageModel(ABC):
    @abstractmethod
    def generate(self,
                 inputs: list[str],
                 max_length: Optional[int] = None,
                 max_new_tokens: Optional[int] = None,
                 do_sample: bool = False,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 1.0,
                 num_return_sequences: int = 1,
                 eos_token_id: Union[None, str, int, list[str, int]] = None,
                 hide_input: bool = True,
                 output_log_probs: bool = False,
                 stopping_criteria: Optional[StoppingCriteriaList] = None,
                 **kwargs) -> GenerateOutput:
        """Generate text from a list of prompts.

        :param inputs: List of prompts.
        :param max_length: Maximum length of the total output (input + generated).
        :param max_new_tokens: Maximum length of generated tokens. Override max_length.
        :param do_sample: If False, do greedy decoding.
        :param temperature: Temperature for sampling.
        :param top_k: Top-k for sampling.
        :param top_p: Top-p for sampling.
        :param num_return_sequences:
        :param eos_token_id: Token id for end of sentence. Passed *str* will be translated into token_id.
                             Passed *list* will be treated as multiple possible tokens ending the generation.
        :param hide_input: If set true, decode only the generated part.
        :param output_log_probs: If set true, also output the log_probs of each generated token
        :param stopping_criteria:
        """
        ...

    @abstractmethod
    def get_next_token_logits(self,
                              prompt: Union[str, list[str]],
                              candidates: Union[list[str], list[list[str]]],
                              postprocess: Optional[str] = None,
                              **kwargs) -> list[np.ndarray]:
        """ TODO: doc

        :param prompt:
        :param candidates:
        :param postprocess: optional, can be 'log_softmax' or 'softmax'. Apply the corresponding function to logits before returning
        :return:
        """
        ...

    @abstractmethod
    def get_loglikelihood(self,
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
    def fast_reward(self, state: State, action: Action) -> tuple[float, dict]:
        return 0, {}

    @abstractmethod
    def reward(self, state, action, **kwargs) -> tuple[float, dict]: ...

    def update_example(self, example: Example) -> None:
        self.example = example


class HasTerminalStateAndTrace(Protocol[State]):
    terminal_state: State
    trace: Trace


class SearchAlgorithm(ABC):
    def __init__(self, **kwargs): ...

    @abstractmethod
    def __call__(self, world_model: WorldModel, search_config: SearchConfig, **kwargs) -> HasTerminalStateAndTrace: ...


class Reasoner(ABC, Generic[State, Action, Example]):
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

import copy
import dataclasses
import re
from typing import Optional
from reasoners import WorldModel, LanguageModel


@dataclasses.dataclass
class Game24State:
    input: str
    current: str
    history: list[str]
    output: Optional[str] = None


Game24Action = str


class Game24WorldModel(WorldModel[Game24State, Game24Action]):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_confidence=8,
                 batch_size=2, ) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence

    def init_state(self) -> Game24State:
        return Game24State(self.example, self.example, [])

    def step(self, state: Game24State, action: Game24Action) -> tuple[Game24State, dict]:
        state = copy.deepcopy(state)
        if 'Answer' in action:
            match = re.match(r'Answer: (.*)', action)
            state.output = match[1] if match is not None else ''
        else:
            match = re.match(r'.*\(left: (.*)\)', action)
            state.current = match[1] if match is not None else ''
            state.history.append(action)
        return state, {'next_state': state}

    def is_terminal(self, state: Game24State) -> bool:
        return state.output is not None

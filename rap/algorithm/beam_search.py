from typing import Generic
from collections import defaultdict
from ..rap import SearchAlgorithm, WorldModel, RAPAgent, SearchConfig, State, Action

class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, beam_size: int, max_depth: int):
        self.beam_size = beam_size
        self.max_depth = max_depth

    def __call__(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], output_trace: bool = False) -> State | list[tuple[Action, State]]:
        init_state = world.init_state()
        cur_beam = [([(None, init_state)], 0)]  # (trace, reward)
        terminal_beam = []
        for _ in range(self.max_depth):
            new_beam = []
            for trace, reward in cur_beam:
                state = trace[-1][-1]
                if world.is_terminal(state) or len(trace) == self.max_depth:
                    terminal_beam.append((trace, reward))
                else:
                    for action in config.get_actions(state):
                        next_state = world.step(state, action)
                        next_reward = config.reward(state, action, next_state=next_state)
                        new_beam.append((trace + [(action, next_state)], reward + next_reward))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            cur_beam = new_beam[:self.beam_size]

        if output_trace:
            return terminal_beam[0][0]
        else:
            return terminal_beam[0][0][-1][-1]
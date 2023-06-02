from typing import Generic
from collections import defaultdict
from ..rap import SearchAlgorithm, WorldModel, RewardModel, S, A

class BeamSearch(SearchAlgorithm, Generic[S, A]):
    def __init__(self, world_model: WorldModel[S, A], reward_model: RewardModel[S, A], beam_size=10, max_depth=5):
        self.world_model = world_model
        self.reward_model = reward_model
        self.beam_size = beam_size
        self.max_depth = max_depth

    def __call__(self, init_state: S, output_trace: bool = False) -> S | list[tuple[A, S]]:
        cur_beam = [([(None, init_state)], 0)]  # (trace, reward)
        terminal_beam = []
        for _ in range(self.max_depth):
            new_beam = []
            for trace, reward in cur_beam:
                state = trace[-1][-1]
                if self.world_model.is_terminal(state):
                    terminal_beam.append((trace, reward))
                else:
                    for action in self.world_model.get_actions(state):
                        next_state = self.world_model.step(state, action)
                        next_reward = self.reward_model.reward(state, action, next_state)
                        new_beam.append((trace + [(action, next_state)], reward + next_reward))
            new_beam.sort(key=lambda x: x[1], reverse=True)
            cur_beam = new_beam[:self.beam_size]

        if output_trace:
            return terminal_beam[0][0]
        else:
            return terminal_beam[0][0][-1][-1]
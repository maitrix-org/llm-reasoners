from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, RAPAgent, SearchConfig, State, Action


class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, beam_size: int, max_depth: int):
        self.beam_size = beam_size
        self.max_depth = max_depth

    def __call__(self, world: WorldModel[State, Action], config: SearchConfig[State, Action],
                 output_trace: bool = False):
        init_state = world.init_state()
        cur_beam = [([(None, init_state)], 0)]  # (trace, reward)
        terminal_beam = []
        for i in range(self.max_depth):
            print(f"\n----new step {i}----")
            new_beam = []
            rewards = []
            new_actions = []
            local_cache = set()
            for trace, reward in cur_beam:
                state = trace[-1][-1]
                if world.is_terminal(state):
                    terminal_beam.append((trace, reward))
                else:
                    new_actions += [(state, action) for action in config.get_actions(state)]
                    # for action in config.get_actions(state):
                    #     next_state = world.step(state, action)
                    #     next_reward = config.reward(state, action, next_state=next_state)
                    #     new_beam.append((trace + [(action, next_state)], next_reward))
                    #     # customize reward inside reward function (e.g. the reward of the trace)
                    #     # new_beam.append((trace + [(action, next_state)], reward + next_reward))
            for state, action in new_actions:
                #### remove duplicates
                if action not in local_cache:
                    next_state = world.step(state, action)
                    next_reward = config.reward(state, action, next_state=next_state)
                    local_cache.add(action)
                else:
                    flatten_y = action.replace("\n", " -> ")
                    print(f"duplicate step: {flatten_y}")
                    next_reward = 0
                new_beam.append((trace + [(action, next_state)], next_reward))
                rewards.append(next_reward)
            ids = list(range(len(new_beam)))
            select_ids = sorted(ids, key=lambda x: rewards[x], reverse=True)[:self.beam_size]
            cur_beam = [new_beam[select_id] for select_id in select_ids]
            # new_beam.sort(key=lambda x: x[1], reverse=True)
            # cur_beam = new_beam[:self.beam_size]
            print(f"----choices: {[(x[0][-1][-1].sub_answer) for x in cur_beam]}----")
            print(f"----values: {[(x[-1]) for x in cur_beam]}----")

        terminal_beam += cur_beam
        if output_trace:
            return terminal_beam#[0][0]
        else:
            return terminal_beam#[0][0][-1][-1]
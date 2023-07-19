from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, RAPAgent, SearchConfig, State, Action
from typing import NamedTuple, List, Tuple

class BeamSearchResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: List[Tuple[Action, State]]


class BeamSearch(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, beam_size: int, max_depth: int):
        self.beam_size = beam_size
        self.max_depth = max_depth

    def __call__(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], action_dedup: bool=False, return_beam: bool=False, early_terminate: bool=True, reward_strategy: str='last_iter'):
        init_state = world.init_state()
        cur_beam = [([(None, init_state)], 0)]  # (trace, reward)
        terminal_beam = []
        for i in range(self.max_depth):
            # print(f"\n----new step {i}----")
            new_beam = []
            rewards = []
            new_actions = []
            local_cache = set()
            for trace, reward in cur_beam:
                state = trace[-1][-1]
                if early_terminate and (world.is_terminal(state) or (len(trace) == self.max_depth)):
                    terminal_beam.append((trace, reward))
                else:
                    if action_dedup:
                        new_actions += [(state, action, reward) for action in config.get_actions(state)]
                    else:
                        for action in config.get_actions(state):
                            next_state = world.step(state, action)
                            next_reward = config.reward(state, action, next_state=next_state)
                            if reward_strategy == 'last_iter':
                                next_reward = next_reward
                            elif reward_strategy == 'cumulate':
                                next_reward += reward
                            new_beam.append((trace + [(action, next_state)], next_reward))
            if action_dedup:
                for state, action, reward in new_actions:
                    #### remove duplicates
                    if action not in local_cache:
                        next_state = world.step(state, action)
                        next_reward = config.reward(state, action, next_state=next_state)
                        local_cache.add(action)
                    else:
                        # flatten_y = action.replace("\n", " -> ")
                        # print(f"duplicate step: {flatten_y}")
                        next_reward = 0
                    if reward_strategy == 'last_iter':
                        next_reward = next_reward
                    elif reward_strategy == 'cumulate':
                        next_reward += reward
                    new_beam.append((trace + [(action, next_state)], next_reward))
                    rewards.append(next_reward)
                ids = list(range(len(new_beam)))
                select_ids = sorted(ids, key=lambda x: rewards[x], reverse=True)[:self.beam_size]
                cur_beam = [new_beam[select_id] for select_id in select_ids]
                # print(f"----choices: {[(x[0][-1][-1].sub_answer) for x in cur_beam]}----")
                # print(f"----values: {[(x[-1]) for x in cur_beam]}----")
            else:
                new_beam.sort(key=lambda x: x[1], reverse=True)
                cur_beam = new_beam[:self.beam_size]

        if return_beam:
            terminal_beam += cur_beam
            return terminal_beam
        else:            
            best_result = terminal_beam[0]
            result = BeamSearchResult(
                terminal_state=best_result[0][-1][-1], 
                cum_reward=best_result[1], 
                trace=best_result[0]
                )

            return result
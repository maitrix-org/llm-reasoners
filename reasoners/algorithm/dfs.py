from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, Reasoner, SearchConfig, State, Action
from typing import NamedTuple, List, Tuple

class DFSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: List[Tuple[State]]


class DFS(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, 
                 total_states: int = 100, 
                 max_per_state: int = 3, 
                 depth: int = 5):
        self.max_per_state = max_per_state
        self.depth = depth # not used
        self.total_states = total_states
        self.terminals = [] ## final results 
        self.stat_cnt = 0

    def _reset(self):
        self.terminals = []
        self.stat_cnt = 0

    def __call__(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], best_state: bool=True):
        init_state = world.init_state()
        self._reset()
        self.dfs(world, config, init_state, best_state=best_state)
        return self.terminals

    def dfs(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], cur_state: State, best_state: bool=True):
        self.stat_cnt += 1
        if world.is_terminal(cur_state): # len(info)==0 #change
            self.terminals.append(cur_state) #change
            #return # change
        # get candidate actions (list, (action, score) or action)
        new_actions = config.get_actions(cur_state) # [(candidate, candidate score)]
        print(f'state id: {self.stat_cnt}, state: {cur_state} new actions {len(new_actions)}')
        if len(new_actions) == 0: 
            print('terminal return: no new action')
            return 
        ## sort possible actions by score
        if best_state:
            new_actions = sorted(new_actions, key=lambda x: x[1], reverse=True)

        # try each candidate
        cnt_per_state = 0
        for action in new_actions:
            print('------------- world.step ---------------')
            new_state = world.step(cur_state, action)
            print('------------- world.step Done---------------')
            # check all existing state/depth/branch constraints
            print(f'check condition:')
            print(f'{self.stat_cnt} {self.total_states} {self.stat_cnt < self.total_states}')
            print(f'{cnt_per_state} {self.max_per_state} {cnt_per_state < self.max_per_state}')
            print(f'{config.search_condition(cur_state)} {cur_state}')
            print()
            if self.stat_cnt < self.total_states and config.search_condition(cur_state):
                cnt_per_state += 1
                print(f'dfs_branch cnt: {cnt_per_state}')
                if cnt_per_state > self.max_per_state: 
                    print(f'reach max_per_state {self.max_per_state}: break')
                    break

                ## other state constraints
                if config.state_condition(new_state):  # only continue if the current status is possible
                    neibor_info = self.dfs(world, config, new_state, best_state)
        return
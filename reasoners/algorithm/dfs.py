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
                 depth: int = 10):
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

    def dfs(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], cur_state: State, best_state: bool=True, early_terminate: bool=True):
        ## if it's terminal state
        if world.is_terminal(cur_state): # if is terminal
            self.terminals.append(cur_state) #change
        if not config.state_condition(cur_state):  # only continue if the current status is possible
            return

        # get candidate actions (list, (action, score) or action)
        new_actions = config.get_actions(cur_state) # [(candidate, candidate score)]
        print(f'new actions: {sorted(new_actions, key=lambda x: x[1], reverse=True)}')
        if len(new_actions) == 0: 
            print('terminal return: no new action')
            return 
        ## sort possible actions by score
        if best_state:
            new_actions = sorted(new_actions, key=lambda x: x[1], reverse=True)

        # try each candidate
        cnt_per_state = 0
        for action in new_actions:
            new_state = world.step(cur_state, action)
            if self.stat_cnt < self.total_states and config.search_condition(new_state):
                cnt_per_state += 1
                if cnt_per_state > self.max_per_state: 
                    print(f'reach max_per_state {self.max_per_state}: break')
                    break
                print(f'accepted new_state: {self.stat_cnt}')
                self.stat_cnt += 1
                new_env, new_state_actions, new_info = new_state
                print(new_state_actions)
                print(new_env.render_board())
                print(new_info['info'])
                print(new_info['count'])
                print(f'dfs_branch cnt: {cnt_per_state}')

                neibor_info = self.dfs(world, config, new_state, best_state)
        return
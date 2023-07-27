from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, Reasoner, SearchConfig, State, Action
from typing import NamedTuple, List, Tuple

class DFSResult(NamedTuple):
    terminal_state: State
    cum_reward: float
    trace: List[Tuple[State]]


class DFS(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, total_states: int, max_per_state: int, depth: int):
        self.max_per_state = max_per_state
        self.depth = depth
        self.total_states = total_states
        self.terminals = [] ## final results
        self.stat_cnt = 0

    def __call__(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], best_state: bool=True):
        init_state = world.init_state()
        self.dfs(world, config, init_state, best_state=best_state)
        return self.terminals

    def dfs(self, world: WorldModel[State, Action], config: SearchConfig[State, Action], cur_state: State, best_state: bool=True):
        self.stat_cnt += 1
        if world.is_terminal(cur_state):
            self.terminals.append(cur_state)
            return 
        # get candidate actions (list, (action, score) or action)
        new_actions = config.get_actions(cur_state)
        print(f'new actions {new_actions}')
        if len(new_actions) == 0: 
            return 
        ## sort possible actions by score
        if best_state:
            new_actions = sorted(new_actions, key=lambda x: x[1], reverse=True)


        # try each candidate
        cnt_per_state = 0
        for action in new_actions:
            # check all existing state/depth/branch constraints
            if self.stat_cnt < self.total_states and cnt_per_state < self.max_per_state and config.search_condition(cur_state):
                cnt_per_state += 1
                #### new state
                new_state = world.step(cur_state, action)

                # infos.append(info)
                ## other state constraints
                if config.state_condition(new_state):  # only continue if the current status is possible
                    neibor_info = self.dfs(world, config, new_state, best_state)
                # actions.pop()
            # env.reset(env.idx, board=board.copy(), status=status.copy(), steps=steps)
        return
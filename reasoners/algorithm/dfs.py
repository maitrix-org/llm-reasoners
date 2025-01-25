from typing import Generic
from collections import defaultdict
from .. import SearchAlgorithm, WorldModel, Reasoner, SearchConfig, State, Action
from typing import NamedTuple, List, Tuple
import itertools
from typing import Generic, Optional, NamedTuple, Callable, Hashable
import copy
import multiprocessing.dummy as mp_dummy


class DFSNode:
    id_iter = itertools.count()

    @classmethod
    def reset_id(cls):
        cls.id_iter = itertools.count()

    def __init__(self, state: Optional[State], action: Optional[Action], parent: "Optional[DFSNode]" = None, fast_reward: float = 0., fast_reward_details=None, is_terminal: bool = False) -> None:
        
        self.id = next(DFSNode.id_iter)
        if fast_reward_details is None:
            fast_reward_details = {}
        self.reward_details = {}
        self.cum_rewards: list[float] = []
        self.fast_reward = self.reward = fast_reward
        self.fast_reward_details = fast_reward_details
        self.is_terminal = is_terminal
        self.action = action
        self.state = state
        self.parent = parent
        self.children: 'Optional[list[DFSNode]]' = []
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1

    def add_child(self, child: 'DFSNode'):
        self.children.append(child)
    
    def get_trace(self) -> List[Tuple[Action, State, float]]:
        """ Returns the sequence of actions and states from the root to the current node """
        node, path = self, []
        while node is not None:
            path.append((node.action, node.state, node.reward))
            node = node.parent
        # Reverse the path to get actions and states in order
        path = path[::-1]
        return path

class DFSResult(NamedTuple):
    terminal_state: State
    cum_rewards: float
    tree_state: DFSNode
    terminal_nodes: List[DFSNode]

class DFS(SearchAlgorithm, Generic[State, Action]):
    """
    config.fast_reward is the prior to decide the order of exporation
    config.reward is the actual reward that decides the final result
    """

    def __init__(self, 
                 total_states: int = 100, 
                 max_per_state: int = 3, 
                 depth: int = 10,
                 prior: bool = True,
                 max_terminal_nodes: int = 10,
                 return_if_single_first_action: bool = False,
                 use_mp: bool = False):
        self.max_per_state = max_per_state
        self.depth = depth # not used
        self.total_states = total_states
        self.terminals = [] ## final results 
        self.stat_cnt = 0
        self.prior = prior # use fast_reward as prior score
        self.max_terminal_nodes = max_terminal_nodes
        self.return_if_single_first_action = return_if_single_first_action
        self.use_mp = use_mp

    def _reset(self):
        self.terminals = []
        self.stat_cnt = 0

    def __call__(self, world: WorldModel, config: SearchConfig):
        # reset id
        DFSNode.reset_id()
        
        init_state = world.init_state()
        self._reset()
        init_node = DFSNode(state=init_state, action=None, parent=None, fast_reward=0., fast_reward_details=None, is_terminal=False)
        self.dfs(world, config, init_node)
        sorted_terminals = sorted(self.terminals, key=lambda x: sum(x.cum_rewards), reverse=True)
        result = DFSResult(terminal_state=sorted_terminals[0].state, cum_rewards=sorted_terminals[0].cum_rewards, tree_state=init_node, terminal_nodes=sorted_terminals)
        return result

    def dfs(self, world: WorldModel, config: SearchConfig, cur_node: DFSNode):
    
        # Stop if max_terminal_nodes is reached
        if len(self.terminals) >= self.max_terminal_nodes:
            return
 
        ## if it's terminal state
        if world.is_terminal(cur_node.state) or cur_node.depth == self.depth:
            self.terminals.append(cur_node)  # change
            return

        cur_state = cur_node.state
        # get candidate actions (list, (action, score) or action)
        new_actions = config.get_actions(cur_state)
        if len(new_actions) == 0: 
            print('terminal return: no new action')
            return 
        
        # if only one action on the first layer, return
        if self.return_if_single_first_action and len(new_actions) == 1 and cur_node.depth == 0:
            print('terminal return: only one action no the first layer')
            new_node = DFSNode(state=None, action=new_actions[0], parent=cur_node, 
                               fast_reward=0, fast_reward_details=None, is_terminal=False)
            new_node.cum_rewards = cur_node.cum_rewards + [new_node.reward]
            cur_node.add_child(new_node)
            self.terminals.append(new_node)
            return
        
        ## sort possible actions by score
        if self.prior:
            actions_with_prior = [(a, config.fast_reward(cur_state, a)) for a in new_actions]
            new_actions = sorted(actions_with_prior, key=lambda x: x[1][0], reverse=True)
        else:
            new_actions = [(a, (0, {})) for a in new_actions]
        # try each candidate
        cnt_per_state = 0
        
        # with mp_dummy.Pool(processes=n_actions) as pool:
        #     # Use starmap to pass multiple arguments to the function
        #     arguments = [
        #         (
        #             self.obs_history,
        #             self.states + new_states,
        #             self.strategies + new_actions,
        #             self.explanations,
        #             self.actions,
        #             self.policy,
        #         )
        #     ] * n_actions
        #     sampled_actions = pool.starmap(sample_action, arguments)
        
        if not self.use_mp:
            for action in new_actions:
                action, (fast_reward, fast_reward_details) = action
                # new_state = world.step(cur_state, action)
                if self.stat_cnt < self.total_states:
                    cnt_per_state += 1
                    if cnt_per_state > self.max_per_state: 
                        print(f'reach max_per_state {self.max_per_state}: break')
                        break
                    self.stat_cnt += 1

                    new_state, aux = world.step(cur_state, action)

                    new_node = DFSNode(state=new_state, action=action, parent=cur_node, fast_reward=fast_reward, fast_reward_details=fast_reward_details, is_terminal=False)
                    new_node.reward, new_node.reward_details = config.reward(cur_state, action, **aux, **fast_reward_details)
                    new_node.cum_rewards = cur_node.cum_rewards + [new_node.reward]

                    cur_node.add_child(new_node)
                    self.dfs(world, config, new_node)
        else:
            with mp_dummy.Pool(processes=self.max_per_state) as pool:
                arguments = [(world, config, cur_node, action) for action in new_actions[:self.max_per_state]]
                pool.starmap(self._dfs_mp, arguments)
        return
    
    def _dfs_mp(self, world, config, cur_node, action):
        action, (fast_reward, fast_reward_details) = action
        new_state, aux = world.step(cur_node.state, action)
        new_node = DFSNode(state=new_state, action=action, parent=cur_node, fast_reward=fast_reward, fast_reward_details=fast_reward_details, is_terminal=False)
        new_node.reward, new_node.reward_details = config.reward(cur_node.state, action, **aux, **fast_reward_details)
        new_node.cum_rewards = cur_node.cum_rewards + [new_node.reward]
        cur_node.add_child(new_node)
        
        self.dfs(world, config, new_node)
    

class CW_DFS(SearchAlgorithm, Generic[State, Action]):
    # specific to crosswords
    # please use the DFS class for general purpose
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

    def __call__(self, world: WorldModel, config: SearchConfig, best_state: bool=True):
        init_state = world.init_state()
        self._reset()
        self.dfs(world, config, init_state, best_state=best_state)
        return self.terminals

    def dfs(self, world: WorldModel, config: SearchConfig, cur_state: State, best_state: bool=True, early_terminate: bool=True):
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
    
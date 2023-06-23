from typing import Generic

from ..rap import SearchAlgorithm, WorldModel, RAPAgent, SearchConfig, State, Action

class MCTSNode(Generic[State, Action]):
    def __init__(self, prior=0):
        self.sum_Q = 0
        self.N = 0
        # prior is an approximation of reward
        self.prior = prior
        self.action: Action = None
        self.state: State = None
        self.parent: "MCTSNode" = None
    
    @property
    def Q(self):
        if self.N == 0:
            return self.prior
        else:
            return self.sum_Q / self.N

    @property
    def reward(self):
        if self._r0 < 0 or self._r1 < 0:
            return min(self._r0, self._r1)
        print("# in @property reward: r0, r1, aggr", self._r0, self._r1, self._r0 ** self._r_alpha * self._r1 ** (1 - self._r_alpha))
        
        return self._r0 ** self._r_alpha * self._r1 ** (1 - self._r_alpha)

class MCTS(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, output_trace: bool = False, output_aggregation: bool = False):
        # TODO: add parameters
        # alpha, max_depth, max_iter
        pass

    def mcts_search(self, init_state: State) -> tuple[MCTSNode, MCTSNode]:
        # TODO: generate init_state with world_model
        # TODO: implement MCTS
        raise NotImplementedError
    
    def iterate(self, node: MCTSNode):
        path = self._select(node)
        self._expand(path[-1])
        self._simulate(path)
        self._back_propagate(path)

    def _select(self, node: MCTSNode):
        path = []
        while True:
            path.append(node)
            if node not in self.children or node.is_terminal:
                return path
            for child in self.children[node]:
                if child not in self.children.keys():
                    path.append(child)
                    return path
            node = self._uct_select(node)
    
    def _back_propagate(self, path: list[MCTSNode], reward=0.):
        coeff = 1
        for node in reversed(path):
            reward = reward * self.discount + node.reward
            coeff = coeff * self.discount + 1
            if self.aggr_reward == 'mean':
                c_reward = reward / coeff
            else:
                c_reward = reward
            if node not in self.N:
                self.Q[node] = c_reward
            else:
                self.Q[node] += c_reward
            self.N[node] += 1
            self.M[node] = max(self.M[node], c_reward)

    def __call__(self, init_state: State, output_trace: bool = False):
        root, chosen_leaf = self.mcts_search(init_state=init_state)
        if output_trace:
            ret = []
            cur = chosen_leaf
            while cur is not None:
                ret.append((cur.action, cur.state))
                cur = cur.parent
            return reversed(cur)
        else:
            return chosen_leaf.state


'''
class MCTSAggregation(MCTS[State, Action]):
    def __call__(self, init_state: State, output_trace: bool = False) -> State | list[tuple[Action, State]]:
        # TODO: implement aggregate
        pass
'''
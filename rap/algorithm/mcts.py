from typing import Generic, Optional, NamedTuple

from ..rap import SearchAlgorithm, WorldModel, SearchConfig, State, Action


class MCTSNode(Generic[State, Action]):
    def __init__(self, prior: float = 0.):
        self.sum_Q = 0
        self.N = 0
        self.prior = self.reward = prior  # an estimation of the reward of the last step
        self.action: Optional[Action] = None  # the action of the last step
        self.state: State = None  # the current state
        self.parent: "Optional[MCTSNode]" = None  # None if root of the tree

    @property
    def Q(self):
        if self.N == 0:
            return self.prior
        else:
            return self.sum_Q / self.N


class MCTSResult(NamedTuple, Generic[State, Action]):
    last_state: State
    trace: list[tuple[State, Optional[Action]]] = None


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

    def __call__(self,
                 world_model: WorldModel,
                 search_config: SearchConfig,
                 output_trace=False,
                 retrieve_ans=None):
        root, chosen_leaf = self.mcts_search(init_state=world_model.init_state())
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

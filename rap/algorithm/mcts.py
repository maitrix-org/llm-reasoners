from typing import Generic

from ..rap import SearchAlgorithm, WorldModel, RAPAgent, SearchConfig, State, Action

class MCTSNode(Generic[State, Action]):
    def __init__(self):
        self.Q = self.N = 0
        self.action: Action | None = None
        self.state: State = None
        self.parent = None

'''
class MCTS(SearchAlgorithm, Generic[State, Action]):
    def __init__(self, world_model: WorldModel[State, Action], reward_model: RewardModel[State, Action], aggregation=False):
        self.world_model = world_model
        self.reward_model = reward_model
        self.aggregation = aggregation

    def mcts_search(self, init_state: State) -> tuple[MCTSNode, MCTSNode]:
        # TODO: generate init_state with world_model
        # TODO: implement MCTS
        raise NotImplementedError

    def __call__(self, init_state: State, output_trace: bool = False) -> State | list[tuple[Action | None, State]]:
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


class MCTSAggregation(MCTS[State, Action]):
    def __call__(self, init_state: State, output_trace: bool = False) -> State | list[tuple[Action, State]]:
        # TODO: implement aggregate
        pass
'''
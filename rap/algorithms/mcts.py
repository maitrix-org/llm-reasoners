from typing import Generic

from ..rap import SearchAlgorithm, WorldModel, RewardModel, S, A

class MCTSNode(Generic[S, A]):
    def __init__(self):
        self.Q = self.N = 0
        self.action: A | None = None
        self.state: S = None
        self.parent = None


class MCTS(SearchAlgorithm, Generic[S, A]):
    def __init__(self, world_model: WorldModel[S, A], reward_model: RewardModel[S, A], aggregation=False):
        self.world_model = world_model
        self.reward_model = reward_model
        self.aggregation = aggregation

    def mcts_search(self, init_state: S) -> tuple[MCTSNode, MCTSNode]:
        # TODO: implement MCTS
        raise NotImplementedError

    def __call__(self, init_state: S, output_trace: bool = False) -> S | list[tuple[A | None, S]]:
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


class MCTSAggregation(MCTS[S, A]):
    def __call__(self, init_state: S, output_trace: bool = False) -> S | list[tuple[A, S]]:
        # TODO: implement aggregate
        pass

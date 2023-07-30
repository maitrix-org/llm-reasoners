import json
from typing import Sequence, Union

from reasoners.algorithm import MCTSNode, MCTSResult
from reasoners.visualization.tree_snapshot import NodeId, EdgeId, TreeSnapshot, NodeData, EdgeData


class TreeLogEncoder(json.JSONEncoder):
    def default(self, o):
        from numpy import float32

        if isinstance(o, TreeSnapshot.Node):
            return o.__dict__
        elif isinstance(o, TreeSnapshot.Edge):
            return o.__dict__
        elif isinstance(o, TreeSnapshot):
            return o.__dict__()
        elif isinstance(o, float32):
            return float(o)
        if isinstance(o, TreeLog):
            return {"logs": list(o)}
        else:
            return super().default(o)


class TreeLog:
    def __init__(self, tree_snapshots: Sequence[TreeSnapshot]) -> None:
        self._tree_snapshots = tree_snapshots

    def __getitem__(self, item):
        return self._tree_snapshots[item]

    def __iter__(self):
        return iter(self._tree_snapshots)

    def __len__(self):
        return len(self._tree_snapshots)

    def __str__(self):
        return json.dumps(self, cls=TreeLogEncoder, indent=2)

    @classmethod
    def from_mcts_results(cls, mcts_results: MCTSResult, node_data_factory: callable = None,
                          edge_data_factory: callable = None) -> 'TreeLog':

        def get_reward_details(n: MCTSNode) -> Union[dict, None]:
            if hasattr(n, "reward_details"):
                return n.reward_details
            return n.fast_reward_details if hasattr(n, "fast_reward_details") else None

        def default_node_data_factory(n: MCTSNode) -> NodeData:
            return NodeData(n.state._asdict() if n.state else {})

        def default_edge_data_factory(n: MCTSNode) -> EdgeData:
            return EdgeData({"Q": n.Q, "reward": n.reward, **get_reward_details(n)})

        node_data_factory = node_data_factory or default_node_data_factory
        edge_data_factory = edge_data_factory or default_edge_data_factory

        snapshots = []

        def all_nodes(node: MCTSNode):
            node_id = NodeId(node.id)

            nodes[node_id] = TreeSnapshot.Node(node_id, node_data_factory(node))
            if node.children is None:
                return
            for child in node.children:
                edge_id = EdgeId(len(edges))
                edges.append(TreeSnapshot.Edge(edge_id, node.id, child.id, edge_data_factory(child)))
                all_nodes(child)

        if mcts_results.tree_state_after_each_iter is None:
            tree_states = [mcts_results.tree_state]
        else:
            tree_states = mcts_results.tree_state_after_each_iter
        for step in range(len(tree_states)):
            edges = []
            nodes = {}

            root = mcts_results.tree_state_after_each_iter[step]
            all_nodes(root)
            tree = TreeSnapshot(list(nodes.values()), edges)

            # select edges with highest Q value
            for node in tree.nodes.values():
                if node.selected_edge is None and tree.children(node.id):
                    node.selected_edge = max(
                        tree.out_edges(node.id),
                        key=lambda edge: edge.data.get("Q", -float("inf"))
                    ).id

            snapshots.append(tree)

        return cls(snapshots)

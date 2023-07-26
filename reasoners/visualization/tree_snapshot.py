from collections import defaultdict
from dataclasses import dataclass
from typing import NewType, Optional, Collection

NodeId = NewType("NodeId", int)
EdgeId = NewType("EdgeId", int)
NodeData = NewType("NodeData", dict)
EdgeData = NewType("EdgeData", dict)


class TreeSnapshot:
    @dataclass
    class Node:
        id: NodeId
        data: NodeData
        selected_edge: Optional[EdgeId] = None

    @dataclass
    class Edge:
        id: EdgeId
        source: NodeId
        target: NodeId
        data: EdgeData

    def __init__(self, nodes: Collection[Node], edges: Collection[Edge]) -> None:
        self.nodes: dict[NodeId, TreeSnapshot.Node] = {node.id: node for node in nodes}
        self.edges: dict[EdgeId, TreeSnapshot.Edge] = {edge.id: edge for edge in edges}
        self._parent = {}
        self._children: dict[NodeId, set[NodeId]] = defaultdict(set)

        for edge in edges:
            self._parent[edge.target] = edge.source
            self._children[edge.source].add(edge.target)

        assert len(self._parent) == len(self.nodes) - 1
        assert self._connected()

    def _connected(self) -> bool:
        visited = set()
        queue = [next(iter(self.nodes))]
        while queue:
            node = queue.pop()
            visited.add(node)
            queue.extend(self._children[node] - visited)
        return len(visited) == len(self.nodes)

    def node(self, node_id: NodeId) -> Node:
        return self.nodes[node_id]

    def edge(self, edge_id: EdgeId) -> Edge:
        return self.edges[edge_id]

    def out_edges(self, node_id: NodeId) -> Collection[Edge]:
        return [self.edge(edge_id) for edge_id in self.edges if self.edge(edge_id).source == node_id]

    def in_edges(self, node_id: NodeId) -> Collection[Edge]:
        return [self.edge(edge_id) for edge_id in self.edges if self.edge(edge_id).target == node_id]

    def parent(self, node_id: NodeId) -> NodeId:
        return self._parent[node_id]

    def children(self, node_id: NodeId) -> Collection[NodeId]:
        return self._children[node_id]

    def __dict__(self):
        return {
            "nodes": self.nodes,
            "edges": self.edges,
        }

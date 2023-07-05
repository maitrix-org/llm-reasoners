import pickle
from ..algorithm.mcts import MCTSNode

class Tree():
    def __init__(self,
                 nodes: list[dict],
                 edges: list[tuple[int, int, dict]]) -> None:
        self.nodes = nodes
        self.edges = edges
        self.parent = {}
        self.children = {}
        for i, j in edges:
            self.parent[j] = i
            self.children.setdefault(i, []).append(j)

    def get_node(self, node_id: int) -> dict:
        return self.nodes[node_id]
    
    def get_edge(self, edge_id: int) -> tuple[int, int]:
        return self.edges[edge_id]
    
    def get_out_edge_id(self, node_id: int) -> list[int]:
        return [i for i, j in enumerate(self.edges) if j[0] == node_id]
    
    def get_in_edge_id(self, node_id: int) -> list[int]:
        return [i for i, j in enumerate(self.edges) if j[1] == node_id]

    def get_parent(self, node_id: int) -> int:
        return self.parent[node_id]
    
    def get_children(self, node_id: int) -> list[int]:
        return self.children[node_id]
    

class TreeLog():
    def __init__(self, tree_snapshots: list[Tree]) -> None:
        self.tree_snapshots = tree_snapshots

    def get_tree(self, time_step: int) -> Tree:
        return self.tree_snapshots[time_step]

    @classmethod
    def from_MCTSResults(cls,
                         path: str,
                         node_translator: callable[[MCTSNode], dict],
                         edge_translator: callable[[MCTSNode, MCTSNode], dict]) -> 'TreeLog':
        '''load tree log from MCTSResults
        
        :param path: path to MCTSResults
        :param node_translator: function to translate MCTSNode to dictionary to be stored in TreeLog
        :param edge_translator: function to translate the incoming edge to a MCTSNode to a dictionary to be stored in TreeLog'''
        with open(path, 'rb') as f:
            results = pickle.load(f)
        
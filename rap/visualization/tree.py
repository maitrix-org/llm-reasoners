import pickle
from ..algorithm.mcts import MCTSNode
from typing import Callable
import numpy as np
import textwrap
import json

class Tree():
    def __init__(self,
                 nodes: list[dict],
                 edges: list[tuple[int, int, dict]]) -> None:
        """Tree data structure for visualization

        :param nodes: list of nodes, each node is a dictionary of information you want to show
        :param edges: list of edges, each edge is a tuple of (parent_id, child_id, edge_info). edge_info is a dictionary of information you want to show
        """
        self.nodes = nodes
        self.edges = edges
        self.n_nodes = len(nodes)
        self.n_edges = len(edges)
        self.parent = {}
        self.children = {}
        for i, j, info in edges:
            self.parent[j] = i
            self.children.setdefault(i, []).append(j)

    def get_node(self, node_id: int) -> dict:
        return self.nodes[node_id]
    
    def get_edge(self, edge_id: int) -> tuple[int, int]:
        return self.edges[edge_id]
    
    def get_out_edge_ids(self, node_id: int) -> list[int]:
        return [i for i, j in enumerate(self.edges) if j[0] == node_id]
    
    def get_in_edge_id(self, node_id: int) -> list[int]:
        return [i for i, j in enumerate(self.edges) if j[1] == node_id]

    def get_parent(self, node_id: int) -> int:
        return self.parent[node_id]
    
    def get_children(self, node_id: int) -> list[int]:
        return self.children[node_id]
    
    def visualize(self):
        try:
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("Please install plotly for visualization") from exc
        
        Xn, Yn = np.zeros(self.n_nodes), np.zeros(self.n_nodes)
        leaves_cnt = 0
        for i in range(self.n_nodes):
            if i not in self.parent:
                Yn[i] = 0
            else:
                Yn[i] = Yn[self.parent[i]] - 1
        for i in range(self.n_nodes - 1, -1, -1):
            if i not in self.children:
                Xn[i] = leaves_cnt
                leaves_cnt += 1
            else:
                Xn[i] = (np.max([Xn[j] for j in self.children[i]]) + np.min([Xn[j] for j in self.children[i]])) / 2
        Xe = []
        Ye = []
        for edge in self.edges:
            Xe+=[Xn[edge[0]],Xn[edge[1]], None]
            Ye+=[Yn[edge[0]],Yn[edge[1]], None]

        v_label = [str(i) for i in range(self.n_nodes)]
        labels = []
        for i in range(self.n_nodes):
            edge_info = {} if i not in self.parent else self.edges[self.get_in_edge_id(i)[0]][-1]
            edge_info = {str(k): str(v) for k, v in edge_info.items()}
            node_info = self.nodes[i]
            node_info = {str(k): str(v) for k, v in node_info.items()}
            s = json.dumps({**node_info, **edge_info}, indent=4)
            s = "\n".join([textwrap.fill(line) for line in s.split("\n")]).replace("\n", "<br>")
            labels.append(s)
        font_size = 10
        font_color = 'rgb(250,250,250)'
        
        L = len(Xn)
        annotations = []
        for k in range(L):
            annotations.append(
                dict(
                    text=v_label[k],
                    x=Xn[k], y=Yn[k],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Xe,
                        y=Ye,
                        mode='lines',
                        line=dict(color='rgb(210,210,210)', width=1),
                        hoverinfo='none'
                        ))
        fig.add_trace(go.Scatter(x=Xn,
                        y=Yn,
                        mode='markers',
                        name='bla',
                        marker=dict(symbol='circle-dot',
                                        size=18,
                                        color='#6175c1',
                                        line=dict(color='rgb(50,50,50)', width=1)
                                        ),
                        text=labels,
                        hoverinfo='text',
                        opacity=0.8
                        ))

        axis = dict(showline=False,
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    )

        fig.update_layout(
                    annotations=annotations,
                    font_size=12,
                    showlegend=False,
                    xaxis=axis,
                    yaxis=axis,
                    margin=dict(l=10, r=10, b=10, t=10),
                    hovermode='closest',
                    plot_bgcolor='rgb(248,248,248)'
                    )
        fig.show()
        return self.nodes, self.edges


class TreeLog():

    def __init__(self, tree_snapshots: list[Tree]) -> None:
        """Data structure to store tree snapshots as the log

        :param tree_snapshots: list of tree snapshots. The index of the list is the time step. Note that the node with the same id in different time steps should be the same. This will enable the dynamic visualization of the tree in the frontend (to be available).
        """

        self.tree_snapshots = tree_snapshots

    def get_tree(self, time_step: int) -> Tree:
        return self.tree_snapshots[time_step]

    @classmethod
    def from_MCTSResults(cls,
                         path: str,
                         node_translator: Callable[[MCTSNode], dict],
                         edge_translator: Callable[[MCTSNode, MCTSNode], dict]) -> 'TreeLog':
        '''load tree log from MCTSResults
        
        :param path: path to MCTSResults
        :param node_translator: function to translate a `MCTSNode` to a dictionary to be stored in the Tree
        :param edge_translator: function to translate the incoming edge to a `MCTSNode` to a dictionary to be stored in TreeLog'''
        snapshots = []
        def get_all_nodes(node, edges, nodes):
            # print(int(node.id))
            # print(node)
            # print(node_translator)
            # print(nodes)
            nodes[int(node.id)] = {"a": "a"}
            # print(nodes[int(node.id)])
            nodes[int(node.id)] = node_translator(node)
            if node.children is None:
                return
            for child in node.children:
                edges.append((node.id, child.id, edge_translator(child)))
                get_all_nodes(child, edges, nodes)

        with open(path, 'rb') as f:
            obj = pickle.load(f)

        n_trajs = len(obj.tree_state_after_each_iter)

        for step in range(n_trajs):
            edges = []
            nodes = {}
            root = obj.tree_state_after_each_iter[step]
            get_all_nodes(root, edges, nodes)
            n_nodes = len(nodes)
            try:
                nodes = [nodes[i] for i in range(n_nodes)]
            except KeyError as exc:
                raise ValueError(f'Node id must be consecutive from 0 to {n_nodes - 1}') from exc
            tree = Tree(nodes, edges)
            snapshots.append(tree)
        return cls(snapshots)